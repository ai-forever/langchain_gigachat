import copy
import functools
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_type_hints,
)

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseCumulativeTransformOutputParser,
    BaseGenerationOutputParser,
    BaseOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    _get_python_function_name,
)
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel, ValidationError, root_validator


class GigaFunctionDescription(FunctionDescription):
    """The parameters of the function."""

    return_parameters: Optional[dict]
    """The result settings of the function."""
    few_shot_examples: Optional[list]
    """The examples of the function."""


class JsonOutputToolsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse tools from OpenAI response."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
    """
    return_id: bool = False
    """Whether to return the tool call id."""
    first_tool_only: bool = False
    """Whether to return only the first tool call.

    If False, the result will be a list of tool calls, or an empty list 
    if no tool calls are found.

    If true, and multiple tool calls are found, only the first one will be returned,
    and the other tool calls will be ignored. 
    If no tool calls are found, None will be returned. 
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            tool_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError:
            return []

        final_tools = [{"type": tool_call["name"], "args": tool_call["arguments"]}]
        if self.first_tool_only:
            return final_tools[0] if final_tools else None
        return final_tools

    def parse(self, text: str) -> Any:
        raise NotImplementedError()


class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    key_name: str
    """The type of tools to return."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        parsed_result = super().parse_result(result, partial=partial)

        if self.first_tool_only:
            single_result = (
                parsed_result
                if parsed_result and parsed_result["type"] == self.key_name
                else None
            )
            if self.return_id:
                return single_result
            elif single_result:
                return single_result["args"]
            else:
                return None
        parsed_result = [res for res in parsed_result if res["type"] == self.key_name]
        if not self.return_id:
            parsed_result = [res["args"] for res in parsed_result]
        return parsed_result


class PydanticToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    tools: List[Type[BaseModel]]

    # TODO: Support more granular streaming of objects. Currently only streams once all
    # Pydantic object fields are present.
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        json_results = super().parse_result(result, partial=partial)
        if not json_results:
            return None if self.first_tool_only else []

        json_results = [json_results] if self.first_tool_only else json_results
        name_dict = {tool.__name__: tool for tool in self.tools}
        pydantic_objects = []
        for res in json_results:
            try:
                if not isinstance(res["args"], dict):
                    raise ValueError(
                        f"Tool arguments must be specified as a dict, received: "
                        f"{res['args']}"
                    )
                pydantic_objects.append(name_dict[res["type"]](**res["args"]))
            except (ValidationError, ValueError) as e:
                if partial:
                    continue
                else:
                    raise e
        if self.first_tool_only:
            return pydantic_objects[0] if pydantic_objects else None
        else:
            return pydantic_objects


def flatten_all_of(schema: Any) -> Any:
    """GigaChat не поддерживает allOf/anyOf, поэтому правим вложенную структуру"""
    if isinstance(schema, dict):
        obj_out: Any = {}
        for k, v in schema.items():
            if k == "title":
                continue
            if k == "allOf":
                obj = flatten_all_of(v[0])
                outer_description = schema.get("description")
                obj_out = {**obj_out, **obj}
                if outer_description:
                    # Внешнее описания приоритетнее внутреннего для ref
                    obj_out["description"] = outer_description
            elif isinstance(v, (list, dict)):
                obj_out[k] = flatten_all_of(v)
            else:
                obj_out[k] = v
        return obj_out
    elif isinstance(schema, list):
        return [flatten_all_of(el) for el in schema]
    else:
        return schema


def _convert_return_schema(return_model: Type[BaseModel]) -> Dict[str, Any]:
    return_schema = dereference_refs(return_model.schema())
    return_schema.pop("definitions", None)
    return_schema.pop("title", None)

    for key in return_schema["properties"]:
        if "type" not in return_schema["properties"][key]:
            return_schema["properties"][key]["type"] = "object"
        if "description" not in return_schema["properties"][key]:
            return_schema["properties"][key]["description"] = ""

    return return_schema


"""TODO: Support GigaBaseTool with return schema and few shot! """


def format_tool_to_gigachat_function(tool: BaseTool) -> GigaFunctionDescription:
    """Format tool into the GigaChat function API."""
    if not tool.description or tool.description == "":
        raise Exception(
            "Incorrect function or tool description. Description is required."
        )
    tool_schema = tool.args_schema
    if tool.tool_call_schema:
        tool_schema = tool.tool_call_schema

    if hasattr(tool, "return_schema") and tool.return_schema:
        return_schema = _convert_return_schema(tool.return_schema)
    else:
        return_schema = None

    if hasattr(tool, "few_shot_examples") and tool.few_shot_examples:
        few_shot_examples = tool.few_shot_examples
    else:
        few_shot_examples = None

    if tool_schema:
        return convert_pydantic_to_gigachat_function(
            tool_schema,
            name=tool.name,
            description=tool.description,
            return_model=return_schema,
            few_shot_examples=few_shot_examples,
        )
    else:
        if hasattr(tool, "return_schema") and tool.return_schema:
            return_schema = _convert_return_schema(tool.return_schema)
        else:
            return_schema = None

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"properties": {}, "type": "object"},
            "few_shot_examples": few_shot_examples,
            "return_parameters": return_schema,
        }


def convert_pydantic_to_gigachat_function(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    return_model: Optional[Type[BaseModel]] = None,
    few_shot_examples: Optional[List[dict]] = None,
) -> GigaFunctionDescription:
    """Converts a Pydantic model to a function description for the GigaChat API."""
    schema = dereference_refs(model.schema())
    schema.pop("definitions", None)
    title = schema.pop("title", None)
    if "properties" in schema:
        for key in schema["properties"]:
            if "type" not in schema["properties"][key]:
                schema["properties"][key]["type"] = "object"
            if "description" not in schema["properties"][key]:
                schema["properties"][key]["description"] = ""

    if return_model:
        return_schema = _convert_return_schema(return_model)
    else:
        return_schema = None

    description = description or schema.get("description", None)
    if not description or description == "":
        raise ValueError(
            "Incorrect function or tool description. Description is required."
        )

    return GigaFunctionDescription(
        name=name or title,
        description=description,
        parameters=schema,
        return_parameters=return_schema,
        few_shot_examples=few_shot_examples,
    )


def _get_type_hints(func: Callable) -> Optional[Dict[str, Type]]:
    if isinstance(func, functools.partial):
        func = func.func
    try:
        return get_type_hints(func)
    except Exception:
        return None


def create_return_schema_from_function(
    model_name: str, func: Callable
) -> Optional[Type[BaseModel]]:
    return_type = get_type_hints(func).get("return", Any)
    if (
        return_type is not str
        and return_type is not int
        and return_type is not float
        and return_type is not None
    ):
        try:
            if isinstance(return_type, type) and issubclass(return_type, BaseModel):
                return return_type
        except TypeError:  # It's normal for testing
            return None

    return None


def convert_python_function_to_gigachat_function(
    function: Callable,
) -> GigaFunctionDescription:
    """Convert a Python function to an GigaChat function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.

    Args:
        function: The Python function to convert.

    Returns:
        The GigaChat function description.
    """
    from langchain_core import tools

    func_name = _get_python_function_name(function)
    model = tools.create_schema_from_function(
        func_name,
        function,
        filter_args=(),
        parse_docstring=True,
        error_on_invalid_docstring=False,
        include_injected=False,
    )
    _return_schema = create_return_schema_from_function(func_name, function)
    return convert_pydantic_to_gigachat_function(
        model, name=func_name, return_model=_return_schema, description=model.__doc__
    )


def convert_to_gigachat_function(
    function: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI function.

    Args:
        function: Either a dictionary, a pydantic.BaseModel class, or a Python function.
            If a dictionary is passed in, it is assumed to already be a valid OpenAI
            function.

    Returns:
        A dict version of the passed in function which is compatible with the
            OpenAI function-calling API.
    """
    from langchain_core.tools import BaseTool

    if isinstance(function, dict):
        return function
    elif isinstance(function, type) and issubclass(function, BaseModel):
        function = cast(Dict, convert_pydantic_to_gigachat_function(function))
    elif isinstance(function, BaseTool):
        function = cast(Dict, format_tool_to_gigachat_function(function))
    elif callable(function):
        function = cast(Dict, convert_python_function_to_gigachat_function(function))
    else:
        raise ValueError(
            f"Unsupported function type {type(function)}. Functions must be passed in"
            f" as Dict, pydantic.BaseModel, or Callable."
        )
    return flatten_all_of(function)


def convert_to_gigachat_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an GigaChat tool.

    Args:
        tool: Either a dictionary, a pydantic.BaseModel class, Python function, or
            BaseTool. If a dictionary is passed in, it is assumed to already be a valid
            GigaChat tool, GigaChat function,
            or a JSON schema with top-level 'title' and
            'description' keys specified.

    Returns:
        A dict version of the passed in tool which is compatible with the
            GigaChat tool-calling API.
    """
    if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
        return tool
    function = convert_to_gigachat_function(tool)
    return {"type": "function", "function": function}


def create_gigachat_fn_runnable(
    functions: Sequence[Type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    enforce_single_function_usage: bool = True,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **llm_kwargs: Any,
) -> Runnable:
    """Create a runnable sequence that uses OpenAI functions.

    Args:
        functions: A sequence of either dictionaries, pydantic.BaseModels classes, or
            Python functions. If dictionaries are passed in, they are assumed to
            already be a valid OpenAI functions. If only a single
            function is passed in, then it will be enforced that the model use that
            function. pydantic.BaseModels and Python functions should have docstrings
            describing what the function does. For best results, pydantic.BaseModels
            should have descriptions of the parameters and Python functions should have
            Google Python style args descriptions in the docstring. Additionally,
            Python functions should only use primitive types (str, int, float, bool) or
            pydantic.BaseModels for arguments.
        llm: Language model to use, assumed to support the OpenAI function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        enforce_single_function_usage: only used if a single function is passed in. If
            True, then the model will be forced to use the given function. If False,
            then the model will be given the option to use the given function or not.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON. If multiple functions are
            passed in and they are not pydantic.BaseModels, the chain output will
            include both the name of the function that was returned and the arguments
            to pass to the function.
        **llm_kwargs: Additional named arguments to pass to the language model.

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.structured_output import create_openai_fn_runnable
                from langchain_community.chat_models import GigaChat
                from langchain_core.pydantic_v1 import BaseModel, Field


                class RecordPerson(BaseModel):
                    '''Record some identifying information about a person.'''

                    name: str = Field(..., description="The person's name")
                    age: int = Field(..., description="The person's age")
                    fav_food: Optional[str] = Field(None, description="The person's favorite food")


                class RecordDog(BaseModel):
                    '''Record some identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = GigaChat(model="GigaChat-Pro")
                structured_llm = create_gigachat_fn_runnable([RecordPerson, RecordDog], llm)
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken)
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    g_functions = [convert_to_gigachat_function(f) for f in functions]
    llm_kwargs_: Dict[str, Any] = {"functions": g_functions, **llm_kwargs}
    if len(g_functions) == 1 and enforce_single_function_usage:
        llm_kwargs_["function_call"] = {"name": g_functions[0]["name"]}
    output_parser = output_parser or get_gigachat_output_parser(functions)
    if prompt:
        return prompt | llm.bind(**llm_kwargs_) | output_parser
    else:
        return llm.bind(**llm_kwargs_) | output_parser


def create_structured_output_runnable(
    output_schema: Union[Type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    enforce_function_usage: bool = True,
    **kwargs: Any,
) -> Runnable:
    """Create a runnable for extracting structured outputs.

    Args:
        output_schema: Either a dictionary or pydantic.BaseModel class. If a dictionary
            is passed in, it's assumed to already be a valid JsonSchema.
            For best results, pydantic.BaseModels should have docstrings describing what
            the schema represents and descriptions for the parameters.
        llm: Language model to use. Assumed to support the OpenAI function-calling API
            if mode is 'openai-function'. Assumed to support OpenAI response_format
            parameter if mode is 'openai-json'.
        prompt: BasePromptTemplate to pass to the model. If mode is 'openai-json' and
            prompt has input variable 'output_schema' then the given output_schema
            will be converted to a JsonSchema and inserted in the prompt.
        output_parser: Output parser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModel is passed
            in, then the OutputParser will try to parse outputs using the pydantic
            class. Otherwise model outputs will be parsed as JSON.
        enforce_function_usage: If True, then the model will be forced to use the given
            output schema. If False, then the model can elect whether to use the output
            schema.
        **kwargs: Additional named arguments.

    Returns:
        A runnable sequence that will return a structured output(s) matching the given
            output_schema.

    OpenAI functions example (mode="openai-functions"):
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_community.chat_models import GigaChat
                from langchain_core.pydantic_v1 import BaseModel, Field


                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = GigaChat(model="GigaChat-Pro", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm)
                structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
                # -> Dog(name="Harry", color="brown", fav_food="chicken")

    Gigachat functions with prompt example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains import create_structured_output_runnable
                from langchain_community.chat_models import GigaChat
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.pydantic_v1 import BaseModel, Field


                class Dog(BaseModel):
                    '''Identifying information about a dog.'''

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = GigaChat(model="GigaChat-Pro", temperature=0)
                structured_llm = create_structured_output_runnable(Dog, llm)
                system = '''Extract information about any dogs mentioned in the user input.'''
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ("human", "{input}")]
                )
                chain = prompt | structured_llm
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> Dog(name="Harry", color="brown", fav_food="chicken")
    """  # noqa: E501
    # for backwards compatibility
    force_function_usage = kwargs.get(
        "enforce_single_function_usage", enforce_function_usage
    )

    return _create_gigachat_functions_structured_output_runnable(
        output_schema,
        llm,
        prompt=prompt,
        output_parser=output_parser,
        enforce_single_function_usage=force_function_usage,
        **kwargs,  # llm-specific kwargs
    )


def get_gigachat_output_parser(
    functions: Sequence[Type[BaseModel]],
) -> Union[BaseOutputParser, BaseGenerationOutputParser]:
    """Get the appropriate function output parser given the user functions.

    Args:
        functions: Sequence where element is a dictionary, a pydantic.BaseModel class,
            or a Python function. If a dictionary is passed in, it is assumed to
            already be a valid GigaChat function.

    Returns:
        A PydanticOutputFunctionsParser if functions are Pydantic classes, otherwise
            a JsonOutputFunctionsParser. If there's only one function and it is
            not a Pydantic class, then the output parser will automatically extract
            only the function arguments and not the function name.
    """
    if len(functions) > 1:
        pydantic_schema: Union[Dict, Type[BaseModel]] = {
            convert_to_gigachat_function(fn)["name"]: fn for fn in functions
        }
    else:
        pydantic_schema = functions[0]
    output_parser: Union[BaseOutputParser, BaseGenerationOutputParser] = (
        PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    )
    return output_parser


def _create_gigachat_functions_structured_output_runnable(
    output_schema: Union[Type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]] = None,
    **llm_kwargs: Any,
) -> Runnable:
    class _OutputFormatter(BaseModel):
        """Output formatter. Всегда используй чтобы выдать ответ"""  # noqa: E501

        output: output_schema  # type: ignore

    function = _OutputFormatter
    output_parser = output_parser or PydanticAttrOutputFunctionsParser(
        pydantic_schema=_OutputFormatter, attr_name="output"
    )
    return create_gigachat_fn_runnable(
        [function], llm, prompt=prompt, output_parser=output_parser, **llm_kwargs
    )


class OutputFunctionsParser(BaseGenerationOutputParser[Any]):
    """Parse an output that is one of sets of values."""

    args_only: bool = True
    """Whether to only return the arguments to the function call."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError as exc:
            raise OutputParserException(
                f"Could not parse function call: {exc}"
            ) from exc

        if self.args_only:
            return func_call["arguments"]
        return func_call


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """Parse an output as a pydantic object.

    This parser is used to parse the output of a ChatModel that uses
    OpenAI function format to invoke functions.

    The parser extracts the function call invocation and matches
    them to the pydantic schema provided.

    An exception will be raised if the function call does not match
    the provided schema.

    Example:

        ... code-block:: python

            message = AIMessage(
                content="This is a test message",
                additional_kwargs={
                    "function_call": {
                        "name": "cookie",
                        "arguments": json.dumps({"name": "value", "age": 10}),
                    }
                },
            )
            chat_generation = ChatGeneration(message=message)

            class Cookie(BaseModel):
                name: str
                age: int

            class Dog(BaseModel):
                species: str

            # Full output
            parser = PydanticOutputFunctionsParser(
                pydantic_schema={"cookie": Cookie, "dog": Dog}
            )
            result = parser.parse_result([chat_generation])
    """

    pydantic_schema: Union[Type[BaseModel], Dict[str, Type[BaseModel]]]
    """The pydantic schema to parse the output with.

    If multiple schemas are provided, then the function name will be used to
    determine which schema to use.
    """

    @root_validator(pre=True)
    def validate_schema(cls, values: Dict) -> Dict:
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = isinstance(schema, type) and issubclass(
                schema, BaseModel
            )
        elif values["args_only"] and isinstance(schema, Dict):
            raise ValueError(
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
        return values

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        _result = super().parse_result(result)
        if self.args_only:
            pydantic_args = self.pydantic_schema(**_result)  # type: ignore
        else:
            fn_name = _result.name
            _args = json.dumps(_result.arguments)
            pydantic_args = self.pydantic_schema[fn_name].parse_raw(_args)  # type: ignore  # noqa: E501
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """Parse an output as an attribute of a pydantic object."""

    attr_name: str
    """The name of the attribute to return."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
