import copy
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    _get_python_function_name,
)
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel, ValidationError


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


def format_tool_to_gigachat_function(tool: BaseTool) -> GigaFunctionDescription:
    """Format tool into the GigaChat function API."""
    if not tool.description or tool.description == "":
        raise Exception(
            "Incorrect function or tool description. Description is required."
        )
    tool_schema = tool.args_schema
    if tool.tool_call_schema:
        tool_schema = tool.tool_call_schema
    if tool_schema:
        return convert_pydantic_to_gigachat_function(
            tool_schema,
            name=tool.name,
            description=tool.description,
            return_model=tool.return_schema,
            few_shot_examples=tool.few_shot_examples,
        )
    else:
        if tool.return_schema:
            return_schema = _convert_return_schema(tool.return_schema)
        else:
            return_schema = None

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"properties": {}, "type": "object"},
            "few_shot_examples": tool.few_shot_examples,
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
    _return_schema = tools.create_return_schema_from_function(func_name, function)
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
