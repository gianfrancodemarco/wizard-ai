from typing import Sequence

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

# from .context_reset import ContextReset
# from .context_update import ContextUpdate
# from .form_tool import AgentState, FormTool, FormToolActivator


# def filter_active_tools(
#     tools: Sequence[BaseTool],
#     context: AgentState
# ):
#     """
#     Form tools are replaced by their activators if they are not active.
#     """

#     base_tools = list(filter(lambda tool: not isinstance(
#         tool, FormToolActivator) and not isinstance(tool, FormTool), tools))

#     if context.get("active_form_tool") is None:
#         activator_tools = [
#             FormToolActivator(
#                 form_tool_class=tool.__class__,
#                 form_tool=tool,
#                 name=f"{tool.name}Activator",
#                 description=tool.description,
#                 context=context
#             )
#             for tool in tools
#             if isinstance(tool, FormTool)
#         ]
#         tools = [
#             *base_tools,
#             *activator_tools
#         ]
#     else:
#         # If a form_tool is active, remove the Activators and add the form
#         # tool and the context update tool
#         tools = [
#             # context.get("active_form_tool"),
#             *base_tools,
#             ContextUpdate(context=context),
#             ContextReset(context=context)
#         ]
#     return tools


def make_optional_model(original_model: BaseModel) -> BaseModel:
    """
    Takes a Pydantic model and returns a new model with all attributes optional.
    """
    optional_attributes = {attr_name: (
        attr_type, None) for attr_name, attr_type in original_model.__annotations__.items()}

    # Define a custom Pydantic model with optional attributes
    new_class_name = original_model.__name__ + 'Optional'
    OptionalModel = create_model(
        new_class_name,
        **optional_attributes,
        __base__=original_model
    )
    OptionalModel.model_config["validate_assignment"] = True

    return OptionalModel
