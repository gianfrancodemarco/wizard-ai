import operator
from typing import (Annotated, Any, Dict, Optional, Sequence, Type, TypedDict,
                    Union)

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel, ConfigDict
from langchain_core.tools import BaseTool, ToolException, SchemaAnnotationError
from pydantic import BaseModel, Field, ValidationError

from mai_assistant.conversational_engine.langchain_extention.intent_helpers import \
    make_optional_model
from abc import abstractmethod, ABC
from .tool_dummy_payload import ToolDummyPayload


class FormTool(BaseTool, ABC):
    """
    FormTool methods should take context as AgentState, but this creates circular references
    So we use BaseModel instead
    """

    form: BaseModel = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.form = self.args_schema()

    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:

        if self.is_form_complete():
            return self.run_when_complete(**kwargs)
        else:
            return self.update_form(**kwargs)

    def update_form(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self.form, key, value)
            except ValidationError as e:
                # build a string message with the error
                messages = []
                for error in e.errors():
                    messages.append(
                        f"Error at {error['loc'][0]}: {error['msg']}")
                message = "\n".join(messages)
                raise ToolException(message)
        return {
            "state_update": {
                "active_form_tool": self
            },
            "output": "Form updated",
        }

    #@abstractmethod
    def run_when_complete(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        pass

    def is_form_complete(self) -> bool:
        """
        The default implementation checks if all values except optional ones are set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            #if field_info.is_required():
            if not getattr(self.form, field_name):
                return False
        return True

    def get_next_field_to_collect(
        self,
        form: Optional[BaseModel],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        The default implementation returns the first field that is not set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            if not getattr(form, field_name):
                return field_name
        return None

    def get_tool_start_message(self, input: dict) -> str:
        return "Creating form\n"

    def get_information_to_collect(self) -> str:
        return str(list(self.args.keys()))


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: Annotated[list[BaseMessage], operator.setitem]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Annotated[Union[AgentAction,
                                   AgentFinish, None], operator.setitem]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction,
                                             FunctionMessage]], operator.add]
    error: Annotated[Optional[str], operator.setitem]

    active_form_tool: Annotated[Optional[FormTool], operator.setitem]
    #form: Annotated[Optional[BaseModel], operator.setitem]


class ContextReset(BaseTool):
    name = "ContextReset"
    description = """Call this tool when the user doesn't want to fill the form anymore."""
    args_schema: Type[BaseModel] = ToolDummyPayload

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return {
            "state_update": {
                "active_form_tool": None
            },
            "output": "Context reset. Form cleared. Ask the user what he wants to do next."
        }

# class ContextUpdatePayload(BaseModel):
#     model_config = ConfigDict(extra='allow')

# class ContextUpdate(BaseTool):
#     name = "ContextUpdate"
#     description = """Stores information given by the user in the form."""
#     args_schema: Type[BaseModel] = ContextUpdatePayload
#     handle_tool_error = True

#     context: Optional[AgentState] = None

#     def _run(self, *args: Any, **kwargs: Any) -> Any:
#         for key, value in kwargs['values'].items():
#             try:
#                 setattr(self.context.form, key, value)
#             except ValidationError as e:
#                 # build a string message with the error
#                 messages = []
#                 for error in e.errors():
#                     messages.append(
#                         f"Error at {error['loc'][0]}: {error['msg']}")
#                 message = "\n".join(messages)
#                 raise ToolException(message)
#         return {
#             "state_update": {
#                 "form": self.context.form
#             },
#             "output": "Form updated",
#         }


def filter_active_tools(
    tools: Sequence[BaseTool],
    context: AgentState
):
    """
    Form tools are replaced by their activators if they are not active.
    """

    base_tools = list(filter(lambda tool: not isinstance(
        tool, FormToolActivator) and not isinstance(tool, FormTool), tools))

    if context.get("active_form_tool") is None:
        activator_tools = [
            FormToolActivator(
                form_tool_class=tool.__class__,
                form_tool=tool,
                name=f"{tool.name}Activator",
                description=tool.description,
                context=context
            )
            for tool in tools
            if isinstance(tool, FormTool)
        ]
        tools = [
            *base_tools,
            *activator_tools
        ]
    else:
        # If a form_tool is active, remove the Activators and add the form
        # tool and the context update tool
        tools = [
            # context.get("active_form_tool"),
            *base_tools,
            context.get("active_form_tool"),
            # ContextUpdate(
            #     context=context
            # ),
            ContextReset(context=context)
        ]
    return tools


class FormToolActivator(BaseTool):
    args_schema: Type[BaseModel] = ToolDummyPayload
    form_tool_class: Type[FormTool]
    form_tool: FormTool
    context: AgentState

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        return {
            "state_update": {
                "active_form_tool": self.form_tool,
                #"form": make_optional_model(self.form_tool.args_schema)()
            },
            "output": f"Activating form {self.form_tool.name}"
        }

    def _parse_input(self, tool_input: str | Dict) -> str | Dict[str, Any]:
        """FormToolActivator shouldn't have any input, so we ovveride the default implementation."""
        return {}

    def get_tool_start_message(self, input: dict) -> str:
        return f"Starting form {self.form_tool.name}"
