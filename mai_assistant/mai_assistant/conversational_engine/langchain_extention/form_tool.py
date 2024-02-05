import operator
from typing import (Annotated, Any, Dict, Optional, Sequence, Type, TypedDict, Union)

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ValidationError
from enum import Enum

from .tool_dummy_payload import ToolDummyPayload

class FormToolOutput(BaseModel):
    """
    Represents a form tool output.
    The output is returned as str.
    Any other kwarg is returned in the state_update dict
    """

    output: str
    state_update: Optional[Dict[str, Any]] = None

    def __init__(self, output: str, **kwargs):
        super().__init__(output=output)
        self.state_update = kwargs
    
class FormToolState(Enum):
    DUMMY = "DUMMY"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"


class FormTool(StructuredTool):
    """
    FormTool methods should take context as AgentState, but this creates circular references
    So we use BaseModel instead
    """
    form: BaseModel = None
    args_schema_: Optional[Type[BaseModel]] = None
    name_: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.form = self.args_schema()
        self.args_schema_ = None
        self.name_ = None
        
    @property
    def state(self) -> FormToolState:
        if self.is_dummy_state():
            return FormToolState.DUMMY
        elif self.is_form_complete():
            return FormToolState.COMPLETE
        else:
            return FormToolState.ACTIVE

    def set_dummy_state(self):
        # Guard so that we don't overwrite the original args_schema if
        # set_dummy_state is called multiple times
        if not self.is_dummy_state():
            self.name_ = self.name
            self.name = f"{self.name}Initiator"
            self.args_schema_ = self.args_schema
            self.args_schema = ToolDummyPayload

    def unset_dummy_state(self):
        self.args_schema = self.args_schema_
        self.name = self.name_

    def is_dummy_state(self):
        return self.args_schema == ToolDummyPayload

    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        match self.state:
            case FormToolState.DUMMY:
                self.unset_dummy_state()
                return FormToolOutput(
                    output=f"Starting intent {self.name}. Ask the user for the first field.",
                    active_form_tool = self
                )
            case FormToolState.ACTIVE:
                self._update_form(**kwargs)
                return FormToolOutput(
                    active_form_tool=self,
                    output="Form updated with the provided information. Ask the user for the next field."
                )
            case FormToolState.COMPLETE:
                result = self._run_when_complete(**kwargs)
                # if no exception is raised, the form is complete and the tool is
                # done, so reset the active form tool
                return FormToolOutput(
                    active_form_tool=None,
                    output=result
                )
        
    def is_form_complete(self) -> bool:
        """
        The default implementation checks if all values except optional ones are set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            # if field_info.is_required():
            if not getattr(self.form, field_name):
                return False
        return True

    # TODO: @abstractmethod
    def _run_when_complete(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Should raise an exception if something goes wrong.
        The message should describe the error and will be sent back to the agent to try to fix it.
        """

    def _update_form(self, **kwargs):
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
        message = ""
        match self.state:
            case FormToolState.DUMMY:
                message = f"Starting {self.name_}"
            case FormToolState.ACTIVE:
                message = f"Updating form for {self.name}"
            case FormToolState.COMPLETE:
                message = f"Completed {self.name}"
        return message

    def get_information_to_collect(self) -> str:
        return str(list(self.args.keys()))


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: Annotated[Optional[list[BaseMessage]], operator.setitem]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Annotated[Optional[Union[AgentAction,
                                            AgentFinish, None]], operator.setitem]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[Optional[list[tuple[AgentAction,
                                                      FunctionMessage]]], operator.add]
    error: Annotated[Optional[str], operator.setitem]

    active_form_tool: Annotated[Optional[FormTool], operator.setitem]


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


def filter_active_tools(
    tools: Sequence[BaseTool],
    context: AgentState
):
    """
    Form tools are replaced by their activators if they are not active.
    """
    if context.get("active_form_tool"):
        # If a form_tool is active, it is the only form tool available
        base_tools = [tool for tool in tools if not isinstance(tool, FormTool)]
        tools = [
            *base_tools,
            context.get("active_form_tool"),
            ContextReset(context=context)
        ]
    else:
        # When a form tool is not active, change the args_schema to DummyPayload
        # so that the model calls the tool without asking the user to input the fields
        for tool in tools:
            if isinstance(tool, FormTool):
                tool.set_dummy_state()
    return tools