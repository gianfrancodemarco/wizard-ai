import operator
from typing import Annotated, Any, Dict, Optional, Type, TypedDict, Union

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .tool_dummy_payload import ToolDummyPayload


class FormTool(BaseTool):
    """
    FormTool methods should take context as AgentState, but this creates circular references
    So we use BaseModel instead
    """

    def is_form_complete(
        self,
        context: Optional[BaseModel],
    ) -> bool:
        """
        The default implementation checks if all values except optional ones are set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            if field_info.is_required():
                if not getattr(context.form, field_name):
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
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, FunctionMessage]], operator.add]
    
    active_form_tool: Optional[FormTool] = None
    form: BaseModel = None

class FormToolActivator(BaseTool):
    args_schema: Type[BaseModel] = ToolDummyPayload
    form_tool_class: Type[FormTool]
    form_tool: FormTool

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        state: Optional[AgentState] = None,
        **kwargs
    ) -> str:
        return f"Entered in {self.form_tool.name} context"

    def _parse_input(self, tool_input: str | Dict) -> str | Dict[str, Any]:
        """FormToolActivator shouldn't have any input, so we ovveride the default implementation."""
        return {}

    def get_tool_start_message(self, input: dict) -> str:
        return f"Starting form {self.form_tool.name}"
