import json
import operator
from abc import ABC, abstractmethod
from enum import Enum
from typing import (Annotated, Any, Dict, Optional, Sequence, Type, TypedDict,
                    Union)

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field, ValidationError, create_model


class IntentToolState(Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"

# We cannot pass directly the BaseModel class as args_schema as pydantic will raise errors,
# so we need to create a dummy class that inherits from BaseModel.


class IntentToolInactivePayload(BaseModel):
    pass


class IntentToolConfirmPayload(BaseModel):
    confirm: bool = Field(
        description="True if the user confirms the intent, False if not or wants to change something."
    )


class IntentToolOutcome(BaseModel):
    """
    Represents a form tool output.
    The output is returned as str.
    Any other kwarg is returned in the state_update dict
    """

    output: str
    state_update: Optional[Dict[str, Any]] = None
    return_direct: Optional[bool] = False

    def __init__(
        self,
        output: str,
        return_direct: bool = False,
        **kwargs
    ):
        super().__init__(
            output=output,
            return_direct=return_direct
        )
        self.state_update = kwargs


def make_optional_model(original_model: BaseModel) -> BaseModel:
    """
    Takes a Pydantic model and returns a new model with all attributes optional.
    """
    optional_attributes = {
        attr_name: (
            Union[None, attr_type],
            Field(
                default=None, description=original_model.model_fields[attr_name].description)
        )
        for attr_name, attr_type in original_model.__annotations__.items()
    }

    # Define a custom Pydantic model with optional attributes
    new_class_name = original_model.__name__ + 'Optional'
    OptionalModel = create_model(
        new_class_name,
        **optional_attributes,
        __base__=original_model
    )
    OptionalModel.model_config["validate_assignment"] = True

    return OptionalModel


class IntentTool(StructuredTool, ABC):
    form: BaseModel = None
    args_schema_: Optional[Type[BaseModel]] = None
    description_: Optional[str] = None
    state: Union[IntentToolState | None] = None
    skip_confirm: Optional[bool] = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args_schema_ = None
        self.description_ = None
        self.init_state()

    def init_state(self):
        state_initializer = {
            None: self.set_inactive_state,
            IntentToolState.INACTIVE: self.set_inactive_state,
            IntentToolState.ACTIVE: self.set_active_state,
            IntentToolState.FILLED: self.set_filled_state
        }
        state_initializer[self.state]()

    def set_inactive_state(self):
        # Guard so that we don't overwrite the original args_schema if
        # set_inactive_state is called multiple times
        if not self.state == IntentToolState.INACTIVE:
            self.state = IntentToolState.INACTIVE
            self.description_ = self.description
            self.description = f"Starts the intent {self.name}, which {self.description_}"
            self.args_schema_ = self.args_schema
            self.args_schema = IntentToolInactivePayload

    def set_active_state(self):
        # if not self.state == FormToolState.ACTIVE:
        self.state = IntentToolState.ACTIVE
        self.description = f"Updates data for intent {self.name}, which {self.description_}"
        self.args_schema = make_optional_model(self.args_schema_)
        if not self.form:
            self.form = self.args_schema()
        elif isinstance(self.form, str):
            self.form = self.args_schema(**json.loads(self.form))

    def set_filled_state(self):
        self.state = IntentToolState.FILLED
        self.description = f"Finalizes intent {self.name}, which {self.description_}"
        self.args_schema = make_optional_model(self.args_schema_)
        if not self.form:
            self.form = self.args_schema()
        elif isinstance(self.form, str):
            self.form = self.args_schema(**json.loads(self.form))
        self.args_schema = IntentToolConfirmPayload

    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        match self.state:
            case IntentToolState.INACTIVE:
                self.set_active_state()
                return IntentToolOutcome(
                    output=f"Starting intent {self.name}. If the user as already provided some information, call {self.name}.",
                    active_intent_tool=self,
                    function_call=self.name
                )
            case IntentToolState.ACTIVE:
                self._update_form(**kwargs)
                if self.is_form_filled():
                    self.set_filled_state()
                    if self.skip_confirm:
                        result = self._run_when_complete()
                        return IntentToolOutcome(
                            active_intent_tool=None,
                            output=result,
                            return_direct=self.return_direct
                        )
                    else:
                        return IntentToolOutcome(
                            active_intent_tool=self,
                            output="Form is filled. Ask the user to confirm the information."
                        )
                else:
                    return IntentToolOutcome(
                        active_intent_tool=self,
                        output="Form updated with the provided information. Ask the user for the next field."
                    )
            case IntentToolState.FILLED:
                if kwargs.get("confirm"):
                    result = self._run_when_complete()
                    # if no exception is raised, the form is complete and the tool is
                    # done, so reset the active form tool
                    return IntentToolOutcome(
                        active_intent_tool=None,
                        output=result,
                        return_direct=self.return_direct
                    )
                else:
                    self.set_active_state()
                    return IntentToolOutcome(
                        active_intent_tool=self,
                        output="Ask the user to update the form."
                    )

    def is_form_filled(self) -> bool:
        """
        The default implementation checks if all values except optional ones are set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            # if field_info.is_required():
            if not getattr(self.form, field_name):
                return False
        return True

    @abstractmethod
    def _run_when_complete(self) -> str:
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        The default implementation returns the first field that is not set.
        """
        if self.state == IntentToolState.FILLED:
            return None

        for field_name, field_info in self.args_schema.__fields__.items():
            if not getattr(self.form, field_name):
                return field_name

    def get_tool_start_message(self, input: dict) -> str:
        message = ""
        match self.state:
            case IntentToolState.INACTIVE:
                message = f"Starting {self.name}"
            case IntentToolState.ACTIVE:
                message = f"Updating form for {self.name}"
            case IntentToolState.FILLED:
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
    # The outcome of a given call to a tool
    # Needs `None` as a valid type, since this is what this will start as
    tool_outcome: Annotated[Optional[Union[IntentToolOutcome,
                                           str, None]], operator.setitem]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[Optional[list[tuple[AgentAction,
                                                      FunctionMessage]]], operator.add]
    error: Annotated[Optional[str], operator.setitem]

    active_intent_tool: Annotated[Optional[IntentTool], operator.setitem]

    function_call: Annotated[Optional[str], operator.setitem]


class ContextReset(BaseTool):
    name = "ContextReset"
    description = """Call this tool when the user doesn't want to complete the intent anymore. DON'T call it when he wants to change some data."""
    args_schema: Type[BaseModel] = IntentToolInactivePayload

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return IntentToolOutcome(
            state_update={
                "active_intent_tool": None
            },
            output="Context reset. Form cleared. Ask the user what he wants to do next."
        )


def filter_active_tools(
    tools: Sequence[BaseTool],
    context: AgentState
):
    """
    Form tools are replaced by their activators if they are not active.
    """
    if context.get("active_intent_tool"):
        # If a form_tool is active, it is the only form tool available
        base_tools = [
            tool for tool in tools if not isinstance(
                tool, IntentTool)]
        tools = [
            *base_tools,
            context.get("active_intent_tool"),
            ContextReset(context=context)
        ]
    return tools