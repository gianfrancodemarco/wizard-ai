"""This module defines various classes and functions related to intent tools and agent state management.

Classes:
- `IntentTool`: An abstract base class for managing intent-related processes. It includes methods for handling tool states, running the intent tool, updating form fields, and more.
- `IntentToolState`: An enumeration class that enumerates the states of an intent tool.
- `IntentToolInactivePayload`: A dummy class representing the payload schema for an inactive intent tool.
- `IntentToolConfirmPayload`: A class representing the payload for confirming an intent tool.
- `IntentToolOutcome`: Represents the output of a form tool operation.
- `ContextReset`: A tool class to reset the context and clear form data for an intent.

Functions:
- `make_optional_model`: Takes a Pydantic model and returns a new model with all attributes optional.
- `filter_active_tools`: Replaces form tools by their activators if they are not active.

Enums:
- `AgentState`: Represents the state of an agent during a conversation.

Attributes and Types:
- Various imported modules and data types used within the module.

The module provides detailed documentation for each class, method, and function, including their purpose, parameters, return values, and potential exceptions raised. It outlines the structure and behavior of intent tools, form tool outcomes, and agent state management."""
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
    """
    Enumerates the state of an intent tool. 
    
    Attributes:
        INACTIVE (str): Represents the inactive state of the intent tool.
        ACTIVE (str): Represents the active state of the intent tool.
        FILLED (str): Represents the filled state of the intent tool.
    """
    """
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"

# We cannot pass directly the BaseModel class as args_schema as pydantic will raise errors,
# so we need to create a dummy class that inherits from BaseModel.


class IntentToolInactivePayload(BaseModel):
    """A dummy class representing the payload schema for an inactive intent tool.
    
    This class serves as a workaround to pass a schema for an inactive intent tool. It is a placeholder class that does not contain any specific attributes or methods.
    
    Note:
        This class is used as a workaround for passing a schema to Pydantic that represents an inactive intent tool.
    
    Attributes:
        N/A
    
    Methods:
        N/A
    """
    pass


class IntentToolConfirmPayload(BaseModel):
    """
    A class representing the payload for confirming an intent tool.
    
    Attributes:
        confirm (bool): Indicates whether the user confirms the intent (True) or not/wants to make changes (False).
    
    Parameters:
        confirm (bool): The confirmation status of the intent.
    
    """
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
        """Initialize the object with output, return_direct flag, and additional keyword arguments.
        
        Args:
            output (str): The output of the object.
            return_direct (bool, optional): A flag indicating whether to return direct results. Defaults to False.
            **kwargs: Additional keyword arguments for state update.
        
        Attributes:
            state_update (dict): A dictionary of additional keyword arguments for state update.
        """
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
    """
    ```plaintext
    The IntentTool class is a structured tool for managing intent-related processes. It inherits from StructuredTool and is an abstract base class. 
    
    Attributes:
        - form: Optional BaseModel attribute (initialized as None) for form data.
        - args_schema_: Optional BaseModel type for arguments schema.
        - description_: Optional attribute for the tool's description.
        - state: Union of IntentToolState or None representing the tool's current state.
        - skip_confirm: Optional boolean indicating whether confirmation is skipped.
    
    Methods:
        - __init__: Initializes the instance with optional arguments and keyword arguments.
        - init_state: Initializes the state of the intent tool based on predefined states.
        - set_inactive_state: Set the state of the IntentTool to inactive.
        - set_active_state: Set the active state for the intent tool.
        - set_filled_state: Update the state of the intent to FILLED and finalize the intent description.
        - _run: Run the intent tool based on its current state.
        - is_form_filled: Check if all required values in the form are set.
        - _run_when_complete: Abstract method to be implemented when the intent run is completed.
        - _update_form: Update form fields with provided key-value pairs.
        - get_next_field_to_collect: Get the next field to collect based on the current tool state.
        - get_tool_start_message: Generate a start message based on the state of the tool.
        - get_information_to_collect: Get the information needed to collect from the arguments.
    
    Raises:
        - ToolException: If a ValidationError occurs while setting form field values in _update_form.
    
    Returns:
        - Various return types based on the methods, including None, str, and IntentToolOutcome.
    ```
    ```python
    class IntentTool(StructuredTool, ABC):
        form: BaseModel = None
        args_schema_: Optional[Type[BaseModel]] = None
        description_: Optional[str] = None
        state: Union[IntentToolState | None] = None
        skip_confirm: Optional[bool] = False
    
        def __init__(self, *args, **kwargs):
            ...
        
        def init_state(self):
            ...
    
        def set_inactive_state(self):
            ...
    
        def set_active_state(self):
            ...
    
        def set_filled_state(self):
            ...
    
        def _run(self, *args, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> str:
            ...
    
        def is_form_filled(self) -> bool:
            ...
        
        @abstractmethod
        def _run_when_complete(self) -> str:
            ...
    
        def _update_form(self, **kwargs):
            ...
    
        def get_next_field_to_collect(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
            ...
    
        def get_tool_start_message(self, input: dict) -> str:
            ...
    
        def get_information_to_collect(self) -> str:
            ...
    ```
    ```plaintext
    IntentToolState: Enum used to represent the various states of the IntentTool.
    BaseModel: A custom base class for creating data models.
    CallbackManagerForToolRun: Class for managing callbacks during the tool run.
    ToolException: Exception raised when a form field value cannot be set due to ValidationError.
    ValidationError: Exception indicating a validation error.
    ```
    ```python
    
    class IntentToolOutcome(NamedTuple):
        output: str
        active_intent_tool: Optional['IntentTool']
        return_direct: Optional[bool]
    
    
    IntentToolState = Enum('IntentToolState', ['INACTIVE', 'ACTIVE', 'FILLED'])
    
    
    def make_optional_model(model: Optional[Type[BaseModel]]) -> Optional[Type[BaseModel]]:
        ...
    
    ```
    ```plaintext
    NamedTuple: Factory function for creating tuple subclasses with named fields.
    Optional: Type hint for an optional value.
    'IntentTool': Forward reference to the IntentTool class.
    ```
    ```python
    
    def make_optional_model(model: Optional[Type[BaseModel]]) -> Optional[Type[BaseModel]]:
        ...
    ```
    """
    form: BaseModel = None
    args_schema_: Optional[Type[BaseModel]] = None
    description_: Optional[str] = None
    state: Union[IntentToolState | None] = None
    skip_confirm: Optional[bool] = False

    def __init__(self, *args, **kwargs):
        """
        Initialize the instance with optional arguments and keyword arguments.
        
        This method initializes the instance by calling the superclass' init method with optional arguments and keyword arguments. It sets the args_schema_ attribute to None and the description_ attribute to None. Finally, it calls the init_state method.
        
        No parameters are directly accepted by this method.
        
        This method does not return anything.
        
        Raises:
            This method does not raise any exceptions.
        """
        super().__init__(*args, **kwargs)
        self.args_schema_ = None
        self.description_ = None
        self.init_state()

    def init_state(self):
        """Initialize the state of the intent tool based on predefined states.
        
        The method initializes the state of the intent tool based on predefined states mapping the current state to corresponding state setting functions.
        
        Predefined States:
            None: The tool is in an inactive state.
            IntentToolState.INACTIVE: The tool is in an inactive state.
            IntentToolState.ACTIVE: The tool is in an active state.
            IntentToolState.FILLED: The tool is in a filled state.
        
        The state of the intent tool is updated based on the current state and the mapping to the respective state setting function is executed.
        """
        state_initializer = {
            None: self.set_inactive_state,
            IntentToolState.INACTIVE: self.set_inactive_state,
            IntentToolState.ACTIVE: self.set_active_state,
            IntentToolState.FILLED: self.set_filled_state
        }
        state_initializer[self.state]()

    def set_inactive_state(self):
        """Set the state of the IntentTool to inactive.
        
        If the current state is not already inactive, changes the state to IntentToolState.INACTIVE and modifies the description
        and args_schema attributes accordingly. The description is updated to include the starting intent's name and original description.
        The args_schema is set to IntentToolInactivePayload to indicate the inactive state schema.
        
        Returns:
            None
        """
        # Guard so that we don't overwrite the original args_schema if
        # set_inactive_state is called multiple times
        if not self.state == IntentToolState.INACTIVE:
            self.state = IntentToolState.INACTIVE
            self.description_ = self.description
            self.description = f"Starts the intent {self.name}, which {self.description_}"
            self.args_schema_ = self.args_schema
            self.args_schema = IntentToolInactivePayload

    def set_active_state(self):
        """Set the active state for the intent tool.
        
        This function updates the state of the intent tool to 'ACTIVE'. It also updates the description based on the tool's name and description. The function initializes the args_schema using make_optional_model. If the form is not already set, it creates a new form using the args_schema. If the form is a string, it converts it into a dictionary and sets the form using the args_schema.
        
        Raises:
            No specific exceptions are raised.
        
        Returns:
            None
        """
        # if not self.state == FormToolState.ACTIVE:
        self.state = IntentToolState.ACTIVE
        self.description = f"Updates data for intent {self.name}, which {self.description_}"
        self.args_schema = make_optional_model(self.args_schema_)
        if not self.form:
            self.form = self.args_schema()
        elif isinstance(self.form, str):
            self.form = self.args_schema(**json.loads(self.form))

    def set_filled_state(self):
        """Update the state of the intent to FILLED and finalize the intent description.
        
        This function sets the state attribute to IntentToolState.FILLED, updates the description attribute
        to include the finalization message, updates the args_schema attribute to a processed schema,
        and initializes or updates the form attribute based on the args_schema_ value.
        
        Raises:
            No specific exceptions are raised.
        
        Returns:
            None
        """
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
        """
        Run the intent tool based on its current state.
        
        Args:
            run_manager (Optional[CallbackManagerForToolRun]): Optional parameter for managing callbacks during the tool run.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            str: Returns a string containing the outcome of running the intent tool.
        
        Description:
            - If the intent tool is inactive:
                - Sets the tool to an active state.
            - If the intent tool is active:
                - Updates the form with provided information.
                - Checks if the form is filled.
                    - If filled, sets the tool to a filled state.
                    - If not filled, prompts for the next field.
            - If the intent tool is filled:
                - If confirmation is received, completes the tool's run.
                - If confirmation is not received, sets the tool back to an active state.
        """
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
        """Update form fields with the provided key-value pairs.
        
        Args:
            **kwargs: Arbitrary keyword arguments representing form field names and their updated values.
        
        Raises:
            ToolException: If a ValidationError occurs while setting form field values, an exception is raised with detailed error messages about the validation errors.
        
        Note:
            If a ValidationError occurs, the errors are processed and concatenated into a string, then encapsulated within a ToolException for handling.
        """
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
        """Generate a start message based on the state of the tool.
        
        Args:
            input (dict): A dictionary containing input data.
        
        Returns:
            str: A message indicating the status of the tool's state.
        
        Raises:
            KeyError: If the input dictionary does not contain required keys.
        
        This function determines and returns a message based on the state of the tool. 
        If the tool is INACTIVE, it generates a starting message for the tool.
        If the tool is ACTIVE, it creates a message for updating the form of the tool.
        If the tool is FILLED, it produces a message to indicate the completion of the tool.
        """
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
        """
        Get the information needed to collect from the arguments.
        
        Returns:
            str: A string representation of the keys of the arguments to be collected.
        """
        return str(list(self.args.keys()))


class AgentState(TypedDict):
    """Represents the state of an agent during a conversation.
    
    Attributes:
        input (str): The input string.
        chat_history (Optional[list[BaseMessage]]): The list of previous messages in the conversation.
        agent_outcome (Optional[Union[AgentAction, AgentFinish, None]]): The outcome of a given call to the agent.
        tool_outcome (Optional[Union[IntentToolOutcome, str, None]]): The outcome of a given call to a tool.
        intermediate_steps (Optional[list[tuple[AgentAction, FunctionMessage]]]): List of actions and corresponding observations. Operations to this state should be added to the existing values.
        error (Optional[str]): Any error message associated with the state.
        active_intent_tool (Optional[IntentTool]): The active intent tool.
        function_call (Optional[str]): Information about a function call in the state.
    """
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
    """A tool to reset the context and clear the form data for an intent.
    
    Attributes:
        name (str): The name of the tool.
        description (str): Description of the tool's purpose.
        args_schema (Type[BaseModel]): Schema for the arguments expected by the tool.
        
    Methods:
        _run: Run the intent tool by resetting the context and clearing the form data.
    
    Returns:
        Any: An object representing the outcome of running the intent tool.
    
    Outcome Properties:
        state_update (dict): Dictionary with the state update after the run.
        output (str): Message to display to the user after resetting context and clearing the form.
      
    Example:
        _run()
    """
    name = "ContextReset"
    description = """Call this tool when the user doesn't want to complete the intent anymore. DON'T call it when he wants to change some data."""
    args_schema: Type[BaseModel] = IntentToolInactivePayload

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the intent tool by resetting the context and clearing the form data.
        
        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Any: An object representing the outcome of running the intent tool.
        
        Outcome Properties:
            state_update (dict): Dictionary containing the state update after the run.
            output (str): Message to be displayed to the user after resetting context and clearing the form.
        
        Example:
            _run()
        """
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
