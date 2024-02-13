"""
A class representing an Intent Agent Executor that manages the execution flow of a state graph.

The IntentAgentExecutor class extends the StateGraph class and includes methods for initializing the agent with tools and callbacks, building a workflow control graph, retrieving active tools, getting tools by name, obtaining tool executors, determining actions based on agent state, model building, calling the agent and tools, handling tool start and end events, and parsing the graph output.

Attributes:
    MAX_INTERMEDIATE_STEPS (int): Maximum number of intermediate steps allowed in a prompt.
    _on_tool_start (callable): Callback function for when a tool starts.
    _on_tool_end (callable): Callback function for when a tool ends.
    _tools (Sequence[Type[Any]]): List of tools used by the agent.

Methods:
    __init__: Initialize the Intent Agent Executor with tools and callbacks.
    __build_graph: Build a graph for workflow control.
    get_tools: Return active tools based on the given state.
    get_tool_by_name: Get a tool by name based on the provided AgentState.
    get_tool_executor: Get the tool executor based on the provided agent state.
    should_continue_after_agent: Determine action to take after evaluating the agent state.
    should_continue_after_tool: Determine the next action based on the current state after a tool has been executed.
    build_model: Build a model based on the provided agent state.
    call_agent: Call the agent model with the provided state data and return updates.
    on_tool_start: Call a callback function when a tool starts.
    on_tool_end: Execute a callback function when a tool operation is completed.
    call_tool: Call a tool based on the provided agent state and handle tool execution outcomes.
    parse_output: Parse the final state of the graph and return the output.

Note: The IntentAgentExecutor class is part of a system that manages agent workflows and tool executions within a defined graph structure.
"""
import logging
import pprint
import traceback
from typing import Any, Sequence, Type

from langchain.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import *
from langchain_core.messages import FunctionMessage
from langgraph.graph import END, StateGraph

from wizard_ai.conversational_engine.intent_agent.intent_tool import (
    AgentState, IntentToolOutcome, filter_active_tools)
from wizard_ai.conversational_engine.intent_agent.intent_tool_executor import \
    IntentToolExecutor
from wizard_ai.conversational_engine.intent_agent.model_factory import \
    ModelFactory

logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)


class IntentAgentExecutor(StateGraph):
    """A class representing an Intent Agent Executor that manages the execution flow of a state graph.
    
    The IntentAgentExecutor class extends the StateGraph class and includes methods for initializing the agent with tools and callbacks, building a workflow control graph, retrieving active tools, getting tools by name, obtaining tool executors, determining actions based on agent state, model building, calling agent and tools, handling tool start and end events, and parsing the graph output.
    
    Attributes:
        MAX_INTERMEDIATE_STEPS (int): Maximum number of intermediate steps allowed in a prompt.
        _on_tool_start (callable): Callback function for when a tool starts.
        _on_tool_end (callable): Callback function for when a tool ends.
        _tools (Sequence[Type[Any]]): List of tools used by the agent.
    
    Methods:
        __init__: Initialize the Intent Agent Executor with tools and callbacks.
        __build_graph: Build a graph for workflow control.
        get_tools: Return active tools based on the given state.
        get_tool_by_name: Get a tool by name based on the provided AgentState.
        get_tool_executor: Get the tool executor based on the provided agent state.
        should_continue_after_agent: Determine action to take after evaluating the agent state.
        should_continue_after_tool: Determine the next action based on the current state after a tool has been executed.
        build_model: Build a model based on the provided agent state.
        call_agent: Call the agent model with the provided state data and return updates.
        on_tool_start: Call a callback function when a tool starts.
        on_tool_end: Execute a callback function when a tool operation is completed.
        call_tool: Call a tool based on the provided agent state and handle tool execution outcomes.
        parse_output: Parse the final state of the graph and return the output.
    
    Note: The IntentAgentExecutor class is part of a system that manages agent workflows and tool executions within a defined graph structure.
    """

    MAX_INTERMEDIATE_STEPS = 5

    def __init__(
        self,
        tools: Sequence[Type[Any]] = [],
        on_tool_start: callable = None,
        on_tool_end: callable = None,
    ) -> None:
        """Initialize an Agent object with a list of tools and callbacks for tool start and end.
        
        Args:
            tools (Sequence[Type[Any]]): A list of tools to be used by the Agent. Default is an empty list.
            on_tool_start (callable): A callback function to be executed when a tool starts. Default is None.
            on_tool_end (callable): A callback function to be executed when a tool ends. Default is None.
        
        Returns:
            None
        
        Attributes:
            _on_tool_start (callable): Callback function for tool start.
            _on_tool_end (callable): Callback function for tool end.
            _tools (Sequence[Type[Any]]): List of tools used by the Agent.
        
        Note:
            This method initializes the Agent object by setting the tools, tool start and end callbacks, and building the agent graph.
        """
        super().__init__(AgentState)

        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._tools = tools
        self.__build_graph()

    def __build_graph(self):
        """Builds a graph for workflow control.
        
        This method sets up nodes and conditional edges within the graph for the workflow control.
        Nodes "agent" and "tool" are added with corresponding function calls.
        Conditional edges are defined based on conditions returned by functions should_continue_after_agent and should_continue_after_tool.
        The edges connect nodes based on conditions with possible transitions to "tool", "error", "end", or "continue".
        The entry point for the graph is set to "agent".
        The compiled graph is stored in the 'app' attribute.
        
        Note: This function is likely part of a larger workflow or state machine implementation.
        """

        self.add_node("agent", self.call_agent)
        self.add_node("tool", self.call_tool)

        self.add_conditional_edges(
            "agent",
            self.should_continue_after_agent,
            {
                "tool": "tool",
                "error": "agent",
                "end": END
            }
        )

        self.add_conditional_edges(
            "tool",
            self.should_continue_after_tool,
            {
                "error": "agent",
                "continue": "agent",
                "end": END
            }
        )

        self.set_entry_point("agent")
        self.app = self.compile()

    def get_tools(self, state: AgentState):
        """Return active tools based on the given state.
        
        Args:
            state (AgentState): The state of the agent to filter tools.
        
        Returns:
            list: A list of active tools based on the provided agent state.
        """
        return filter_active_tools(self._tools[:], state)

    def get_tool_by_name(self, name: str, agent_state: AgentState):
        """Get a tool by name from the available tools based on a provided AgentState.
        
        Args:
            name (str): The name of the tool to retrieve.
            agent_state (AgentState): The state of the agent for filtering the available tools.
        
        Returns:
            Tool or None: The tool with the provided name if found, otherwise None.
        """
        return next((tool for tool in self.get_tools(
            agent_state) if tool.name == name), None)

    def get_tool_executor(self, state: AgentState):
        """
        Get the tool executor based on the provided agent state.
        
        Args:
            state (AgentState): The state of the agent for which the tool executor is being retrieved.
        
        Returns:
            IntentToolExecutor: An instance of IntentToolExecutor initialized with the tools specific to the provided agent state.
        """
        return IntentToolExecutor(self.get_tools(state))

    def should_continue_after_agent(self, state: AgentState):
        """Determines the action to be taken based on the AgentState.
        
        Args:
            state (AgentState): The current state of the agent.
        
        Returns:
            str: Represents the action to take after evaluating the agent state.
                 - "error": Indicates that there is an error in the agent state.
                 - "end": Indicates that the agent has finished.
                 - "tool": Indicates that the agent needs a tool action.
        
        """
        if state.get("error"):
            return "error"
        elif isinstance(state.get("agent_outcome"), AgentFinish):
            return "end"
        elif isinstance(state.get("agent_outcome"), AgentAction):
            return "tool"

    def should_continue_after_tool(self, state: AgentState):
        """Determines the next action based on the current state after a tool has been executed.
        
        Args:
            state (AgentState): The current state of the agent.
        
        Returns:
            str: Indicates the next action to be taken. Possible values are "error" if an error occurred, "end" if the tool outcome is intended to end, or "continue" if the process should continue.
        """
        if state.get("error"):
            return "error"
        elif isinstance(state.get("tool_outcome"), IntentToolOutcome) and state.get("tool_outcome").return_direct:
            return "end"
        else:
            return "continue"

    def build_model(self, state: AgentState):
        """
        Build a model based on the provided agent state.
        
        Args:
            state (AgentState): The state of the agent used for model creation.
        
        Returns:
            Model: The constructed model based on the provided state.
        
        Raises:
            SomeSpecificException: An exception raised if model construction fails due to a specific reason.
        """
        return ModelFactory.build_model(
            state=state,
            tools=self.get_tools(state)
        )

    # Define the function that calls the model
    def call_agent(self, state: AgentState):
        """Call the agent model with the provided state data and return updates based on the outcome.
        
        Args:
            state (AgentState): The state data to pass to the agent model.
        
        Returns:
            dict: A dictionary containing updates with the following keys:
                - "agent_outcome": The outcome generated by the agent model.
                - "function_call": None to reset the function call.
                - "tool_outcome": None to reset the tool outcome.
                - "error": None to reset the error.
        
        Raises:
            OutputParserException: If an exception related to output parsing occurs during the invocation.
        """
        try:
            # Cap the number of intermediate steps in a prompt to 5
            if len(state.get("intermediate_steps")
                   ) > self.MAX_INTERMEDIATE_STEPS:
                state["intermediate_steps"] = state.get(
                    "intermediate_steps")[-self.MAX_INTERMEDIATE_STEPS:]

            agent_outcome = self.build_model(state=state).invoke(state)
            updates = {
                "agent_outcome": agent_outcome,
                "function_call": None,  # Reset the function call
                "tool_outcome": None,  # Reset the tool outcome
                "error": None  # Reset the error
            }
            return updates
        # TODO: if other exceptions are raised, we should handle them here
        except OutputParserException as e:
            traceback.print_exc()
            updates = {"error": str(e)}
            return updates

    def on_tool_start(self, tool: BaseTool, tool_input: dict):
        """Call a callback function when a tool starts.
        
        Args:
            tool (BaseTool): The tool that is starting.
            tool_input (dict): The input data for the tool.
        
        Raises:
            TypeError: If the callback function is not provided.
        """
        if self._on_tool_start:
            self._on_tool_start(tool, tool_input)

    def on_tool_end(self, tool: BaseTool, tool_output: Any):
        """Execute a callback function when a tool operation is completed.
        
        This function takes in a tool instance and its output data as parameters. If a callback function is provided, it will be executed with the tool instance and its output as arguments.
        
        Args:
            tool (BaseTool): The tool instance that has finished its operation.
            tool_output (Any): The output data produced by the tool operation.
        
        Returns:
            None
        
        Raises:
            No specific exceptions are raised within this function.
        """
        if self._on_tool_end:
            self._on_tool_end(tool, tool_output)

    def call_tool(self, state: AgentState):
        """Call a tool based on the provided agent state and handle tool execution outcomes.
        
        Args:
            state (AgentState): The current state of the agent, containing details about the action to be performed.
        
        Returns:
            dict: A dictionary containing updates to the agent state after executing the tool. May include
                'intermediate_steps', 'tool_outcome', 'agent_outcome', and 'error'.
        
        Raises:
            Exception: If any error occurs during the tool execution, the exception is caught and logged in the 'error' field.
        """
        try:
            action = state.get("agent_outcome")
            tool = self.get_tool_by_name(action.tool, state)

            self.on_tool_start(tool=tool, tool_input=action.tool_input)
            tool_outcome = self.get_tool_executor(state).invoke(action)
            self.on_tool_end(tool=tool, tool_output=tool_outcome.output)

            updates = {
                **tool_outcome.state_update,
                "intermediate_steps": [(
                    action,
                    FunctionMessage(
                        content=str(tool_outcome.output),
                        name=action.tool
                    ))],
                "tool_outcome": tool_outcome,
                "agent_outcome": None,
                "error": None
            }

        except Exception as e:
            traceback.print_exc()
            updates = {
                "intermediate_steps": [(action, FunctionMessage(
                    content=str(e),
                    name=action.tool
                ))],
                "error": str(e)
            }
        finally:
            return updates

    def parse_output(self, graph_output: dict) -> str:
        """
        Parses the final state of the graph.
        Theoretically, only one between tool_outcome and agent_outcome are set.
        Returns the str to be considered the output of the graph.
        """

        state = graph_output[END]

        output = None
        if state.get("tool_outcome"):
            output = state.get("tool_outcome").output
        elif state.get("agent_outcome"):
            output = state.get("agent_outcome").return_values["output"]

        return output
