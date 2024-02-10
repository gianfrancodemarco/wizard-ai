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

    MAX_INTERMEDIATE_STEPS = 5

    def __init__(
        self,
        tools: Sequence[Type[Any]] = [],
        on_tool_start: callable = None,
        on_tool_end: callable = None,
    ) -> None:
        super().__init__(AgentState)

        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._tools = tools
        self.__build_graph()

    def __build_graph(self):

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
        return filter_active_tools(self._tools[:], state)

    def get_tool_by_name(self, name: str, agent_state: AgentState):
        return next((tool for tool in self.get_tools(agent_state) if tool.name == name), None)

    def get_tool_executor(self, state: AgentState):
        return IntentToolExecutor(self.get_tools(state))

    def should_continue_after_agent(self, state: AgentState):
        if state.get("error"):
            return "error"
        elif isinstance(state.get("agent_outcome"), AgentFinish):
            return "end"
        elif isinstance(state.get("agent_outcome"), AgentAction):
            return "tool"

    def should_continue_after_tool(self, state: AgentState):
        if state.get("error"):
            return "error"
        elif isinstance(state.get("tool_outcome"), IntentToolOutcome) and state.get("tool_outcome").return_direct:
            return "end"
        else:
            return "continue"

    def build_model(self, state: AgentState):
        return ModelFactory.build_model(
            state=state,
            tools=self.get_tools(state)
        )

    # Define the function that calls the model
    def call_agent(self, state: AgentState):
        try:
            # Cap the number of intermediate steps in a prompt to 5
            if len(state.get("intermediate_steps")) > self.MAX_INTERMEDIATE_STEPS:
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
        if self._on_tool_start:
            self._on_tool_start(tool, tool_input)

    def on_tool_end(self, tool: BaseTool, tool_output: Any):
        if self._on_tool_end:
            self._on_tool_end(tool, tool_output)

    def call_tool(self, state: AgentState):
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
