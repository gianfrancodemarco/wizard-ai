from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException

from wizard_ai.conversational_engine.intent_agent.intent_tool import (
    AgentState, ContextReset)
from wizard_ai.conversational_engine.intent_agent.intent_agent_executor import *

from .mocks import *


class MockIntentAgentExecutorOkModel(IntentAgentExecutor):
    def build_model(self, state):
        agent = MagicMock()
        agent.invoke = MagicMock(return_value="Mocked response")
        return agent


class MockIntentAgentExecutorErrorModel(IntentAgentExecutor):
    def build_model(self, state):
        agent = MagicMock()
        agent.invoke = MagicMock(
            side_effect=OutputParserException("Mocked error"))
        return agent


class MockToolError(MockBaseTool):
    name = "MockToolError"
    description = "Mock tool that raises an error"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise Exception("Mocked error")


class TestIntentAgentExecutor:

    def test_get_tools_no_active_intent_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        tools = graph.get_tools(state)
        assert len(tools) == 0

    def test_get_tools_with_active_intent_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_intent_tool = MockIntentTool()
        state["active_intent_tool"] = active_intent_tool
        tools = graph.get_tools(state)
        assert len(tools) == 2
        assert tools[0] == active_intent_tool
        assert tools[1] == ContextReset()

    def test_get_tool_by_name_existing_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_intent_tool = MockIntentTool()
        state["active_intent_tool"] = active_intent_tool
        tool = graph.get_tool_by_name("MockIntentTool", state)
        assert tool == active_intent_tool

    def test_get_tool_by_name_non_existing_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_intent_tool = MockIntentTool()
        state["active_intent_tool"] = active_intent_tool
        tool = graph.get_tool_by_name("NonExistingTool", state)
        assert tool is None

    def test_should_continue_error(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["error"] = True
        result = graph.should_continue_after_agent(state)
        assert result == "error"

    def test_should_continue_end(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = AgentFinish({}, "end")
        result = graph.should_continue_after_agent(state)
        assert result == "end"

    def test_should_continue_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = AgentAction(
            tool="MockBaseTool", tool_input={}, log="")
        result = graph.should_continue_after_agent(state)
        assert result == "tool"

    def test_should_continue_after_tool_error(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["error"] = True
        result = graph.should_continue_after_tool(state)
        assert result == "error"

    def test_should_continue_after_tool_return_direct_end(self):
        graph = IntentAgentExecutor()
        state = AgentState(tool_outcome=IntentToolOutcome(
            return_direct=True, output=""))
        result = graph.should_continue_after_tool(state)
        assert result == "end"

    def test_should_continue_after_tool_continue(self):
        graph = IntentAgentExecutor()
        state = AgentState(tool_outcome=IntentToolOutcome(
            return_direct=False, output=""))
        result = graph.should_continue_after_tool(state)
        assert result == "continue"

    def test_call_agent(self):
        graph = MockIntentAgentExecutorOkModel()
        state = AgentState()
        state = {
            "input": "Hello",
            "chat_history": [],
            "intermediate_steps": [],
            "error": None
        }
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is None
        assert "tool_outcome" in response
        assert response["tool_outcome"] is None
        assert "agent_outcome" in response
        assert response["agent_outcome"] == "Mocked response"
    def test_call_agent_error(self):
        graph = MockIntentAgentExecutorErrorModel()
        state = AgentState()
        state = {
            "input": "Hello",
            "chat_history": [],
            "intermediate_steps": [],
            "error": None
        }
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is not None

    def test_call_tool(self):
        graph = IntentAgentExecutor(
            tools=[MockIntentTool()],
        )
        state = AgentState()
        state["agent_outcome"] = AgentAction(
            tool="MockIntentTool", tool_input={}, log="")
        response = graph.call_tool(state)
        assert "intermediate_steps" in response
        assert "error" in response
        assert response["error"] is None

    def test_call_tool_error(self):
        graph = IntentAgentExecutor(
            tools=[MockToolError()],
        )
        state = AgentState()
        state["agent_outcome"] = AgentAction(
            tool="MockToolError", tool_input={}, log="")
        response = graph.call_tool(state)
        assert "intermediate_steps" in response
        assert "error" in response
        assert response["error"] is not None

    def test_on_tool_start_on_tool_end(self):
        on_tool_start = MagicMock()
        on_tool_end = MagicMock()
        graph = IntentAgentExecutor(
            tools=[MockIntentTool()],
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        state = AgentState(
            agent_outcome=AgentAction(
                tool="MockIntentTool", tool_input={}, log="")
        )
        graph.call_tool(state)
        on_tool_start.assert_called_once()
        on_tool_end.assert_called_once()

    def test_call_agent_more_intermediate_steps(self):

        graph = MockIntentAgentExecutorOkModel(
            tools=[MockIntentTool()],
        )
        state = AgentState(
            input="Hello",
            agent_outcome=AgentAction(
                tool="MockIntentTool", tool_input={}, log=""),
            intermediate_steps=(
                (
                    AgentAction(tool="MockIntentTool", tool_input={}, log=""),
                    FunctionMessage(
                        content="Tool output",
                        name="MockIntentTool"
                    )),
            )*10,
        )
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is None
        assert "tool_outcome" in response
        assert response["tool_outcome"] is None
        assert "agent_outcome" in response
        assert response["agent_outcome"] == "Mocked response"

    def test_parse_output_tool_outcome(self):
        graph = IntentAgentExecutor()
        graph_output = {END: {"tool_outcome": IntentToolOutcome(
            return_direct=True, output="Tool output")}}
        output = graph.parse_output(graph_output)
        assert output == "Tool output"

    def test_parse_output_agent_outcome(self):
        graph = IntentAgentExecutor()
        graph_output = {END: {"agent_outcome": AgentFinish(
            {"output": "Agent output"}, "end")}}
        output = graph.parse_output(graph_output)
        assert output == "Agent output"

    def test_parse_output_no_outcome(self):
        graph = IntentAgentExecutor()
        graph_output = {END: {}}
        output = graph.parse_output(graph_output)
        assert output is None

    def test_build_model(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        model = graph.build_model(state)
        assert model is not None