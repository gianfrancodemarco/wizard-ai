from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException

from wizard_ai.conversational_engine.intent_agent.form_tool import (
    AgentState, ContextReset)
from wizard_ai.conversational_engine.intent_agent.intent_agent_executor import *

from .mocks import *


class MockIntentAgentExecutorOkModel(IntentAgentExecutor):
    def get_model(self, state):
        agent = MagicMock()
        agent.invoke = MagicMock(return_value="Mocked response")
        return agent


class MockIntentAgentExecutorErrorModel(IntentAgentExecutor):
    def get_model(self, state):
        agent = MagicMock()
        agent.invoke = MagicMock(
            side_effect=OutputParserException("Mocked error"))
        return agent


class TestIntentAgentExecutor:

    def test_get_tools_no_active_form_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        tools = graph.get_tools(state)
        assert len(tools) == 0

    def test_get_tools_with_active_form_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        tools = graph.get_tools(state)
        assert len(tools) == 2
        assert tools[0] == active_form_tool
        assert tools[1] == ContextReset()

    def test_get_tool_by_name_existing_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        tool = graph.get_tool_by_name("MockFormTool", state)
        assert tool == active_form_tool

    def test_get_tool_by_name_non_existing_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        tool = graph.get_tool_by_name("NonExistingTool", state)
        assert tool is None

    def test_get_model_default_prompt(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        model = graph.get_model(state)
        prompt_template = model.steps[1].messages[0].prompt.template
        basic_template = "\nYou are a personal assistant trying to help the user. You always answer in English.\n"
        assert prompt_template == basic_template

    def test_get_model_active_form_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        active_form_tool = MockFormTool()
        state["active_form_tool"] = active_form_tool
        model = graph.get_model(state)
        assert isinstance(model.steps[1], ChatPromptTemplate)

    def test_should_continue_error(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["error"] = True
        result = graph.should_continue(state)
        assert result == "error"

    def test_should_continue_end(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = AgentFinish({}, "end")
        result = graph.should_continue(state)
        assert result == "end"

    def test_should_continue_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = AgentAction(
            tool="MockBaseTool", tool_input={}, log="")
        result = graph.should_continue(state)
        assert result == "tool"

    def test_should_continue_after_tool_error(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["error"] = True
        result = graph.should_continue_after_tool(state)
        assert result == "error"

    def test_should_continue_after_tool_return_direct(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["intermediate_steps"] = [(AgentAction(
            tool="MockBaseTool", tool_input={}, log=""), "output")]
        tool = MockBaseTool()
        tool.return_direct = True
        state["active_form_tool"] = tool
        result = graph.should_continue_after_tool(state)
        assert result == "end"

    def test_should_continue_after_tool_continue(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["intermediate_steps"] = [(AgentAction(
            tool="MockBaseTool", tool_input={}, log=""), "output")]
        tool = MockBaseTool()
        tool.return_direct = False
        state["active_form_tool"] = tool
        result = graph.should_continue_after_tool(state)
        assert result == "continue"

    def test_call_agent(self):
        graph = MockIntentAgentExecutorOkModel()
        state = AgentState()
        state["input"] = "Hello"
        state["chat_history"] = []
        state["intermediate_steps"] = []
        state["error"] = None
        response = graph.call_agent(state)
        assert "agent_outcome" in response
        assert "error" in response
        assert response["agent_outcome"] == "Mocked response"
        assert response["error"] is None

    def test_call_agent_error(self):
        graph = MockIntentAgentExecutorErrorModel()
        state = AgentState()
        state["input"] = "Hello"
        state["chat_history"] = []
        state["intermediate_steps"] = []
        state["error"] = None
        response = graph.call_agent(state)
        assert "error" in response
        assert response["error"] is not None

    def test_call_tool(self):
        graph = IntentAgentExecutor()
        state = AgentState()
        state["agent_outcome"] = AgentAction(
            tool="MockBaseTool", tool_input={}, log="")
        response = graph.call_tool(state)
        assert "intermediate_steps" in response
        assert "error" in response
