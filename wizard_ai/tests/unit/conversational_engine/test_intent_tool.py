from wizard_ai.conversational_engine.intent_agent.intent_tool import (
    AgentState, ContextReset, IntentTool, IntentToolState, BaseTool, filter_active_tools)

from .mocks import MockBaseTool, MockIntentTool


def test_filter_active_tools_no_active_intent_tool():
    tools = [
        MockBaseTool(),
        MockBaseTool(),
        MockIntentTool(),
        MockIntentTool(),
    ]
    agent_state = AgentState()

    filtered_tools = filter_active_tools(tools, agent_state)

    assert len(filtered_tools) == 4
    assert isinstance(filtered_tools[0], BaseTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert isinstance(filtered_tools[2], IntentTool)
    assert filtered_tools[2].state == IntentToolState.INACTIVE
    assert isinstance(filtered_tools[3], IntentTool)
    assert filtered_tools[3].state == IntentToolState.INACTIVE

def test_filter_active_tools_with_active_intent_tool():

    active_intent_tool = MockIntentTool()

    tools = [
        MockBaseTool(),
        MockBaseTool(),
        active_intent_tool,
        MockIntentTool(),
    ]
    agent_state = AgentState()
    agent_state["active_intent_tool"] = active_intent_tool

    filtered_tools = filter_active_tools(tools, agent_state)

    assert len(filtered_tools) == 4
    assert isinstance(filtered_tools[0], BaseTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert filtered_tools[2] == active_intent_tool
    assert isinstance(filtered_tools[3], ContextReset)
