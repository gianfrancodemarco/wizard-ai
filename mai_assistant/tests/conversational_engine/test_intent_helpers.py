from typing import Any

from mai_assistant.conversational_engine.langchain_extention.intent_helpers import (
    BaseTool, AgentState, FormTool, FormToolActivator, ContextReset, ContextUpdate,
    filter_active_tools)

class MockBaseTool(BaseTool):
    name = "MockBaseTool"
    description = "MockBaseTool description"

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        pass

class MockFormTool(FormTool):
    name = "MockFormTool"
    description = "MockFormTool description"

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        pass


def test_filter_active_tools_no_active_form_tool():
    tools = [
        MockBaseTool(),
        MockBaseTool(),
        MockFormTool(),   
        MockFormTool(),
    ]
    context = AgentState(active_form_tool=None)

    filtered_tools = filter_active_tools(tools, context)

    assert len(filtered_tools) == 4
    assert isinstance(filtered_tools[0], BaseTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert isinstance(filtered_tools[2], FormToolActivator)
    assert isinstance(filtered_tools[3], FormToolActivator)

def test_filter_active_tools_with_active_form_tool():

    active_form_tool = MockFormTool()

    tools = [
        MockBaseTool(),
        MockBaseTool(),
        active_form_tool,   
        MockFormTool(),
    ]
    context = AgentState()
    context.get("active_form_tool") = active_form_tool

    filtered_tools = filter_active_tools(tools, context)

    assert len(filtered_tools) == 5
    assert isinstance(filtered_tools[0], FormTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert isinstance(filtered_tools[2], BaseTool)
    assert isinstance(filtered_tools[3], ContextUpdate)
    assert isinstance(filtered_tools[4], ContextReset)