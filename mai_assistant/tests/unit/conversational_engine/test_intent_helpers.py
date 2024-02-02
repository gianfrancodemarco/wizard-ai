from typing import Any

from pydantic import BaseModel

from mai_assistant.conversational_engine.langchain_extention.form_tool import (
    AgentState, ContextReset, FormTool, FormToolActivator, filter_active_tools)
from mai_assistant.conversational_engine.langchain_extention.intent_helpers import \
    BaseTool
from typing import Type


class MockBaseTool(BaseTool):
    name = "MockBaseTool"
    description = "MockBaseTool description"

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        pass


class _DummyPayload(BaseModel):
    pass


class MockFormTool(FormTool):
    name = "MockFormTool"
    description = "MockFormTool description"
    args_schema: Type[BaseModel] = _DummyPayload


def test_filter_active_tools_no_active_form_tool():
    tools = [
        MockBaseTool(),
        MockBaseTool(),
        MockFormTool(),
        MockFormTool(),
    ]
    agent_state = AgentState()

    filtered_tools = filter_active_tools(tools, agent_state)

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
    agent_state = AgentState()
    agent_state["active_form_tool"] = active_form_tool

    filtered_tools = filter_active_tools(tools, agent_state)

    assert len(filtered_tools) == 4
    assert isinstance(filtered_tools[0], BaseTool)
    assert isinstance(filtered_tools[1], BaseTool)
    assert filtered_tools[2] == active_form_tool
    assert isinstance(filtered_tools[3], ContextReset)
