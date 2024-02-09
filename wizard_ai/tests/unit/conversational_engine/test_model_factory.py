from langchain_core.prompts.chat import ChatPromptTemplate

from wizard_ai.conversational_engine.intent_agent import (AgentState,
                                                          IntentAgentExecutor,
                                                          ModelFactory)

from .mocks import *


class TestModelFactory:
    def test_get_model_default_prompt(self):
        state = AgentState()
        model = ModelFactory.build_model(state=state)
        prompt_template = model.steps[1].messages[0].prompt.template
        basic_template = "You are a personal assistant trying to help the user. You always answer in English."
        assert prompt_template.startswith(basic_template)

    def test_get_model_active_intent_tool(self):
        state = AgentState()
        active_intent_tool = MockFormTool()
        active_intent_tool.set_active_state()
        state["active_intent_tool"] = active_intent_tool
        model = ModelFactory.build_model(state=state)
        assert isinstance(model.steps[1], ChatPromptTemplate)
