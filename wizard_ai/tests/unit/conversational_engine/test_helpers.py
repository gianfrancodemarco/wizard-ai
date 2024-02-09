import pytest

from wizard_ai.conversational_engine.intent_agent.helpers import \
    StateGraphDrawer
from wizard_ai..conversational_engine.intent_agent.intent_agent_executor import \
    IntentAgentExecutor


def test_draw_wizard_ai_graph():
    drawer = StateGraphDrawer()
    graph = IntentAgentExecutor()
    drawer.draw(graph)