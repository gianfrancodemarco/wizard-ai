import pytest

from wizard_ai.conversational_engine.langchain_extention.helpers import \
    StateGraphDrawer
from wizard_ai.conversational_engine.langchain_extention.wizard_ai_graph import \
    MAIAssistantGraph


def test_draw_wizard_ai_graph():
    drawer = StateGraphDrawer()
    graph = MAIAssistantGraph()
    drawer.draw(graph)