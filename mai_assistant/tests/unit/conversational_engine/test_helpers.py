import pytest

from mai_assistant.conversational_engine.langchain_extention.helpers import \
    StateGraphDrawer
from mai_assistant.conversational_engine.langchain_extention.mai_assistant_graph import \
    MAIAssistantGraph


def test_draw_mai_assistant_graph():
    drawer = StateGraphDrawer()
    graph = MAIAssistantGraph()
    drawer.draw(graph)