import json
import logging
import pickle

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory

from mai_assistant.conversational_engine.langchain_extention.structured_agent_executor import (
    AgentState, make_optional_model)

logger = logging.getLogger(__name__)


def get_stored_memory(
        redis_client: redis.Redis,
        chat_id: str
) -> BaseChatMemory:
    memory = redis_client.hget(
        chat_id,
        "memory"
    )
    if memory is not None:
        memory = pickle.loads(memory)
        logger.info("Loaded memory from redis")
    else:
        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="history",
            human_prefix="Human",
            ai_prefix="Answer",
            input_key="messages",
            return_messages=True
        )
    return memory


def get_stored_context(
    redis_client: redis.Redis,
    chat_id: str
) -> dict:
    context = redis_client.hget(
        chat_id,
        "context"
    )
    if context is not None:
        context: AgentState = pickle.loads(context)
        active_form_tool = context.get("active_form_tool")

        if active_form_tool:
            loaded = json.loads(context.form)
            loaded = {key: value for key, value in loaded.items() if value}
            context_form_class = make_optional_model(active_form_tool.args_schema)
            context.form = context_form_class(**loaded)
        logger.info("Loaded context from redis")
    else:
        context = AgentState()
    return context
