import logging
import pickle

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from mai_assistant.src.conversational_engine.langchain_extention.structured_agent_executor import FormStructuredChatExecutorContext, make_optional_model
import json

logger = logging.getLogger(__name__)


def get_stored_memory(
        redis_client: redis.Redis,
        chat_id: str) -> BaseChatMemory:
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
            input_key="input"
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
        context: FormStructuredChatExecutorContext = pickle.loads(context)

        if context.active_form_tool:
            loaded = json.loads(context.form)
            loaded = {key: value for key, value in loaded.items() if value}
            context_form_class = make_optional_model(context.active_form_tool.args_schema)
            context.form = context_form_class(**loaded)
        logger.info("Loaded context from redis")
    else:
        context = FormStructuredChatExecutorContext()
    return context