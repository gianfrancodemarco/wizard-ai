import logging
import pickle

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from mai_assistant.src.conversational_engine.langchain_extention.structured_agent_executor import FormStructuredChatExecutorContext

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
            ai_prefix="Answer"
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
        context = pickle.loads(context)
        logger.info("Loaded context from redis")
    else:
        context = FormStructuredChatExecutorContext()
    return context