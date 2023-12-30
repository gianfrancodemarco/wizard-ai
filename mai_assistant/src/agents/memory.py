import logging
import pickle

from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory

from mai_assistant.src.clients import RedisClient

logger = logging.getLogger(__name__)


def get_stored_memory(redis_client: RedisClient, chat_id: str) -> BaseChatMemory:
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
            human_prefix="Question",
            ai_prefix="Answer"
        )
    return memory
