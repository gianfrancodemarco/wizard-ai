import logging
import pickle

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory

logger = logging.getLogger(__name__)


def get_stored_memory(redis_client: redis.Redis, chat_id: str) -> BaseChatMemory:
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
