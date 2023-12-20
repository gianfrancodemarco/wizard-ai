import logging
import pickle

from fastapi import APIRouter
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import BaseModel

from mai_assistant.src.agents.gpt import get_gpt_agent
from mai_assistant.src.dependencies import RedisClient

logger = logging.getLogger(__name__)


def get_stored_memory(redis_client: RedisClient, conversation_id: str) -> BaseChatMemory:
    memory = redis_client.get(conversation_id)
    if memory is not None:
        memory = pickle.loads(memory)
        logger.info("Loaded memory from redis")
    else:
        memory = ConversationBufferWindowMemory(k=3, memory_key="history")
    return memory


chat_router = APIRouter()


class ChatPayload(BaseModel):
    conversation_id: str
    question: str


@chat_router.post("/chat")
def chat(data: ChatPayload, redis_client: RedisClient):

    # Prepare input and memory
    input = {"input": data.question}
    memory = get_stored_memory(redis_client, data.conversation_id)

    # Run agent
    answer = get_gpt_agent(memory).run(input)

    # Save memory
    memory.save_context(input, {"history": answer})
    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")

    return {"answer": answer}
