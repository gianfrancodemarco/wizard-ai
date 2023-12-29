import logging
import pickle

from fastapi import APIRouter

from mai_assistant.src.agents import GPTAgent, get_stored_memory
from mai_assistant.src.callbacks import LoggerCallbackHandler
from mai_assistant.src.clients import RedisClient
from mai_assistant.src.models.chat_payload import ChatPayload

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat")

@chat_router.post
async def chat(data: ChatPayload, redis_client: RedisClient):

    # Prepare input and memory
    input = {"input": data.question}
    memory = get_stored_memory(redis_client, data.conversation_id)

    # Run agent
    answer = await GPTAgent(memory).agent_chain.arun(
        input,
        callbacks=[LoggerCallbackHandler()]
    )

    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")

    return {"answer": answer}