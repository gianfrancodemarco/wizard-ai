import json
import logging
import pickle

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mai_assistant.src.agents import GPTAgent, get_stored_memory
from mai_assistant.src.callbacks import LoggerCallbackHandler
from mai_assistant.src.callbacks.tool_logger import ToolLoggerCallback
from mai_assistant.src.clients import RabbitMQClient, RedisClient
from mai_assistant.src.constants import MessageQueues, MessageType 
from mai_assistant.src.models.chat_payload import ChatPayload

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat")


@chat_router.post("")
async def chat(
    data: ChatPayload,
    redis_client: RedisClient,
    rabbitmq_client: RabbitMQClient
) -> JSONResponse:

    # Prepare input and memory
    input = {"input": data.question}
    memory = get_stored_memory(redis_client, data.chat_id)

    # Run agent
    answer = await GPTAgent(memory).agent_chain.arun(
        input,
        #callbacks=[LoggerCallbackHandler()]
        callbacks=[
            ToolLoggerCallback(
                chat_id=data.chat_id,
                rabbitmq_client=rabbitmq_client,
                queue=MessageQueues.MAI_ASSISTANT_OUT.value
            ),
            LoggerCallbackHandler()
        ]
    )

    __update_stored_memory(redis_client, data.chat_id, memory)
    __publish_answer(rabbitmq_client, data.chat_id, answer)

    return JSONResponse({"content": answer})

def __update_stored_memory(redis_client: RedisClient, chat_id: str, memory: dict):
    redis_client.hset(
        chat_id,
        "memory",
        pickle.dumps(memory)
    )
    logger.info("Saved memory to redis")

def __publish_answer(rabbitmq_client: RabbitMQClient, chat_id: str, answer: str):
    rabbitmq_client.publish(
        queue=MessageQueues.MAI_ASSISTANT_OUT.value,
        message=json.dumps({
            "type": MessageType.TEXT.value,
            "chat_id": chat_id,
            "content": answer
        })
    )
    logger.info("Published answer to RabbitMQ")