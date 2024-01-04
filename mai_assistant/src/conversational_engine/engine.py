
import json
import logging
import pickle

import redis
from fastapi.responses import JSONResponse
from langchain_core.callbacks import StdOutCallbackHandler

from mai_assistant.src.clients import (RabbitMQProducer, get_rabbitmq_producer,
                                       get_redis_client)
from mai_assistant.src.constants import MessageQueues, MessageType
from mai_assistant.src.conversational_engine.agents import (GPTAgent,
                                                            get_stored_memory)
from mai_assistant.src.conversational_engine.callbacks import (
    LoggerCallbackHandler, ToolLoggerCallback)
from mai_assistant.src.models.chat_payload import ChatPayload

logger = logging.getLogger(__name__)

rabbitmq_producer = get_rabbitmq_producer()
redis_client = get_redis_client()


async def process_message(data: dict) -> None:

    data: ChatPayload = ChatPayload.model_validate(data)

    # Prepare input and memory
    memory = get_stored_memory(redis_client, data.chat_id)

    # Run agent
    agent = GPTAgent(
        memory=memory,
        chat_id=data.chat_id,
    )
    callbacks = [
        ToolLoggerCallback(
            chat_id=data.chat_id,
            rabbitmq_client=rabbitmq_producer,
            queue=MessageQueues.MAI_ASSISTANT_OUT.value,
            tools=agent.tools,
        ),
        StdOutCallbackHandler(),
        LoggerCallbackHandler()
    ]

    answer = await agent.agent_chain.arun(
        input=data.content,
        callbacks=callbacks
    )

    __update_stored_memory(redis_client, data.chat_id, memory)
    __publish_answer(rabbitmq_producer, data.chat_id, answer)

    return JSONResponse({"content": answer})


def __update_stored_memory(
        redis_client: redis.Redis,
        chat_id: str,
        memory: dict):

    redis_client.hset(
        chat_id,
        "memory",
        pickle.dumps(memory)
    )
    logger.info("Saved memory to redis")


def __publish_answer(
        rabbitmq_client: RabbitMQProducer,
        chat_id: str,
        answer: str):
    rabbitmq_client.publish(
        queue=MessageQueues.MAI_ASSISTANT_OUT.value,
        message=json.dumps({
            "type": MessageType.TEXT.value,
            "chat_id": chat_id,
            "content": answer
        })
    )
    logger.info("Published answer to RabbitMQ")
