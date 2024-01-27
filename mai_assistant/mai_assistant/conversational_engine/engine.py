
import json
import logging
import pickle
from typing import Any

import redis
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage

from mai_assistant.clients import (RabbitMQProducer, get_rabbitmq_producer,
                                   get_redis_client)
from mai_assistant.clients.rabbitmq import RabbitMQProducer
from mai_assistant.constants import MessageQueues, MessageType
from mai_assistant.constants.message_queues import MessageQueues
from mai_assistant.constants.message_type import MessageType
from mai_assistant.conversational_engine.agents import get_stored_memory
from mai_assistant.conversational_engine.langchain_extention.graph import Graph
from mai_assistant.conversational_engine.langchain_extention.structured_agent_executor import \
    FormStructuredChatExecutorContext
from mai_assistant.conversational_engine.tools import *
from mai_assistant.models.chat_payload import ChatPayload

logger = logging.getLogger(__name__)

rabbitmq_producer = get_rabbitmq_producer()
redis_client = get_redis_client()


class TelegramConnector:
    
    def __init__(
        self,
        chat_id: str,
        tools: list = None,
        rabbitmq_producer: RabbitMQProducer = None,
        queue: MessageQueues = None
    ) -> None:
        self.chat_id = chat_id
        self.tools = tools
        self.rabbitmq_client = rabbitmq_producer
        self.queue = queue
        
            
    def on_tool_start(
        self,
        tool_name: str,
        tool_input: str
    ) -> Any:
        """Run when tool starts running."""
        if not self.rabbitmq_client:
            return
        
        try:
            tool = next((tool for tool in self.tools if tool.name == tool_name), None)
            tool_start_message = tool.get_tool_start_message(tool_input)
        except BaseException:
            tool_start_message = f"{tool_name}: {tool_input}"

        self.rabbitmq_client.publish(
            queue=self.queue,
            message=json.dumps({
                "chat_id": self.chat_id,
                "type": MessageType.TOOL_START.value,
                "content": tool_start_message
            })
        )

    def on_tool_end(
        self,
        tool_name: str,
        tool_output: str
    ) -> Any:
        """Run when tool ends running."""

        if not self.rabbitmq_client:
            return

        self.rabbitmq_client.publish(
            queue=self.queue,
            message=json.dumps({
                "chat_id": self.chat_id,
                "type": MessageType.TOOL_END.value,
                "content": tool_output
            })
        )

async def process_message(data: dict) -> None:

    data: ChatPayload = ChatPayload.model_validate(data)

    chat_id = data.chat_id
    tools = [
        # Calculator(),
        RandomNumberGenerator(),
        GoogleSearch(),
        GoogleCalendarCreator(chat_id=chat_id),
        GoogleCalendarRetriever(chat_id=chat_id),
        GmailRetriever(chat_id=chat_id),
        Python(),
        # DateCalculatorTool()
    ]


    # Prepare input and memory
    memory = get_stored_memory(redis_client, data.chat_id)
    
    inputs = {
        "messages": [
            SystemMessage(content="You are MAI Assistant, a virtual assistant."),
            *memory.buffer,
            HumanMessage(content=data.content)
        ]
    }

    telegram_connector = TelegramConnector(
        chat_id=chat_id,
        tools=tools,
        rabbitmq_producer=rabbitmq_producer,
        queue=MessageQueues.MAI_ASSISTANT_OUT.value
    )

    graph = Graph(
        tools=tools,
        on_tool_start=telegram_connector.on_tool_start,
        on_tool_end=telegram_connector.on_tool_end

    )

    logger.info("---")
    logger.info(f"Executing graph with inputs: {inputs}")
    logger.info("---")
    for output in graph.app.stream(inputs):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            logger.info(f"Output from node '{key}':")
            logger.info("---")
            logger.info(value)
    answer = value['messages'][-1].content

    # Prepare input and memory
    memory.save_context(
        inputs={"messages": data.content},
        outputs={"output": answer}
    )
    # context = get_stored_context(redis_client, data.chat_id)

    # __update_stored_context(
    #     redis_client,
    #     data.chat_id,
    #     agent.agent_chain.context)
    __update_stored_memory(redis_client, data.chat_id, memory)
    __publish_answer(rabbitmq_producer, data.chat_id, answer)

    return JSONResponse({"content": answer})


def __update_stored_context(
        redis_client: redis.Redis,
        chat_id: str,
        context: FormStructuredChatExecutorContext):

    # this shouldn't be here
    if context.active_form_tool:
        context.form = context.form.model_dump_json()

    redis_client.hset(
        chat_id,
        "context",
        pickle.dumps(context)
    )
    logger.info("Saved memory to redis")


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
