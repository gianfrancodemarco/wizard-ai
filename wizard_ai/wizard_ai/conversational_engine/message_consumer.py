
import json
import logging
import pprint
from textwrap import dedent
from typing import Any

from fastapi.responses import JSONResponse

from wizard_ai.clients import (RabbitMQProducer, get_rabbitmq_producer,
                               get_redis_client)
from wizard_ai.clients.rabbitmq import RabbitMQProducer
from wizard_ai.constants import MessageQueues, MessageType
from wizard_ai.constants.message_queues import MessageQueues
from wizard_ai.constants.message_type import MessageType
from wizard_ai.conversational_engine.form_agent import (FormAgentExecutor,
                                                        FormTool,
                                                        get_stored_agent_state,
                                                        store_agent_state)
from wizard_ai.conversational_engine.tools import *
from wizard_ai.models.chat_payload import ChatPayload

pp = pprint.PrettyPrinter(indent=4)

logger = logging.getLogger(__name__)

rabbitmq_producer = get_rabbitmq_producer()
redis_client = get_redis_client()


class RabbitMQConnector:

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
        tool: FormTool,
        tool_input: str
    ) -> Any:
        """Run when tool starts running."""
        if not self.rabbitmq_client:
            return

        try:
            tool_start_message = tool.get_tool_start_message(tool_input)
        except BaseException:
            tool_start_message = f"{tool.name}: {tool_input}"

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
        tool: FormTool,
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
        GoogleSearch(),
        GoogleCalendarCreator(chat_id=chat_id),
        GoogleCalendarRetriever(chat_id=chat_id),
        GmailRetriever(chat_id=chat_id),
        GmailSender(chat_id=chat_id),
        PythonCodeInterpreter()
    ]

    stored_agent_state = get_stored_agent_state(redis_client, data.chat_id)

    inputs = {
        "input": data.content,
        "chat_history": [*stored_agent_state.memory.buffer],
        "intermediate_steps": [],
        "active_form_tool": stored_agent_state.active_form_tool
    }

    rabbitmq_connector = RabbitMQConnector(
        chat_id=chat_id,
        tools=tools,
        rabbitmq_producer=rabbitmq_producer,
        queue=MessageQueues.WIZARD_AI_OUT.value
    )

    graph = FormAgentExecutor(
        tools=tools,
        on_tool_start=rabbitmq_connector.on_tool_start,
        on_tool_end=rabbitmq_connector.on_tool_end
    )

    logger.info(dedent(f"""
        ---
        Executing graph with inputs: {inputs}"
        ---
    """))
    nodes = []
    for output in graph.app.stream(inputs, config={"recursion_limit": 25}):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            logger.info(dedent(f"""
                Output from node '{key}':"
                ---
                {pp.pprint(value)}
            """))
            nodes.append(key)
    logger.info(dedent(f"""
        ---



        Executed nodes:
        {" -> ".join(nodes)}




        ---
    """))

    answer = graph.parse_output(output)

    # Prepare input and memory
    stored_agent_state.memory.save_context(
        inputs={"messages": data.content},
        outputs={"output": answer}
    )
    stored_agent_state.active_form_tool = value["active_form_tool"]

    store_agent_state(redis_client, data.chat_id, stored_agent_state)
    __publish_answer(rabbitmq_producer, data.chat_id, answer)

    return JSONResponse({"content": answer})


def __publish_answer(
        rabbitmq_client: RabbitMQProducer,
        chat_id: str,
        answer: str):
    rabbitmq_client.publish(
        queue=MessageQueues.WIZARD_AI_OUT.value,
        message=json.dumps({
            "type": MessageType.TEXT.value,
            "chat_id": chat_id,
            "content": answer
        })
    )
    logger.info("Published answer to RabbitMQ")
