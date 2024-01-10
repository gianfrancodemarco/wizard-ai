import json
from typing import Any, Dict, List

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.tools import Tool

from mai_assistant.src.clients import RabbitMQProducer
from mai_assistant.src.constants import MessageType


class ToolLoggerCallback(AsyncCallbackHandler):

    def __init__(
        self,
        chat_id: str,
        rabbitmq_client: RabbitMQProducer,
        queue: str,
        tools: List[Tool] = []
    ) -> None:
        super().__init__()
        self.chat_id = chat_id
        self.rabbitmq_client = rabbitmq_client
        self.queue = queue
        self.tools = tools

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        try:
            tool = next((tool for tool in self.tools if tool.name == serialized["name"]), None)
            tool_start_message = tool.get_tool_start_message(input_str)
        except BaseException:
            tool_start_message = f"{serialized['name']}: {input_str}"

        self.rabbitmq_client.publish(
            queue=self.queue,
            message=json.dumps({
                "chat_id": self.chat_id,
                "type": MessageType.TOOL_START.value,
                "content": tool_start_message
            })
        )

    async def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> Any:
        """Run when tool ends running."""
        self.rabbitmq_client.publish(
            queue=self.queue,
            message=json.dumps({
                "chat_id": self.chat_id,
                "type": MessageType.TOOL_END.value,
                "content": output
            })
        )
