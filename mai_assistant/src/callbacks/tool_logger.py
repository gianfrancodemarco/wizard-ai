import json
from typing import Any, Dict

from langchain_core.callbacks import AsyncCallbackHandler

from mai_assistant.src.clients import RabbitMQClient
from mai_assistant.src.constants import MessageType


class ToolLoggerCallback(AsyncCallbackHandler):

    def __init__(
        self,
        chat_id: str,
        rabbitmq_client: RabbitMQClient,
        queue: str
    ) -> None:
        super().__init__()
        self.chat_id = chat_id
        self.rabbitmq_client = rabbitmq_client
        self.queue = queue

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.rabbitmq_client.publish(
            queue=self.queue,
            message=json.dumps({
                "chat_id": self.chat_id,
                "type": MessageType.TOOL_START.value,
                "content": f"{serialized['name']}: {input_str}"
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