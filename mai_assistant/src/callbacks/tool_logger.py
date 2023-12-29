import json
from typing import Any, Dict

from langchain_core.callbacks import AsyncCallbackHandler
from starlette.websockets import WebSocket

from mai_assistant.src.constants import MessageType


class ToolLoggerCallback(AsyncCallbackHandler):

    def __init__(
        self,
        ws: WebSocket
    ) -> None:
        super().__init__()
        self.ws = ws

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        await self.ws.send_text(json.dumps({
            "tool": serialized["name"],
            "type": MessageType.TOOL_START.value,
            "content": f"{serialized['name']}: {input_str}"
        }))
        return input_str

    async def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> Any:
        """Run when tool ends running."""
        await self.ws.send_text(json.dumps({
            "tool": "",
            "type": MessageType.TOOL_END.value,
            "content": ""
        }))
        return output
