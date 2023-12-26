from typing import Any, Dict
from langchain_core.callbacks import AsyncCallbackHandler
import json
import logging
import pickle

from fastapi import APIRouter
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import BaseModel
from starlette.websockets import WebSocket

from mai_assistant.src.agents.gpt import GPTAgent
from mai_assistant.src.dependencies import RedisClient

logger = logging.getLogger(__name__)

chat_router = APIRouter()


def get_stored_memory(redis_client: RedisClient, conversation_id: str) -> BaseChatMemory:
    memory = redis_client.get(conversation_id)
    if memory is not None:
        memory = pickle.loads(memory)
        logger.info("Loaded memory from redis")
    else:
        memory = ConversationBufferWindowMemory(
            k=3, memory_key="history")
    return memory


class ChatPayload(BaseModel):
    conversation_id: str
    question: str


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
            "type": "tool",
            "content": f"{serialized['name']}: {input_str}"
        }))
        return input_str


@chat_router.websocket("/chat/ws")
async def websocket_endpoint(websocket: WebSocket, redis_client: RedisClient):

    await websocket.accept()
    payload = await websocket.receive_json()
    data = ChatPayload.model_validate(payload)

    # Prepare input and memory
    input = {"input": data.question}
    memory = get_stored_memory(redis_client, data.conversation_id)

    # Run agent
    answer = GPTAgent(memory).agent_chain.run(
        input,
        #callbacks=[ToolLoggerCallback(websocket)]
    )

    # Save memory
    # memory.save_context(input, {"history": answer})
    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")

    await websocket.send_text(json.dumps({"answer": answer, "type": "answer"}))
    await websocket.close()