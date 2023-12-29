import json
import logging
import pickle

from fastapi import APIRouter
from starlette.websockets import WebSocket

from mai_assistant.src.agents import GPTAgent, get_stored_memory
from mai_assistant.src.callbacks import LoggerCallbackHandler
from mai_assistant.src.callbacks.tool_logger import ToolLoggerCallback
from mai_assistant.src.dependencies import RedisClient
from mai_assistant.src.models.chat_payload import ChatPayload

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat")

@chat_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, redis_client: RedisClient):

    await websocket.accept()
    payload = await websocket.receive_json()
    data = ChatPayload.model_validate(payload)

    # Prepare input and memory
    input = {"input": data.question}
    memory = get_stored_memory(redis_client, data.conversation_id)

    # Run agent
    answer = await GPTAgent(memory).agent_chain.arun(
        input,
        callbacks=[ToolLoggerCallback(websocket), LoggerCallbackHandler()]
    )

    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")

    await websocket.send_text(json.dumps({"answer": answer, "type": "answer"}))
    await websocket.close()