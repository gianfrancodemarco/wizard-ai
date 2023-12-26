from contextlib import asynccontextmanager
import os
import websockets
import requests

class MAIAssistantClient:

    def __init__(self) -> None:
        self.host = os.environ.get('MAI_ASSISTANT_URL', 'localhost:8000')
    
    @asynccontextmanager
    async def chat_ws(
        self
    ):
        websocket = await websockets.connect(f"ws://{self.host}/chat/ws")
        try:
            yield websocket
        finally:
            await websocket.close()


    def chat(
        self,
        conversation_id: str,
        message: str
    ) -> str:
        response = requests.post(
            f"http://{self.host}/chat",
            json={"conversation_id": conversation_id, "question": message},
        )
        return response.json()