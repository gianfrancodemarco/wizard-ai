import os
from contextlib import asynccontextmanager

import requests
import websockets


class MAIAssistantClient:


    def __init__(self) -> None:
        self.HOST = os.environ.get('MAI_ASSISTANT_URL', 'localhost:8000')
        self.REST_URL = f"http://{self.HOST}"


    @asynccontextmanager
    async def chat_ws(
        self
    ):
        websocket = await websockets.connect(f"ws://{self.HOST}/chat/ws")
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
            f"{self.REST_URL}/chat",
            json={"conversation_id": conversation_id, "question": message},
        )
        return response.json()


    def reset_conversation(
        self,
        conversation_id: str
    ) -> None:
        requests.delete(f"{self.REST_URL}/conversations/{conversation_id}")


    def login_to_google(
        self,
        conversation_id: str
    ) -> str:
        response = requests.post(f"{self.REST_URL}/google/login/{conversation_id}")
        return response.json()["content"]