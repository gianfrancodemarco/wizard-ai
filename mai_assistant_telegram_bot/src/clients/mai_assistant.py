from contextlib import asynccontextmanager
import os
import websockets

class MAIAssistantClient:

    def __init__(self) -> None:
        self.host = os.environ.get('MAI_ASSISTANT_URL', 'http://localhost:8000')
    
    @asynccontextmanager
    async def chat_ws(
        self
    ):
        websocket = await websockets.connect(f"ws://{self.host}/chat/ws")
        yield websocket
        await websocket.close()