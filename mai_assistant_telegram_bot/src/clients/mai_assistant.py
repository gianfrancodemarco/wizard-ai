import os
import requests


class MAIAssistantClient:

    def __init__(self) -> None:
        self.url = os.environ.get('MAI_ASSISTANT_URL', 'http://localhost:8000')

    def chat(
        self,
        conversation_id: str,
        message: str
    ) -> str:
        response = requests.post(
            f"{self.url}/chat",
            json={"conversation_id": conversation_id, "question": message},
        )
        return response.json()