from pydantic import BaseModel


class ChatPayload(BaseModel):
    chat_id: str
    question: str
