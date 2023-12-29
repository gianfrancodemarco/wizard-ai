from pydantic import BaseModel


class ChatPayload(BaseModel):
    conversation_id: str
    question: str
