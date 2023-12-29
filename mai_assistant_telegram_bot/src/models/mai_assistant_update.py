from pydantic import BaseModel
from mai_assistant.src.constants import MessageType

class MaiAssistantUpdate(BaseModel):
    """
    Messages from the WebSocket 
    """

    # Type is one of the values of MessageType
    type: MessageType
    content: str