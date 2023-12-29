import logging

from fastapi import APIRouter, HTTPException

from mai_assistant.src.clients import RedisClient

logger = logging.getLogger(__name__)


conversations_router = APIRouter(prefix="/conversations")


@conversations_router.delete
def delete_conversations(redis_client: RedisClient):
    """Delete all conversations"""
    redis_client.flushdb()
    return {"content": None}


@conversations_router.delete("/{conversation_id}")
async def chat(conversation_id: str, redis_client: RedisClient):
    """Delete a conversation"""

    if not redis_client.exists(conversation_id):
        raise HTTPException(status_code=404, detail="Item not found")

    redis_client.delete(conversation_id)
    logger.info(f"Deleted conversation {conversation_id}")
    return {"content": "Conversation deleted"}
