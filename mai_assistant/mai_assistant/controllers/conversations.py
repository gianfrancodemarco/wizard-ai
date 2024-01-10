import logging

from fastapi import APIRouter, HTTPException

from mai_assistant.clients import RedisClientDep

logger = logging.getLogger(__name__)


conversations_router = APIRouter(prefix="/conversations")


@conversations_router.delete
def delete_conversations(redis_client: RedisClientDep):
    """Delete all conversations"""
    redis_client.flushdb()
    return {"content": None}


@conversations_router.delete("/{chat_id}")
async def chat(chat_id: str, redis_client: RedisClientDep):
    """Delete a conversation"""

    if not redis_client.exists(chat_id):
        raise HTTPException(status_code=404, detail="Item not found")

    redis_client.hdel(
        chat_id,
        "memory"
    )
    redis_client.hdel(
        chat_id,
        "context"
    )
    logger.info(f"Deleted conversation {chat_id}")
    return {"content": "Conversation deleted"}
