from mai_assistant.src.dependencies import RedisClient

from fastapi import APIRouter

redis_router = APIRouter()

@redis_router.delete("/conversations/delete")
def delete_conversations(redis_client: RedisClient):
    """Delete all conversations"""
    redis_client.flushdb()
    return {"success": True}