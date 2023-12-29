import logging
import pickle

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mai_assistant.src.clients import GoogleClient, RedisClient
from mai_assistant.src.models.create_calendar_event_payload import \
    CreateCalendarEventPayload

logger = logging.getLogger(__name__)
google_actions_router = APIRouter(prefix="/google")


@google_actions_router.post("/{conversation_id}/calendar")
def create_calendar_event(
    conversation_id: str,
    data: CreateCalendarEventPayload,
    redis_client: RedisClient
):

    credentials = redis_client.hget(
        conversation_id,
        "google_credentials"
    )
    credentials = pickle.loads(credentials)

    # TODO: This can be obtained with dependency injection
    # The endpoint should set the credentials in the object
    google_client = GoogleClient(credentials=credentials)

    google_client.create_calendar_event(
        data
    )

    return JSONResponse(
        status_code=200,
        content={"message": "Event created"}
    )
