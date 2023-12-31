from datetime import datetime
from typing import Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from mai_assistant.src.clients import GoogleClient
from mai_assistant.src.models.create_calendar_event_payload import \
    CreateCalendarEventPayload

from mai_assistant.src.clients import get_redis_client

import pickle

class GoogleCalendar(BaseTool):
    name = "GoogleCalendar"
    description = "Useful to retrieve, create, update and delete events on Google Calendar"
    chat_id: Optional[str] = None

    args_schema: Type[BaseModel] = CreateCalendarEventPayload

    def _run(
        self,
        summary: str,
        description: str,
        start: datetime,
        end: datetime,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        credentials = get_redis_client().hget(
            self.chat_id,
            "google_credentials"
        )
        credentials = pickle.loads(credentials)

        google_client = GoogleClient(credentials)
        payload = CreateCalendarEventPayload(
            summary=summary,
            description=description,
            start=start,
            end=end
        )
        google_client.create_calendar_event(payload)
        return "The event was created successfully"


tools = [
    GoogleCalendar()
]
