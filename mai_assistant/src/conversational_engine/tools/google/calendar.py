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

        #TODO: Change hardcoded chat id
        credentials = get_redis_client().hget(
            "1213778192",
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
        return "Event created"


tools = [
    GoogleCalendar()
]
