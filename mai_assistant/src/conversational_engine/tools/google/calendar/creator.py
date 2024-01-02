import pickle
from datetime import datetime
from typing import Optional, Type
import textwrap

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel

from mai_assistant.src.clients import GoogleClient, CreateCalendarEventPayload, get_redis_client


class GoogleCalendarCreator(BaseTool):

    name = "GoogleCalendarCreator"
    description = """Useful to create events on Google Calendar"""
    return_direct = True
    args_schema: Type[BaseModel] = CreateCalendarEventPayload

    chat_id: Optional[str] = None
    
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

    def get_tool_start_message(self, input: dict) -> str:
        payload = CreateCalendarEventPayload(**input)

        return "Creating event on Google Calendar\n" +\
            textwrap.dedent(f"""
                Summary: {payload.summary}
                Description: {payload.description}
                Start: {payload.start}
                End: {payload.end}
            """)
