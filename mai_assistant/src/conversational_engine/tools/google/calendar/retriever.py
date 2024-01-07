import pickle
from datetime import datetime
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel

from mai_assistant.src.clients import (GetCalendarEventsPayload, GoogleClient,
                                       get_redis_client)
from mai_assistant.src.conversational_engine.langchain_extention import \
    FormTool, FormToolActivator

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from mai_assistant.src.conversational_engine.langchain_extention import FormStructuredChatExecutorContext

class GoogleCalendarRetriever(FormTool):

    name = "GoogleCalendarRetriever"
    description = """Useful to retrieve events from Google Calendar"""
    return_direct = True
    args_schema: Type[BaseModel] = GetCalendarEventsPayload

    chat_id: Optional[str] = None

    def _run(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        number_of_events: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        credentials = get_redis_client().hget(
            self.chat_id,
            "google_credentials"
        )
        credentials = pickle.loads(credentials)

        google_client = GoogleClient(credentials)
        payload = GetCalendarEventsPayload(
            start=start,
            end=end,
            number_of_events=number_of_events
        )
        return google_client.get_calendar_events_html(payload)

    def get_tool_start_message(self, input: dict) -> str:

        payload = GetCalendarEventsPayload(**input)

        if payload.start and payload.end:
            return f"Retrieving events from Google Calendar between {payload.start} and {payload.end}"
        elif payload.number_of_events:
            return f"Retrieving next {payload.number_of_events} from Google Calendar"


    async def ais_form_complete(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
    ) -> bool:
        """
        User should provide number_of_events or start and end dates
        """
        if context.form.number_of_events:
            return True
        elif context.form.start and context.form.end:
            return True
        else:
            return False
    
    def get_information_to_collect(self) -> str:
        return ["number_of_events OR (start and end dates)"]