import pickle
import textwrap
from typing import Optional, Type

from pydantic import BaseModel

from wizard_ai.clients import (GetCalendarEventsPayload, GoogleClient,
                               get_redis_client)
from wizard_ai.constants import RedisKeys
from wizard_ai.conversational_engine.langchain_extention import (FormTool,
                                                                 FormToolState)


class GoogleCalendarRetriever(FormTool):

    name = "GoogleCalendarRetriever"
    description = """Useful to retrieve events from Google Calendar"""
    args_schema: Type[BaseModel] = GetCalendarEventsPayload

    return_direct = True
    skip_confirm = True
    chat_id: Optional[str] = None

    def _run_when_complete(self) -> str:
        credentials = get_redis_client().hget(
            self.chat_id,
            RedisKeys.GOOGLE_CREDENTIALS.value
        )
        google_client = GoogleClient(pickle.loads(credentials))
        payload = GetCalendarEventsPayload(
            start=self.form.start,
            end=self.form.end,
            number_of_events=self.form.number_of_events
        )
        return google_client.get_calendar_events_html(payload)

    def get_tool_start_message(self, input: dict) -> str:
        base_message = super().get_tool_start_message(input)
        if self.state in (FormToolState.ACTIVE, FormToolState.FILLED):
            payload = GetCalendarEventsPayload(**input)

            return f"{base_message}\n" +\
                textwrap.dedent(f"""
                    Number of events: {payload.number_of_events}
                    Start: {payload.start}
                    End: {payload.end}
                """)

        return base_message

    def is_form_filled(
        self,
    ) -> bool:
        """
        User should provide number_of_events or start and end dates
        """
        if self.form.number_of_events:
            return True
        elif self.form.start and self.form.end:
            return True
        else:
            return False

    def get_information_to_collect(self) -> str:
        return ["number_of_events OR (start and end dates)"]
