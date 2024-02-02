import pickle
import textwrap
from typing import Dict, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel

from mai_assistant.clients import (CreateCalendarEventPayload,
                                   GoogleClient, get_redis_client)
from mai_assistant.conversational_engine.langchain_extention import (
    FormTool)

from mai_assistant.constants.redis_keys import RedisKeys


class GoogleCalendarCreator(FormTool):

    name = "GoogleCalendarCreator"
    description = """Useful to create events on Google Calendar."""
    args_schema: Type[BaseModel] = CreateCalendarEventPayload

    chat_id: Optional[str] = None

    def _run_when_complete(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Use the tool."""

        credentials = get_redis_client().hget(
            self.chat_id,
            RedisKeys.GOOGLE_CREDENTIALS.value
        )
        credentials = pickle.loads(credentials)

        google_client = GoogleClient(credentials)
        payload = CreateCalendarEventPayload(
            summary=kwargs["summary"],
            description=kwargs["description"],
            start=kwargs["start"],
            end=kwargs["end"]
        )
        google_client.create_calendar_event(payload)
        return "The event was created successfully"

    def get_tool_start_message(self, input: Union[Dict, str]) -> str:

        if isinstance(input, str):
            input = eval(input)

        payload = CreateCalendarEventPayload(**input)

        head_string = "Updated form with the following information:"
        if self.is_form_complete():
            head_string = "Form is complete."

        return f"{head_string}\n" +\
            textwrap.dedent(f"""
                Summary: {payload.summary}
                Description: {payload.description}
                Start: {payload.start}
                End: {payload.end}
            """)
