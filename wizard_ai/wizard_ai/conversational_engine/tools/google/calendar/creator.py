import pickle
import textwrap
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from wizard_ai.clients import (CreateCalendarEventPayload, GoogleClient,
                               get_redis_client)
from wizard_ai.constants.redis_keys import RedisKeys
from wizard_ai.conversational_engine.intent_agent import (IntentTool,
                                                                 IntentToolState)


class GoogleCalendarCreator(IntentTool):

    name = "GoogleCalendarCreator"
    description = """Useful to create events/memos/reminders on Google Calendar."""
    args_schema: Type[BaseModel] = CreateCalendarEventPayload

    chat_id: Optional[str] = None

    def _run_when_complete(self) -> str:
        """Use the tool."""
        credentials = get_redis_client().hget(
            self.chat_id,
            RedisKeys.GOOGLE_CREDENTIALS.value
        )
        google_client = GoogleClient(pickle.loads(credentials))
        payload = CreateCalendarEventPayload(
            summary=self.form.summary,
            description=self.form.description,
            start=self.form.start,
            end=self.form.end
        )
        google_client.create_calendar_event(payload)
        return "The event was created successfully"

    def get_tool_start_message(self, input: Union[Dict, str]) -> str:
        base_message = super().get_tool_start_message(input)
        if self.state in (IntentToolState.ACTIVE, IntentToolState.FILLED):
            payload = self.args_schema(**input)
            return f"{base_message}\n" +\
                textwrap.dedent(f"""
                    Summary: {payload.summary}
                    Description: {payload.description}
                    Start: {payload.start}
                    End: {payload.end}
                """)

        return base_message
