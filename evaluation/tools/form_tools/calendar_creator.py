import pickle
import textwrap
from datetime import datetime
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from wizard_ai.clients import (CreateCalendarEventPayload, GoogleClient,
                               get_redis_client)
from wizard_ai.constants.redis_keys import RedisKeys
from wizard_ai.conversational_engine.form_agent import FormTool, FormToolState


class GoogleCalendarCreator(FormTool):

    name = "GoogleCalendarCreator"
    description = """Useful to create events/memos/reminders on Google Calendar."""
    args_schema: Type[BaseModel] = CreateCalendarEventPayload

    chat_id: Optional[str] = None

    def _run_when_complete(
        self,
        summary: str,
        description: str,
        start: datetime,
        end: datetime
    ) -> str:
        return {
            "summary": summary,
            "description": description,
            "start": start,
            "end": end
        }

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        """
        Use library faker to generate random data for the form.
        """

        import faker

        fake = faker.Faker()
        
        start = fake.date_time_this_month()
        start = start.replace(second=0, microsecond=0)

        return {
            "summary": fake.text(max_nb_chars=30),
            "description": fake.text(),
            "start": start,
            "end": start + fake.time_delta()
        }