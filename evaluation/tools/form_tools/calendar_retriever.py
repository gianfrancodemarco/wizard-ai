import pickle
import textwrap
from datetime import datetime
from typing import Optional, Type

from pydantic import BaseModel

from wizard_ai.clients import (GetCalendarEventsPayload, GoogleClient,
                               get_redis_client)
from wizard_ai.constants import RedisKeys
from wizard_ai.conversational_engine.form_agent.form_tool import (
    FormTool, FormToolState)
from typing import Dict, Optional, Type, Union


class GoogleCalendarRetriever(FormTool):

    name = "GoogleCalendarRetriever"
    description = """Useful to retrieve events from Google Calendar"""
    args_schema: Type[BaseModel] = GetCalendarEventsPayload

    return_direct = True
    skip_confirm = True
    chat_id: Optional[str] = None

    def _run_when_complete(
        self,
        start: datetime,
        end: datetime
    ) -> str:
        return {
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
            "start": start,
            "end": start + fake.time_delta()
        }