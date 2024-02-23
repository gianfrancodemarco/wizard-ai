
import random
from datetime import datetime, timedelta
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from wizard_ai.clients import GetCalendarEventsPayload
from wizard_ai.conversational_engine.tools import GoogleCalendarRetriever


class GoogleCalendarRetrieverEvaluation(GoogleCalendarRetriever):

    name = "GoogleCalendarRetriever"
    description = """Useful to retrieve events from Google Calendar"""
    args_schema: Type[BaseModel] = GetCalendarEventsPayload

    return_direct = True
    chat_id: Optional[str] = None

    def _run_when_complete(
        self,
        start: datetime,
        end: datetime
    ) -> str:
        return "OK"


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
            "end": start + timedelta(days=random.randint(1, 7), hours=random.randint(1, 3), minutes=random.randint(0, 59))
        }