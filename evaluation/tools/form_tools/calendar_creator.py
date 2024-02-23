import random
from datetime import datetime, timedelta
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from wizard_ai.clients import CreateCalendarEventPayload
from wizard_ai.conversational_engine.tools import GoogleCalendarCreator


class GoogleCalendarCreatorEvaluation(GoogleCalendarCreator):

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
            "summary": fake.text(max_nb_chars=30),
            "description": fake.text(),
            "start": start,
            "end": start + timedelta(days=random.randint(1, 7), hours=random.randint(1, 3), minutes=random.randint(0, 59))
        }