from typing import Optional, Type

import faker
from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from wizard_ai.clients import GetCalendarEventsPayload

fake = faker.Faker()

class GoogleCalendarRetriever(StructuredTool):

    name = "GoogleCalendarRetriever"
    description = """Useful to retrieve events from Google Calendar"""
    args_schema: Type[BaseModel] = GetCalendarEventsPayload

    return_direct = True
    skip_confirm = True
    chat_id: Optional[str] = None


    def _run(
        self,
        *args,
        **kwargs,
    ) -> str:
        return "OK"
