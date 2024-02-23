from typing import Optional, Type

import faker
from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from wizard_ai.clients import GetCalendarEventsPayload
from .structured_tool_for_evaluation import StructuredToolForEvaluation

fake = faker.Faker()

class GoogleCalendarRetriever(StructuredTool, StructuredToolForEvaluation):
    name = "GoogleCalendarRetriever"
    description = """Useful to retrieve events from Google Calendar"""
    args_schema: Type[BaseModel] = GetCalendarEventsPayload
    return_direct = True
    chat_id: Optional[str] = None