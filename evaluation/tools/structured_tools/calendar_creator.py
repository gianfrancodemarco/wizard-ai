from typing import Dict, Optional, Type

import faker
from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from wizard_ai.clients import CreateCalendarEventPayload

from .structured_tool_for_evaluation import StructuredToolForEvaluation

fake = faker.Faker()

class GoogleCalendarCreator(StructuredTool, StructuredToolForEvaluation):
    name = "GoogleCalendarCreator"
    description = """Useful to create events/memos/reminders on Google Calendar."""
    args_schema: Type[BaseModel] = CreateCalendarEventPayload
    chat_id: Optional[str] = None