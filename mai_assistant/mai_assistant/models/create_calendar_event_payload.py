from datetime import datetime

from pydantic import BaseModel


class CreateCalendarEventPayload(BaseModel):
    summary: str
    description: str
    start: datetime
    end: datetime
