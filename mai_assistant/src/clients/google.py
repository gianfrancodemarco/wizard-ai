import os
from datetime import datetime, timezone
from typing import Any, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')


class CreateCalendarEventPayload(BaseModel):
    summary: str
    description: str
    start: datetime
    end: datetime


class GetCalendarEventsPayload(BaseModel):
    start: Optional[datetime] = Field(
        default=None,
        description="Start date to retrieve events from. Null if number_of_events is not null")
    end: Optional[datetime] = Field(
        default=None,
        description="End date to retrieve events from. Null if number_of_events is not null")
    number_of_events: Optional[int] = Field(
        default=None,
        description="Number of events to retrieve. Null if start and end are not null")


class GoogleClient:

    def __init__(
        self,
        credentials: Credentials,
    ):
        self.credentials = credentials

    def create_calendar_event(
        self,
        data: CreateCalendarEventPayload
    ):
        service = build('calendar', 'v3', credentials=self.credentials)

        event = {
            'summary': data.summary,
            'description': data.description,
            'start': {'dateTime': data.start.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': data.end.isoformat(), 'timeZone': 'UTC'},
        }

        event = service.events().insert(calendarId='primary', body=event).execute()

    def get_calendar_events(
        self,
        data: GetCalendarEventsPayload
    ) -> List[Any]:
        service = build('calendar', 'v3', credentials=self.credentials)

        if data.start and data.end:
            # Google API need the timezone. For simplicity we set UTC
            data.start = data.start.replace(tzinfo=timezone.utc)
            data.end = data.end.replace(tzinfo=timezone.utc)
        elif data.number_of_events:
            data.start = datetime.now().replace(tzinfo=timezone.utc)
            data.end = None

        events_result = service.events().list(
            calendarId='primary',
            timeMin=data.start.isoformat(),
            timeMax=data.end.isoformat() if data.end else None,
            singleEvents=True,
            orderBy='startTime',
        ).execute()

        events = events_result.get('items', [])
        return events

    def get_calendar_events_markdown(
        self,
        data: GetCalendarEventsPayload
    ) -> str:
        events = self.get_calendar_events(data)
        events_string = self.__events_result_to_markdown_string(events)
        return events_string

    def __events_result_to_markdown_string(self, events: List[Any]) -> str:
        events_string = ""
        for idx, event in enumerate(events):
            event_start = event['start'].get(
                'dateTime', event['start'].get('date'))
            event_end = event['end'].get('dateTime', event['end'].get('date'))
            event_summary = event['summary']
            event_link = event['htmlLink']
            event_string = f"{idx+1}. {event_start} - [{event_summary}]({event_link})\n"
            events_string += event_string
        return events_string
