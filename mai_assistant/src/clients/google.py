import os

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from mai_assistant.src.models.create_calendar_event_payload import \
    CreateCalendarEventPayload

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')


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
