import base64
import os
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, List, Optional

from dateutil.parser import parse
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field, field_validator

from mai_assistant.html_processor import HtmlProcessor

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')


class CreateCalendarEventPayload(BaseModel):

    # TODO: 
    # Let's make everything Optional manually for now
    # Users can implement validations to make sure the required fields are
    # present
    summary: Optional[str] = Field(
        default=None,
        description="Title of the event",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the event",
    )
    start: Optional[datetime] = Field(
        default=None,
        description="Start date of the event",
    )
    end: Optional[datetime] = Field(
        default=None,
        description="End date of the event",
    )

    @field_validator("start", "end", mode="before")
    def parse_date(cls, v, values):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return parse(v)


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


class GetEmailsPayload(BaseModel):
    number_of_emails: Optional[int] = Field(
        default=3,
        description="Number of emails to retrieve"
    )


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

    def get_calendar_events_html(
        self,
        data: GetCalendarEventsPayload
    ) -> str:
        events = self.get_calendar_events(data)
        events_string = self.__events_result_to_html_string(events)
        return events_string

    def __events_result_to_html_string(self, events: List[Any]) -> str:
        events_string = ""
        for idx, event in enumerate(events):
            event_start = event['start'].get(
                'dateTime', event['start'].get('date'))
            event_end = event['end'].get('dateTime', event['end'].get('date'))
            event_summary = event['summary']
            event_link = event['htmlLink']
            event_string = f"{idx+1}. {event_start} - <a href=\"{event_link}\">{event_summary}</a>\n"
            events_string += event_string

        if not events_string:
            events_string = "No events found"

        return events_string

    def get_emails(
        self,
        payload: GetEmailsPayload
    ):
        service = build('gmail', 'v1', credentials=self.credentials)

        messages_list = service.users().messages().list(
            userId='me', maxResults=payload.number_of_emails).execute()
        messages = messages_list.get('messages', [])

        result = []

        for message in messages:
            msg = service.users().messages().get(
                userId="me", id=message['id']).execute()

            # Extracting sender, time, and content from the message
            sender = next((header['value'] for header in msg['payload']
                          ['headers'] if header['name'] == 'From'), None)
            time = next((header['value'] for header in msg['payload']
                        ['headers'] if header['name'] == 'Date'), None)
            subject = next((header['value'] for header in msg['payload']
                           ['headers'] if header['name'] == 'Subject'), None)

            full_content = ''
            GET_FULL_CONTENT = False
            if GET_FULL_CONTENT:
                # Check if the email is in multipart format
                if 'multipart' in msg['payload']['mimeType']:
                    parts = msg['payload']['parts']
                    full_content = ''
                    for part in parts:
                        if 'body' in part and 'data' in part['body']:
                            data = part['body']['data']
                            decoded_data = base64.urlsafe_b64decode(
                                data.encode('UTF-8')).decode('UTF-8')
                            full_content += decoded_data
                else:
                    # If the email is in plain text format
                    full_content = base64.urlsafe_b64decode(
                        msg['payload']['body']['data'].encode('UTF-8')).decode('UTF-8')

                # Clear the full content
                full_content = HtmlProcessor.clear_html(full_content)
            else:
                full_content = msg['snippet']

            result.append({
                "sender": sender,
                "time": time,
                "subject": subject,
                "content": full_content
            })

        return result

    def get_emails_html(
        self,
        payload: GetEmailsPayload
    ) -> str:
        emails = self.get_emails(payload)
        emails_string = self.__emails_result_to_html_string(emails)
        return emails_string

    def __emails_result_to_html_string(self, emails: List[Any]) -> str:

        emails_string = ""
        for idx, email in enumerate(emails):
            emails_string += dedent(f"""
                {idx+1}. Subject: {email['subject']}
                Sender: {email['sender']}
                Time: {email['time']}
                Content: {email['content']}
                \n\n
            """)

        if not emails_string:
            emails_string = "No emails found"

        # Remove < and > from the string for HTML compliance
        emails_string = emails_string.replace("<", "").replace(">", "")

        return emails_string
