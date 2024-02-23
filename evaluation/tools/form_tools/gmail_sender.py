from datetime import datetime
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from wizard_ai.clients import SendEmailPayload
from wizard_ai.conversational_engine.tools import GmailSender


class GmailSenderEvaluation(GmailSender):

    name = "GmailSender"
    description = """Useful to send emails from Gmail"""
    args_schema: Type[BaseModel] = SendEmailPayload

    chat_id: Optional[str] = None

    def _run_when_complete(
        self,
        to: str,
        subject: str,
        body: str
    ) -> str:
        return "OK"

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        """
        Use library faker to generate random data for the form.
        """

        import faker

        fake = faker.Faker()

        return {
            "to": fake.email(),
            "subject": fake.text(),
            "body": fake.text()
        }
