from datetime import datetime
from typing import Dict, Union

import faker

from wizard_ai.conversational_engine.tools import GmailSender

from .form_tool_for_evaluation import FormToolForEvaluation

fake = faker.Faker()

class GmailSenderEvaluation(GmailSender, FormToolForEvaluation):
    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        return {
            "to": fake.email(),
            "subject": fake.text(),
            "body": fake.text()
        }
