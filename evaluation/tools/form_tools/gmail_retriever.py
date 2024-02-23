from datetime import datetime
from typing import Dict, Union

import faker

from wizard_ai.conversational_engine.tools import GmailRetriever

from .form_tool_for_evaluation import FormToolForEvaluation

fake = faker.Faker()

class GmailRetrieverEvaluation(GmailRetriever, FormToolForEvaluation):
    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:        
        return {
            "number_of_emails": fake.random_int(min=1, max=10)
        }