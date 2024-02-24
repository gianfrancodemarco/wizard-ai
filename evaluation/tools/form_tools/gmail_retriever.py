from datetime import datetime
from typing import Dict, Union

import faker

from wizard_ai.conversational_engine.tools import GmailRetriever

fake = faker.Faker()

class GmailRetrieverEvaluation(GmailRetriever):
    
    def _run_when_complete(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:        
        return {
            "number_of_emails": fake.random_int(min=1, max=10)
        }