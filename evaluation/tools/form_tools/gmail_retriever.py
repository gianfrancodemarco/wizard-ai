from datetime import datetime
from typing import Dict, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel

from wizard_ai.clients import GetEmailsPayload
from wizard_ai.conversational_engine.tools import GmailRetriever


class GmailRetrieverEvaluation(GmailRetriever):

    name = "GmailRetriever"
    description = """Useful to retrieve emails from Gmail"""
    args_schema: Type[BaseModel] = GetEmailsPayload

    return_direct = True
    chat_id: Optional[str] = None

    def _run_when_complete(
        self,
        number_of_emails: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return "OK"

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        """
        Use library faker to generate random data for the form.
        """

        import faker

        fake = faker.Faker()
        
        return {
            "number_of_emails": fake.random_int(min=1, max=10)
        }