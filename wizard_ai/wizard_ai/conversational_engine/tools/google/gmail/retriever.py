import pickle
from typing import Optional, Type

from langchain.tools.base import StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel

from wizard_ai.clients import (GetEmailsPayload, GoogleClient,
                                   get_redis_client)
from wizard_ai.constants import RedisKeys


class GmailRetriever(StructuredTool):

    name = "GmailRetriever"
    description = """Useful to retrieve emails from Gmail"""
    args_schema: Type[BaseModel] = GetEmailsPayload
    
    return_direct = True
    chat_id: Optional[str] = None

    def _run(
        self,
        number_of_emails: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        credentials = get_redis_client().hget(
            self.chat_id,
            RedisKeys.GOOGLE_CREDENTIALS.value
        )
        credentials = pickle.loads(credentials)

        google_client = GoogleClient(credentials)
        payload = GetEmailsPayload(
            number_of_emails=number_of_emails
        )
        return google_client.get_emails_html(payload)

    def get_tool_start_message(self, input: dict) -> str:
        payload = GetEmailsPayload(**input)
        return f"Retrieving the last {payload.number_of_emails} emails from Gmail"
