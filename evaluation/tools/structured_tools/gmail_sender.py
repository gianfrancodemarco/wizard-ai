from typing import Optional, Type

from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from wizard_ai.clients import SendEmailPayload


class GmailSender(StructuredTool):

    name = "GmailSender"
    description = """Useful to send emails from Gmail"""
    args_schema: Type[BaseModel] = SendEmailPayload

    chat_id: Optional[str] = None

    def _run(
        self,
        *args,
        **kwargs,
    ) -> str:
        return "OK"