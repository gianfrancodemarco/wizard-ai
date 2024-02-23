from typing import Optional, Type

from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from wizard_ai.clients import SendEmailPayload

from .structured_tool_for_evaluation import StructuredToolForEvaluation


class GmailSender(StructuredTool, StructuredToolForEvaluation):
    name = "GmailSender"
    description = """Useful to send emails from Gmail"""
    args_schema: Type[BaseModel] = SendEmailPayload
    chat_id: Optional[str] = None