from typing import Optional, Type

from langchain.tools.base import StructuredTool
from pydantic import BaseModel

from wizard_ai.clients import GetEmailsPayload

from .structured_tool_for_evaluation import StructuredToolForEvaluation


class GmailRetriever(StructuredTool, StructuredToolForEvaluation):
    name = "GmailRetriever"
    description = """Useful to retrieve emails from Gmail"""
    args_schema: Type[BaseModel] = GetEmailsPayload

    return_direct = True
    chat_id: Optional[str] = None