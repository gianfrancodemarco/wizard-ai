from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .form_tool import FormStructuredChatExecutorContext
from .tool_dummy_payload import ToolDummyPayload


class ContextReset(BaseTool):
    name = "ContextReset"
    description = """Call this tool when the user doesn't want to fill the form anymore."""
    args_schema: Type[BaseModel] = ToolDummyPayload

    context: Optional[FormStructuredChatExecutorContext] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self.context.active_form_tool = None
        self.context.form = None
        return "Context reset. Form cleared. Ask the user what he wants to do next."
