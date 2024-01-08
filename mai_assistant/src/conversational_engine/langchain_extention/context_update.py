from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .form_tool import FormStructuredChatExecutorContext


class ContextUpdatePayload(BaseModel):
    values: Dict[str, Any]

class ContextUpdate(BaseTool):
    name = "ContextUpdate"
    description = """Useful to store the information given by the user."""
    args_schema: Type[BaseModel] = ContextUpdatePayload

    context: Optional[FormStructuredChatExecutorContext] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        for key, value in kwargs['values'].items():
            setattr(self.context.form, key, value)
        return "Context updated"
