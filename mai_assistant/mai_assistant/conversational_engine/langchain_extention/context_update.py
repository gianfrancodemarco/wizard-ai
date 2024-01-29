from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ValidationError

from .form_tool import AgentState
from pydantic import BaseModel, Field

class ContextUpdatePayload(BaseModel):
    values: Dict[str, Any] = Field(description="A dict with the values to update. The keys must match those if the form.")


class ContextUpdate(BaseTool):
    name = "ContextUpdate"
    description = """Stores information given by the user in the form."""
    args_schema: Type[BaseModel] = ContextUpdatePayload
    handle_tool_error = True

    context: Optional[AgentState] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        for key, value in kwargs['values'].items():
            try:
                setattr(self.context.form, key, value)
            except ValidationError as e:
                # build a string message with the error
                messages = []
                for error in e.errors():
                    messages.append(
                        f"Error at {error['loc'][0]}: {error['msg']}")
                message = "\n".join(messages)
                raise ToolException(message)
        return "Context updated"
