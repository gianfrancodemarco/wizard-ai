from typing import Any, Type

from pydantic import BaseModel

from mai_assistant.conversational_engine.langchain_extention.form_tool import \
    FormTool
from mai_assistant.conversational_engine.langchain_extention.intent_helpers import \
    BaseTool


class MockBaseTool(BaseTool):
    name = "MockBaseTool"
    description = "MockBaseTool description"

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        pass


class _DummyPayload(BaseModel):
    pass


class MockFormTool(FormTool):
    name = "MockFormTool"
    description = "MockFormTool description"
    args_schema: Type[BaseModel] = _DummyPayload
