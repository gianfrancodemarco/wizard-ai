from typing import Any, Type

from pydantic import BaseModel

from wizard_ai.conversational_engine.intent_agent.form_tool import \
    FormTool, BaseTool


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
