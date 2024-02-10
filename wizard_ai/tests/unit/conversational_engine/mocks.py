from typing import Any, Type

from pydantic import BaseModel

from wizard_ai.conversational_engine.intent_agent.intent_tool import (
    BaseTool, IntentTool)


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


class _DummyPayloadWithFields(BaseModel):
    name: set
    age: int


class MockIntentTool(IntentTool):
    name = "MockIntentTool"
    description = "MockIntentTool description"
    args_schema: Type[BaseModel] = _DummyPayload

    def _run_when_complete(self) -> Any:
        pass


class MockIntentToolWithFields(IntentTool):
    name = "MockIntentToolWithFields"
    description = "MockIntentToolWithFields description"
    args_schema: Type[BaseModel] = _DummyPayloadWithFields

    def _run_when_complete(self) -> Any:
        pass
