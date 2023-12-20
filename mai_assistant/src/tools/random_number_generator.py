from typing import Optional, Type

from langchain.tools.base import StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class RandomNumberGeneratorInput(BaseModel):
    min: str = Field(default=0)
    max: str = Field(default=100)

class RandomNumberGenerator(StructuredTool):
    name = "RandomNumberGenerator"
    description = "Useful to do calculations"
    args_schema: Type[BaseModel] = RandomNumberGeneratorInput

    def _run(
        self,
        min: str,
        max: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        import random
        return random.randint(int(min), int(max))