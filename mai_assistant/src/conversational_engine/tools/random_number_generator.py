from typing import Optional, Type

from langchain.tools.base import StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class RandomNumberGeneratorInput(BaseModel):
    min: Optional[int] = Field(default=0)
    max: Optional[int] = Field(default=100)


class RandomNumberGenerator(StructuredTool):
    name = "RandomNumberGenerator"
    description = "Generates a random number between min and max; to use only when explicitly requested"
    args_schema: Type[BaseModel] = RandomNumberGeneratorInput

    def _run(
        self,
        min: int = 0,
        max: int = 100,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        import random
        return random.randint(min, max)
