from typing import Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    computation: str = Field(
        description="A valid computation string to be executed using eval()")


class Calculator(BaseTool):
    name = "Calculator"
    description = "Useful to resolve mathematical calculations and expressions"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
            self,
            computation: str,
            run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return eval(computation)
