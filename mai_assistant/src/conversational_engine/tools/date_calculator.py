from datetime import datetime, timedelta
from typing import Optional

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel


class DateCalculatorToolPayload(BaseModel):
    operation: str


class DateCalculatorTool(BaseTool):
    name = "DateCalculatorTool"
    description = """
    A tool to make calculations with dates. The base date is the current date, and you can add or subtract time.
    The operation is a string with the following format: "[+,-] years months days hours minutes seconds"
    """

    def _run(
        self,
        operation: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        now = datetime.now()
        operation = operation.split()
        if len(operation) != 7:
            raise ValueError(
                "The operation must be a string with the following format: '[+,-] years months days hours minutes seconds'")
    
        sign = operation[0]
        years = int(operation[1])
        months = int(operation[2])
        days = int(operation[3])
        hours = int(operation[4])
        minutes = int(operation[5])
        seconds = int(operation[6])

        _timedelta = timedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
        
        if sign == "+":
            result = now + _timedelta
        elif sign == "-":
            result = now - _timedelta
        else:
            raise ValueError(
                "The operation must be a string with the following format: '[+,-] years months days hours minutes seconds'")
        
        return result.strftime("%d/%m/%Y %H:%M:%S")