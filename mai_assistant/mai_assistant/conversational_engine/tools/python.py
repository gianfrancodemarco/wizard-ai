from typing import Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class PythonInput(BaseModel):
    code: str = Field(description="A valid snippet of Python code to be executed using exec(). Must store the final value in a variable called `result`")

class Python(BaseTool):
    name = "Python"
    description = "Python code execution tool"
    args_schema: Type[BaseModel] = PythonInput

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        
        local_vars = {}
        exec(code, {}, local_vars)
        result_value = local_vars.get('result')

        if not result_value:
            raise ValueError("The last operation of the code provided must store the output of the code in a variable called result.")
        
        return str(result_value)

    def get_tool_start_message(self, input: dict) -> str:
        payload = PythonInput(**input)
        return f"Executing code:\n\n <code>{payload.code}</code>"