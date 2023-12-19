from typing import Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class GoogleSearchInput(BaseModel):
    quite: str = Field()


class GoogleSearch(BaseTool):
    name = "GoogleSearch"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("running google search")
        # return search.run(query)


tools = [
    GoogleSearch()
]
