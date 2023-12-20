from typing import Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class GoogleSearchInput(BaseModel):
    quite: str = Field()


class GoogleSearch(BaseTool):
    name = "GoogleSearch"
    description = "Useful to search information on the internet"
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        import requests

        url = "https://google-search3.p.rapidapi.com/api/v1/search/q=" + query
        headers = {
            "x-rapidapi-host": "google-search3.p.rapidapi.com",
            "x-rapidapi-key": "7d6a6f1c6cmshc4c7b9c3e5c4e1dp1f4f7bjsn7d7f3b6a9e8f",
        }
        response = requests.request("GET", url, headers=headers)
        return response.text
    


tools = [
    GoogleSearch()
]