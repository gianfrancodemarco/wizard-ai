from typing import Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel

from wizard_ai.clients.google_search import (GoogleSearchClient,
                                                 GoogleSearchClientPayload)


class GoogleSearch(BaseTool):

    name: str = "GoogleSearch"
    description: str = "Useful for searching the internet with Google to retrieve up to date information."
    args_schema: Type[BaseModel] = GoogleSearchClientPayload
    google_search_client: Optional[GoogleSearchClient] = None

    def __init__(self):
        super().__init__()
        self.google_search_client = GoogleSearchClient()

    def _run(
        self,
        query: str,
        num_expanded_results: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        payload = GoogleSearchClientPayload(
            query=query,
            num_expanded_results=num_expanded_results
        )
        results = self.google_search_client.search(payload)
        return results

    def get_tool_start_message(self, input: dict) -> str:
        payload = GoogleSearchClientPayload(**input)

        return "Searching the internet with query: " + payload.query