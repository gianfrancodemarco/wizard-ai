from typing import Type

from langchain.tools.base import BaseTool
from pydantic import BaseModel

from mai_assistant.src.clients.google_search import (GoogleSearchClient,
                                                     GoogleSearchClientPayload)


class GoogleSearch(BaseTool):

    name: str = "GoogleSearch"
    description: str = "Useful for searching the internet with Google to retrieve up to date information."
    args_schema: Type[BaseModel] = GoogleSearchClientPayload

    def _run(
        self,
        query: str,
        num_expanded_results: int = 1,
    ) -> str:
        """Use the tool."""

        google_search_client = GoogleSearchClient()
        payload = GoogleSearchClientPayload(
            query=query,
            num_expanded_results=3
        )
        results = google_search_client.search(payload)
        return results

    def get_tool_start_message(self, input: dict) -> str:
        payload = GoogleSearchClientPayload(**input)

        return "Searching the internet with query: " + payload.query
