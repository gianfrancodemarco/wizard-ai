from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool


# You can create the tool to pass to an agent
class Search(Tool):

    run_description: str = "Searching the internet with query: "

    def __init__(self):
        super().__init__(
            name="Internet Search",
            description="Search the internet for an answer to your question",
            run_description="Searching the internet with query: ",
            func=SerpAPIWrapper(
                params={
                    "engine": "google",
                    "gl": "it",
                    "hl": "it"
                }
            ).run
        )