from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
import os

import langsmith
from langchain import chat_models, prompts, smith
from langchain.schema import output_parser
from langchain_openai import ChatOpenAI

from mai_assistant.conversational_engine.langchain_extention.mai_assistant_graph import (
    MAIAssistantGraph)
from mai_assistant.conversational_engine.tools import *

os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "ls__fd0fe695f6734dedbf8536715ff000ee"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

tools = [
    GoogleSearch(),
    GoogleCalendarCreator(chat_id=''),
    GoogleCalendarRetriever(chat_id=''),
    GmailRetriever(chat_id=''),
    PythonCodeInterpreter(),
]

_llm = ChatOpenAI(temperature=0)
llm_with_tools = _llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
).bound

eval_config = smith.RunEvalConfig(
    evaluators=[
        "cot_qa"
    ],
    custom_evaluators=[],
    eval_llm=chat_models.ChatOpenAI(model="gpt-4", temperature=0)
)

client = langsmith.Client()
chain_results = client.run_on_dataset(
    dataset_name="LLM BASE",
    llm_or_chain_factory=llm_with_tools,
    evaluation=eval_config,
    project_name="test-kindly-farm-37",
    concurrency_level=5,
    verbose=True,
)
