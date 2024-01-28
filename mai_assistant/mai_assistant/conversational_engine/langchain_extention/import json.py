import json
from typing import Any, Sequence, Type

from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import FunctionMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from mai_assistant.conversational_engine.langchain_extention.form_tool import \
    AgentState
from mai_assistant.conversational_engine.langchain_extention.intent_helpers import \
    filter_active_tools
from langchain import hub
from langchain.agents import create_openai_functions_agent


import os

os.environ["OPENAI_API_KEY"] = "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"

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Construct the OpenAI Functions agent
model = create_openai_functions_agent(ChatOpenAI(temperature=0, verbose=True), [], prompt)

while True:
    user_input = input("Enter your input: ")
    if user_input == "exit":
        break
    if user_input == "reset":
        model.reset()
        continue
    print(model.invoke(user_input))