# import os

# from langchain.agents.structured_chat.base import *
# from langchain.memory.chat_memory import BaseMemory
# from langchain_core.prompts.chat import ChatMessagePromptTemplate

# from mai_assistant.clients.llm import LLM_MODELS, LLMClientFactory
# from mai_assistant.conversational_engine.langchain_extention import (
#     FormStructuredChatExecutor, AgentState)
# from mai_assistant.conversational_engine.tools import *


# class Agent:
#     """
#         Agent that uses an LLM to respond to user messages.

#         Args:
#             memory (BaseMemory): Memory object.
#             chat_id (str): Chat ID. Used by some tools to store and retrieve data from memory.

#     """

#     def __init__(
#         self,
#         memory: BaseMemory,
#         chat_id: str,
#         llm_model: str = LLM_MODELS.GPT3_5_TURBO.value,
#         context: AgentState = AgentState()
#     ):

#         self.memory = memory

#         self.tools = [
#             # Calculator(),
#             # RandomNumberGenerator(),
#             # GoogleSearch(),
#             GoogleCalendarCreator(chat_id=chat_id),
#             GoogleCalendarRetriever(chat_id=chat_id),
#             # GmailRetriever(chat_id=chat_id),
#             # DateCalculatorTool()
#         ]

#         self.agent_chain = FormStructuredChatExecutor.from_llm_and_tools(
#             llm=LLMClientFactory.create(
#                 llm_model,
#                 url=os.environ.get('LLM_URL')
#             ),
#             tools=self.tools,
#             memory=self.memory,
#             context=context,
#             verbose=True
#         )
