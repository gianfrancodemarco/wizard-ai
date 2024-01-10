import datetime
import os
from abc import ABC
from textwrap import dedent

from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.agents.structured_chat.base import *
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseMemory
from langchain_core.prompts.chat import ChatMessagePromptTemplate

from mai_assistant.src.clients.llm import LLM_MODELS, LLMClientFactory
from mai_assistant.src.conversational_engine.langchain_extention import (
    FormStructuredChatExecutor, FormStructuredChatExecutorContext,
    get_form_prompt)
from mai_assistant.src.conversational_engine.tools import *


class Agent(ABC):
    """s
        Agent that uses an LLM to respond to user messages.

        Args:
            memory (BaseMemory): Memory object.
            chat_id (str): Chat ID. Used by some tools to store and retrieve data from memory.

    """

    def __init__(
        self,
        memory: BaseMemory,
        chat_id: str,
        llm_model: str = LLM_MODELS.GPT3_5_TURBO.value,
        context: FormStructuredChatExecutorContext = FormStructuredChatExecutorContext()
    ):

        self.memory = memory

        self.tools = FormStructuredChatExecutor.filter_active_tools([
            # Calculator(),
            # RandomNumberGenerator(),
            # GoogleSearch(),
            GoogleCalendarCreator(chat_id=chat_id),
            GoogleCalendarRetriever(chat_id=chat_id),
            # GmailRetriever(chat_id=chat_id),
            # DateCalculatorTool()
        ], context)

        self.llm = LLMClientFactory.create(
            llm_model,
            url=os.environ.get('LLM_URL')
        )

        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=StructuredChatAgent.create_prompt(
                tools=self.tools,
                memory_prompts=self.get_memory_prompts()
            ),
            verbose=True
        )

        #TODO: deprecated, use create_structured_chat_agent
        self.agent = StructuredChatAgent(
            llm_chain=self.llm_chain,
            tools=self.tools,
            verbose=True
        )

        self.agent_chain = FormStructuredChatExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            context=context,
            verbose=True
        )

    def get_memory_prompts(self):
        return [
            ChatMessagePromptTemplate.from_template(
                role="Previous conversation",
                template=dedent("""
                    \n\n
                    {history}
                    \n\n
                """)
            )
        ]