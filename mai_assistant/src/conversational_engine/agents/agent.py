import datetime
import os
from abc import ABC
from textwrap import dedent
from typing import List

from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.agents.structured_chat.prompt import SUFFIX
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseMemory
from langchain_core.language_models.llms import LLM
from langchain_core.prompts.chat import ChatMessagePromptTemplate
from langchain_core.tools import Tool

from mai_assistant.src.clients.llm import LLM_MODELS, LLMClientFactory
from mai_assistant.src.conversational_engine.langchain_extention import (
    FormStructuredChatExecutor, FormStructuredChatExecutorContext)
from mai_assistant.src.conversational_engine.tools import *


def get_prefix():
    """
    We use a function here to avoid the prefix being cached in the module, so that the current time is always up to date.
    """

    return dedent(f"""
        Respond to the human as helpfully and accurately as possible.
        If the user request is not clear, ask for clarification (using the final answer tool).
        Today is: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        \n
        You have access to the following tools:"""
                  )

def get_suffix():
    return dedent(f"""
        {SUFFIX}. When calling a tool, use only inputs EXPLICITLY given by the user.
    """)

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
        **kwargs
    ):

        self.memory = memory

        self.tools = [
            # Calculator(),
            # RandomNumberGenerator(),
            # GoogleSearch(),
            GoogleCalendarCreatorActivator(chat_id=chat_id),
            GoogleCalendarRetrieverActivator(chat_id=chat_id),
            # GoogleCalendarCreator(chat_id=chat_id),
            # GoogleCalendarRetriever(chat_id=chat_id),
            # GmailRetriever(chat_id=chat_id),
            # DateCalculatorTool()
        ]

        self.llm = LLMClientFactory.create(
            llm_model,
            url=os.environ.get('LLM_URL')
        )

        llm_chain_builder = lambda tools: Agent.llm_chain_builder(
            llm=self.llm,
            tools=tools,
            memory_prompts=self.get_memory_prompts()
        )

        agent_builder = lambda llm_chain, tools: Agent.agent_builder(
            llm_chain=llm_chain,
            tools=tools
        )

        self.agent_chain = FormStructuredChatExecutor.from_tools_and_builders(
            llm_chain_builder=llm_chain_builder,
            agent_builder=agent_builder,
            tools=self.tools,
            memory=self.memory,
            context=kwargs.get("context", FormStructuredChatExecutorContext()),
            verbose=True
        )

    @staticmethod
    def llm_chain_builder(
        llm: LLM,
        tools: List[Tool],
        memory_prompts: List[ChatMessagePromptTemplate]
    ):
        return LLMChain(
            llm=llm,
            prompt=StructuredChatAgent.create_prompt(
                tools=tools,
                prefix=get_prefix(),
                suffix=get_suffix(),
                memory_prompts=memory_prompts
            ),
            verbose=True
        )

    @staticmethod
    def agent_builder(
        llm_chain: LLMChain,
        tools: List[Tool],
    ):
        return StructuredChatAgent(
            llm_chain=llm_chain,
            tools=tools,
            verbose=True
        )

    def get_prefix(self):
        """
        We use a function here to avoid the prefix being cached in the module, so that the current time is always up to date.
        """

        return dedent(f"""
            Respond to the human as helpfully and accurately as possible.
            If the user request is not clear, ask for clarification (using the final answer tool).
            Today is: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            \n
            You have access to the following tools:"""
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