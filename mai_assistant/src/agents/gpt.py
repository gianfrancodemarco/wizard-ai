import os

from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseMemory

from mai_assistant.src.llm_client import LLM_MODELS, LLMClientFactory
from mai_assistant.src.tools import Calculator, RandomNumberGenerator, Search


class GPTAgent():

    def __init__(self, memory: BaseMemory):
        self.memory = memory
        self.tools = [
            Calculator(),
            RandomNumberGenerator(),
            Search()
        ]
        self.llm = LLMClientFactory.create(
            LLM_MODELS.GPT3_5_TURBO.value,
            url=os.environ.get('LLM_URL')
        )
        self.llm_chain = LLMChain(
            llm=self.llm, prompt=StructuredChatAgent.create_prompt(tools=self.tools))
        self.agent = StructuredChatAgent(
            llm_chain=self.llm_chain,
            tools=self.tools,
            verbose=True
        )
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory
        )
