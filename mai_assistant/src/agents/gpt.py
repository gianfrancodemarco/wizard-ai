import os

from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseMemory

from mai_assistant.src.llm_client import LLM_MODELS, LLMClientFactory
from mai_assistant.src.tools.calculator import Calculator
from mai_assistant.src.tools.random_number_generator import \
    RandomNumberGenerator


def get_gpt_agent(memory: BaseMemory):
    
    tools = [
        Calculator(),
        RandomNumberGenerator()
    ]
    
    llm = LLMClientFactory.create(
        LLM_MODELS.GPT3_5_TURBO.value,
        url=os.environ.get('LLM_URL')
    )
    
    llm_chain = LLMChain(llm=llm, prompt=StructuredChatAgent.create_prompt(tools=tools))
    
    agent = StructuredChatAgent(
        llm_chain=llm_chain,
        tools=tools,
        verbose=True
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory
    )

    return agent_chain