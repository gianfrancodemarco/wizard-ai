import os

from langchain.memory.chat_memory import BaseMemory

from mai_assistant.src.clients.llm import (LLM_MODELS,
                                           OPEN_AI_CHAT_COMPLETION_MODELS)
from mai_assistant.src.conversational_engine.tools import *

from .agent import Agent
from .gpt import GPTAgent
from .llama import LLamaAgent


class AgentFactory():
    @staticmethod
    def create(
        memory: BaseMemory,
        chat_id: str,
        model_name: str = os.getenv("model_name", LLM_MODELS.GPT3_5_TURBO.value),
    ) -> Agent:
        if model_name in OPEN_AI_CHAT_COMPLETION_MODELS:
            return GPTAgent(
                chat_id=chat_id,
                memory=memory,
                llm_model=model_name
            )
        elif model_name == LLM_MODELS.LLAMA_2_7B_CHAT_HF.value:
            return LLamaAgent(
                chat_id=chat_id,
                memory=memory,
                llm_model=model_name
            )
        else:
            raise Exception("Unknown Agent type: {}".format(model_name))