import os

from langchain.memory.chat_memory import BaseMemory

from mai_assistant.src.clients.llm import (LLM_MODELS,
                                           OPEN_AI_CHAT_COMPLETION_MODELS)

from .agent import Agent
from .gpt import GPTAgent
from .llama import LLamaAgent


class AgentFactory():
    @staticmethod
    def create(
        memory: BaseMemory,
        chat_id: str,
        model_name: str = os.getenv("model_name", LLM_MODELS.GPT3_5_TURBO.value),
        **kwargs
    ) -> Agent:
        if model_name in OPEN_AI_CHAT_COMPLETION_MODELS:
            return GPTAgent(
                chat_id=chat_id,
                memory=memory,
                llm_model=model_name,
                **kwargs
            )
        elif model_name == LLM_MODELS.LLAMA_2_7B_CHAT_HF.value:
            return LLamaAgent(
                chat_id=chat_id,
                memory=memory,
                llm_model=model_name,
                **kwargs
            )
        else:
            raise Exception("Unknown Agent type: {}".format(model_name))