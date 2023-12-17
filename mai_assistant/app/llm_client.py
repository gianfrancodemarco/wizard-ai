import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger

import requests
from langchain_core.language_models.llms import LLM
from openai._client import OpenAI
from typing import Any, Mapping, Optional, List

from langchain.chat_models import ChatOpenAI
from langchain_core.language_models.llms import LLM

from typing import Any, List, Mapping, Optional

from pydantic import Field
from time import time

logger = getLogger(__name__)


class LLM_MODELS(Enum):
    GPT3_5_TURBO = "gpt-3.5-turbo"
    LLAMA_2_7B_CHAT_HF = "llama-2-7B-chat-hf"

    @classmethod
    def keys(cls):
        return [e.name for e in cls]

    @classmethod
    def values(cls):
        return [e.value for e in cls]


OPEN_AI_CHAT_COMPLETION_MODELS = [
    LLM_MODELS.GPT3_5_TURBO.value
]


class LLMClient(LLM, ABC):

    COST_PER_INPUT_TOKEN: float = Field(default=0.0)
    COST_PER_OUTPUT_TOKEN: float = Field(default=0.0)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    # LLM interface methods

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:

        logger.info(f"----------------------------------------")
        logger.info(f"Querying model: {self._llm_type}")
        start = time()
        response = self.prompt_completion(
            prompt=prompt,
            **kwargs
        )
        end = time()
        logger.info(f"Response time: {end - start}")
        logger.info(f"----------------------------------------")
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": 0}

    @property
    def _llm_type(self) -> str:
        return "custom"

    ###

    @abstractmethod
    def prompt_completion(self, prompt, **kwargs):
        pass

    def estimate_costs(self):
        return (self.input_tokens * self.COST_PER_INPUT_TOKEN) + (self.output_tokens * self.COST_PER_OUTPUT_TOKEN)

    def print_costs(self):
        logger.info(f"Input tokens: {self.input_tokens}")
        logger.info(f"Output tokens: {self.output_tokens}")
        logger.info(f"Estimated costs: {self.estimate_costs()}")


class OpenAIChatCompletitionClient(LLMClient):

    api_key: str = Field(default=os.getenv("API_KEY"))
    model_name: str = Field()
    COST_PER_INPUT_TOKEN: float = Field(default=0.001 / 1000)
    COST_PER_OUTPUT_TOKEN: float = Field(default=0.002 / 1000)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    def prompt_completion(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 250,
        temperature: float = 0,
        **kwargs
    ):
        client = OpenAI(
            api_key=self.api_key
        )
        # model_kwargs={"stop": "\n"},
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Update token counts
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        self.total_tokens += response.usage.total_tokens

        self.print_costs()
        response_text = response.choices[0].message.content

        return response_text


class LlamaChatCompletitionClient(LLMClient):

    url: str = Field()
    endpoint: str = Field(default="/v1/completition")

    def prompt_completion(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_new_tokens: int = 250,
        top_k: int = 1,
        **kwargs
    ):
        response = requests.post(
            url=self.url + self.endpoint,
            json={
                "prompt": prompt,
                "configs": {
                    "stop": stop,
                    "max_new_tokens": max_new_tokens,
                    "top_k": top_k
                }
            }
        )
        
        try:
            response = response.json()
        except Exception as e:
            logger.error(f"Error parsing response: {response}")
            raise e
        
        # Update token counts
        self.input_tokens += response['input_tokens']
        self.output_tokens += response['output_tokens']
        self.total_tokens += response['total_tokens']

        response_text = response['completition']
        return response_text


class LLMClientFactory():
    @staticmethod
    def create(
        model_name: str,
        **kwargs
    ):
        if model_name in OPEN_AI_CHAT_COMPLETION_MODELS:
            return OpenAIChatCompletitionClient(
                model_name=model_name
            )
        elif model_name == LLM_MODELS.LLAMA_2_7B_CHAT_HF.value:
            return LlamaChatCompletitionClient(
                url=kwargs.get("url", None)
            )
        else:
            raise Exception("Unknown LLM type: {}".format(model_name))
