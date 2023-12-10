import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger

import requests
from openai._client import OpenAI

logger = getLogger(__name__)


class LLM_MODELS(Enum):
    GPT3_5_TURBO = "gpt-3.5-turbo"
    LLAMA_2_7B_CHAT_HF = "llama-2-7B-chat-hf"


OPEN_AI_CHAT_COMPLETION_MODELS = [
    LLM_MODELS.GPT3_5_TURBO.value
]


class LLMClient(ABC):
    
    def __init__(self):
        self.__COST_PER_INPUT_TOKEN__ = 0.0
        self.__COST_PER_OUTPUT_TOKEN__ = 0.0
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

    @abstractmethod
    def prompt_completion(self, prompt):
        pass

    def estimate_costs(self):
        return (self.__input_tokens__ * self.__COST_PER_INPUT_TOKEN__) + (self.__output_tokens__ * self.__COST_PER_OUTPUT_TOKEN__)

    def print_costs(self):
        logger.info(f"Input tokens: {self.__input_tokens__}")
        logger.info(f"Output tokens: {self.__output_tokens__}")
        logger.info(f"Estimated costs: {self.estimate_costs()}")


class OpenAIChatCompletitionClient(LLMClient):
    def __init__(
        self,
        model_name: str
    ):
        super().__init__()
        self.__api_key__ = os.getenv("API_KEY")
        self.__model_name__ = model_name
        self.__COST_PER_INPUT_TOKEN__ = 0.001 / 1000
        self.__COST_PER_OUTPUT_TOKEN__ = 0.002 / 1000
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

    def prompt_completion(
        self,
        prompt: str,
        max_tokens: int = 250,
        temperature: float = 0,
    ):

        logger.info(f"----------------------------------------")

        client = OpenAI(
            api_key=self.__api_key__
        )
        # model_kwargs={"stop": "\n"},
        response = client.chat.completions.create(
            model=self.__model_name__,
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
        self.__input_tokens__ += response.usage.prompt_tokens
        self.__output_tokens__ += response.usage.completion_tokens
        self.__total_tokens__ += response.usage.total_tokens

        self.print_costs()
        response_text = response.choices[0].message.content

        logger.info(f"Using model: {self.__model_name__}")
        logger.info(f"PROMPT: {prompt}")
        logger.info(f"RESPONSE: {response_text}")
        logger.info(f"----------------------------------------")

        return response_text


class LlamaChatCompletitionClient(LLMClient):

    def __init__(
        self,
        url: str
    ):
        super().__init__()
        self.url = url
        self.endpoint = "/v1/completition"

    def prompt_completion(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        top_k: int = 1
    ):
        logger.info(f"----------------------------------------")

        response = requests.post(
            url=self.url + self.endpoint,
            json={
                "prompt": prompt,
                "configs": {
                    "stop": ["\n"],
                    "max_new_tokens": max_new_tokens,
                    "top_k": top_k
                }
            }
        )
        response = response.json()

        # Update token counts
        self.__input_tokens__ += response['input_tokens']
        self.__output_tokens__ += response['output_tokens']
        self.__total_tokens__ += response['total_tokens']

        self.print_costs()
        response_text = response['completition']

        logger.info(f"Using model: llama-2-7B-chat-hf")
        logger.info(f"PROMPT: {prompt}")
        logger.info(f"RESPONSE: {response_text}")
        logger.info(f"----------------------------------------")

        return response_text

    def estimate_costs(self):
        raise NotImplementedError

    def print_costs(self):
        raise NotImplementedError


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