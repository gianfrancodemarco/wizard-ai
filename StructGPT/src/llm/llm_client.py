import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger

from openai._client import OpenAI

logger = getLogger(__name__)


class LLM_MODELS(Enum):
    GPT3_5_TURBO = "gpt-3.5-turbo"


OPEN_AI_CHAT_COMPLETION_MODELS = [
    LLM_MODELS.GPT3_5_TURBO.value
]


class LLMClient(ABC):

    def __init__(
        self
    ):
        self.api_key = os.getenv("API_KEY")

    @abstractmethod
    def prompt_completion(self, prompt):
        pass

    def estimate_costs(self):
        raise NotImplementedError

    def print_costs(self):
        raise NotImplementedError


class OpenAIChatCompletitionClient(LLMClient):
    def __init__(
        self,
        model_name: str
    ):
        super().__init__()
        self.__model_name__ = model_name
        self.__COST_PER_INPUT_TOKEN__ = 0.001 / 1000
        self.__COST_PER_OUTPUT_TOKEN__ = 0.002 / 1000
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

    def estimate_costs(self):
        return (self.__input_tokens__ * self.__COST_PER_INPUT_TOKEN__) + (self.__output_tokens__ * self.__COST_PER_OUTPUT_TOKEN__)

    def print_costs(self):
        logger.info(f"Input tokens: {self.__input_tokens__}")
        logger.info(f"Output tokens: {self.__output_tokens__}")
        logger.info(f"Estimated costs: {self.estimate_costs()}")

    def prompt_completion(self, prompt):

        logger.info(f"----------------------------------------")

        client = OpenAI(
            api_key=self.api_key
        )
        response = client.chat.completions.create(
            model=self.__model_name__,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
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


class LLMClientFactory():
    @staticmethod
    def create(
        model_name: str
    ):
        if model_name in OPEN_AI_CHAT_COMPLETION_MODELS:
            return OpenAIChatCompletitionClient(
                model_name=model_name
            )
        else:
            raise Exception("Unknown LLM type: {}".format(model_name))
