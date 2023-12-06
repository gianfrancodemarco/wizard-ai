import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger

from openai._client import OpenAI

logger = getLogger(__name__)


class LLMType(Enum):
    GPT3 = "gpt3"


LLM_MODEL_NAMES = {
    LLMType.GPT3.value: "gpt-3.5-turbo"
}


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


class GPT3Client(LLMClient):

    def __init__(self):
        super().__init__()
        self.__COST_PER_INPUT_TOKEN__ = 0.001 / 1000
        self.__COST_PER_OUTPUT_TOKEN__ = 0.002 / 1000
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

    def estimate_costs(self):
        return (self.__input_tokens__ * self.__COST_PER_INPUT_TOKEN__) + (self.__output_tokens__ * self.__COST_PER_OUTPUT_TOKEN__)

    def print_costs(self):
        logger.debug(f"Input tokens: {self.__input_tokens__}")
        logger.debug(f"Output tokens: {self.__output_tokens__}")
        logger.debug(f"Estimated costs: {self.estimate_costs()}")

    def prompt_completion(self, prompt):

        client = OpenAI(
            api_key=self.api_key
        )
        response = client.chat.completions.create(
            model=LLM_MODEL_NAMES[LLMType.GPT3.value],
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
        return response_text

class LLMClientFactory():
    @staticmethod
    def create(
        llm_type: str
    ):
        if llm_type == LLMType.GPT3.value:
            return GPT3Client()
        else:
            raise Exception("Unknown LLM type: {}".format(llm_type))