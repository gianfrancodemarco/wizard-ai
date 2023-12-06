import os
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger

from openai._client import OpenAI

logger = getLogger(__name__)


class LLMType(Enum):
    GPT3_5_TURBO = "gpt3"
    GPT3__5_TURBO_INSTRUCT = "gpt3-instruct"


LLM_MODEL_NAMES = {
    LLMType.GPT3_5_TURBO.value: "gpt-3.5-turbo",
    LLMType.GPT3__5_TURBO_INSTRUCT.value: "gpt-3.5-turbo-instruct"
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
        self.__model_name__ = None
        self.__COST_PER_INPUT_TOKEN__ = None
        self.__COST_PER_OUTPUT_TOKEN__ = None
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

    def estimate_costs(self):
        return (self.__input_tokens__ * self.__COST_PER_INPUT_TOKEN__) + (self.__output_tokens__ * self.__COST_PER_OUTPUT_TOKEN__)

    def print_costs(self):
        logger.info(f"Input tokens: {self.__input_tokens__}")
        logger.info(f"Output tokens: {self.__output_tokens__}")
        logger.info(f"Estimated costs: {self.estimate_costs()}")


class GPT3TurboClient(GPT3Client):
    def __init__(self):
        super().__init__()
        self.__model_name__ = LLM_MODEL_NAMES[LLMType.GPT3_5_TURBO.value]
        self.__COST_PER_INPUT_TOKEN__ = 0.001 / 1000
        self.__COST_PER_OUTPUT_TOKEN__ = 0.002 / 1000
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

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


class GPT3TurboInstructClient(GPT3Client):
    def __init__(self):
        super().__init__()
        self.__model_name__ = LLM_MODEL_NAMES[LLMType.GPT3__5_TURBO_INSTRUCT.value]
        self.__COST_PER_INPUT_TOKEN__ = 0.0015 / 1000
        self.__COST_PER_OUTPUT_TOKEN__ = 0.002 / 1000
        self.__input_tokens__, self.__output_tokens__, self.__total_tokens__ = 0, 0, 0

    def prompt_completion(self, prompt):
        raise NotImplementedError


class LLMClientFactory():
    @staticmethod
    def create(
        llm_type: str
    ):
        if llm_type == LLMType.GPT3_5_TURBO.value:
            return GPT3TurboClient()
        elif llm_type == LLMType.GPT3__5_TURBO_INSTRUCT.value:
            return GPT3TurboInstructClient()
        else:
            raise Exception("Unknown LLM type: {}".format(llm_type))
