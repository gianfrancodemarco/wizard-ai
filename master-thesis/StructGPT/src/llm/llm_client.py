import os
from abc import ABC, abstractmethod

from openai._client import OpenAI


@abstractmethod
class LLMClient():

    def __init__(
        self
    ):
        self.api_key = os.getenv("API_KEY")

    def prompt_completion(self, prompt):
        pass


class GPT3Client(LLMClient):
    def prompt_completion(self, prompt):

        client = OpenAI(
            api_key=self.api_key
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        response_text = response.choices[0].message.content
        return response_text

class LLMClientFactory():
    @staticmethod
    def create(
        llm_type: str
    ):
        if llm_type == "gpt3":
            return GPT3Client()
        else:
            raise Exception("Unknown LLM type: {}".format(llm_type))
