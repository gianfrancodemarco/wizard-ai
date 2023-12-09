import logging
from typing import List

from agents.react_agent import ReactAgent
from agents.utils import *
from fewshots import REFLECTIONS
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from prompts import react_reflect_agent_prompt, reflect_prompt

log = logging.getLogger(__name__)


class ReactReflectAgent(ReactAgent):
    def __init__(
        self,
        question: str,
        key: str,
        max_steps: int = 20,
        agent_prompt: PromptTemplate = react_reflect_agent_prompt,
        reflect_prompt: PromptTemplate = reflect_prompt,
        react_llm: BaseLLM = get_llm(model_name="text-davinci-003"),
        reflect_llm: BaseLLM = get_llm(model_name="text-davinci-003")
    ) -> None:

        super().__init__(question, key, max_steps, agent_prompt, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''

    def run(self, reset=True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if (self.finished or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset)

    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad)
