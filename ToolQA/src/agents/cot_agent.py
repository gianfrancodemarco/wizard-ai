from typing import List

from agents.utils import *
from fewshots import COT, COT_REFLECT
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from prompts import (cot_reflect_agent_prompt, cot_reflect_prompt)
from agents.agent import Agent

import logging

log = logging.getLogger(__name__)


class CoTAgent(Agent):
    def __init__(
        self,
        prompt: 'easy' or 'hard',
        path: str,
        # question: str,
        # key: str,
        agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
        reflect_prompt: PromptTemplate = cot_reflect_prompt,
        cot_examples: str = COT,
        reflect_examples: str = COT_REFLECT,
        self_reflect_llm: BaseLLM = get_llm(),
        action_llm: BaseLLM = get_llm(),
    ) -> None:
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples
        self.reflect_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm
        self.reflections: List[str] = []
        self.reflections_str = ''
        self.answer = ''
        self.step_n: int = 0
        # self.reset()

    def run(
        self,
        question: str,
        answer: str,
        question_id: str,
        reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
    ) -> None:

        question_context = QuestionContext(
            question=question,
            question_id=question_id,
            answer=answer,
            reflexion_strategy=reflexion_strategy
        )

        # How do we know if it is correct during inference?
        # if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
        #     self.reflect(reflexion_strategy)
        # self.reset()

        self.step(question_context)

    def step(
        self,
        question_context: QuestionContext
    ) -> None:

        # Think
        question_context.scratchpad += f'\nThought: {self.prompt_agent(question_context=question_context)}'
        log.info(question_context.scratchpad.split('\n')[-1])

        # Act
        action = self.prompt_agent(question_context=question_context)
        question_context.scratchpad += f'\nAction: {action}'
        actions = action.split('-->')
        for action in actions:
            action_type, argument = self.parse_action(action)
            log.info(question_context.scratchpad.split('\n')[-1])
            self.act(action_type, argument)

    def prompt_reflection(self) -> str:
        return format_step(self.self_reflect_llm(self._build_reflection_prompt()))

    # def reset(self) -> None:
    #     self.scratchpad: str = ''
    #     self.finished = False

    def prompt_agent(
        self,
        question_context: QuestionContext
    ) -> str:
        prompt = self.agent_prompt.format(
            examples=self.cot_examples,
            reflections=self.reflections_str,
            context=question_context.context,
            question=question_context.question,
            scratchpad=question_context.scratchpad
        )
        answer = self.action_llm(prompt)
        formatted_answer = format_step(answer)
        return formatted_answer

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            context=self.context,
            question=self.question,
            scratchpad=self.scratchpad)

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
