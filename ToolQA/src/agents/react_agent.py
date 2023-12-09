import logging

import tiktoken
from agents.agent import Agent
from agents.utils import *
from fewshots import TOOLQA_EASY8, TOOLQA_HARD3
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from prompts import react_agent_prompt
from tools.graph import graphtools
from tools.table import tabtools

log = logging.getLogger(__name__)


class ReactAgent(Agent):
    def __init__(
        self,
        prompt: 'easy' or 'hard',
        path: str,
        max_steps: int = 20,
        agent_prompt: PromptTemplate = react_agent_prompt,
        react_llm: BaseLLM = get_llm()
    ) -> None:

        self.answer = ''
        # self.question = question
        # self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        if prompt == "easy":
            self.react_examples = TOOLQA_EASY8
        else:
            self.react_examples = TOOLQA_HARD3

        self.llm = react_llm

        self.table_toolkits = tabtools.table_toolkits(path)
        self.graph_toolkits = graphtools.graph_toolkits(path)

        self.enc = tiktoken.encoding_for_model("text-davinci-003")

    def run(
        self,
        question: str,
        answer: str,
        question_id: str,
    ) -> None:
        question_context = QuestionContext(
            question=question,
            question_id=question_id,
            answer=answer,
        )
        while not self.is_halted(question_context):
            self.step(question_context)

        return question_context.predicted_answer

    def step(
        self,
        question_context: QuestionContext
    ) -> None:

        # Think
        question_context.scratchpad += f'\nThought {question_context.step_n}:'
        question_context.scratchpad += f" {self.prompt_agent(question_context=question_context)}"
        log.info(question_context.scratchpad.split('\n')[-1])

        # Act
        question_context.scratchpad += f'\nAction {question_context.step_n}:'
        action = self.prompt_agent(question_context=question_context)
        question_context.scratchpad += f" {action}"
        log.info(question_context.scratchpad.split('\n')[-1])

        action_context = self.parse_action(action)
        action_output = self.act(action_context=action_context)

        # Observe
        if action_output.success and action_context.action_type == 'Finish':
            question_context.finished = True
            question_context.predicted_answer = action_context.argument
        else:
            question_context.scratchpad += f'\nObservation {question_context.step_n}: '
            question_context.scratchpad += f"{action_output.message}"
            question_context.step_n += 1
            log.info(question_context.scratchpad.split('\n')[-1])

    def prompt_agent(
        self,
        question_context: QuestionContext
    ) -> str:
        prompt = self.agent_prompt.format(
            examples=self.react_examples,
            question=question_context.question,
            scratchpad=question_context.scratchpad
        )
        answer = self.llm(prompt)
        formatted_answer = format_step(answer)
        return formatted_answer

    def is_halted(
        self,
        question_context: QuestionContext
    ) -> bool:
        return question_context.step_n > self.max_steps or question_context.finished
        # return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished
