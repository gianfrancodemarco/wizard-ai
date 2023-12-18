import os
import re
import string
from enum import Enum
from typing import List

import tiktoken
from langchain import OpenAI
from langchain.llms.base import BaseLLM
from prompts import LAST_TRIAL_HEADER, REFLECTION_HEADER

### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')


def format_reflections(reflections: List[str],
                       header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])


def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'


def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(
        observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(
            ':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))


class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context
    REFLEXION: Apply reflexion to the next reasoning trace
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial'
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


class QuestionContext:
    def __init__(
        self,
        question: str,
        question_id: str,
        answer: str,
        context: str = '',
        step_n: int = 1,
        finished: bool = False,
        reflexion_strategy: ReflexionStrategy | None = None,
        predicted_answer: str = ''
    ) -> None:
        self.question = question
        self.question_id = question_id
        self.answer = answer
        self.context = context
        self.step_n = step_n
        self.finished = finished
        self.reflexion_strategy = reflexion_strategy
        self.scratchpad = ''
        self.predicted_answer = predicted_answer

    def is_correct(self) -> bool:
        return EM(self.predicted_answer, self.answer)

class ActionContext:
    def __init__(
        self,
        action_type: str,
        argument: str,
    ):
        self.action_type = action_type
        self.argument = argument


class ActionOutput:
    def __init__(
        self,
        success: bool = False,
        message: str = ''
    ):
        self.success = success
        self.message = message
