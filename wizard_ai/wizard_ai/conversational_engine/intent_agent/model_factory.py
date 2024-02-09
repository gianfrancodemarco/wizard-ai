import logging
import os
import pprint
import re
from datetime import datetime
from textwrap import dedent

from langchain.agents import create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import *
from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         MessagesPlaceholder,
                                         SystemMessagePromptTemplate)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from wizard_ai.conversational_engine.intent_agent.intent_tool import AgentState

logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")

BASE_SYSTEM_MESSAGE_PROMPT = dedent(f"""
    You are a personal assistant trying to help the user. You always answer in English. The current datetime is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
    Don't use any of your knowledge or information about the state of the world. If you need something, ask the user for it or use a tool to find or compute it.
""").strip()
BASE_SYSTEM_MESSAGE_PROMPT_TEMPLATE = SystemMessagePromptTemplate.from_template(
    BASE_SYSTEM_MESSAGE_PROMPT)

PROMPT_FOOTER_MESSAGES = [
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate(prompt=PromptTemplate(
        template="{input}", input_variables=["input"])),
    MessagesPlaceholder(variable_name="agent_scratchpad")
]

ERROR_CORRECTION_PROMPT = dedent(f"""
    There was an error with your last action.
    Please fix it and try again.
                                
    Error:
    {{error}}.
""").strip()

ERROR_CORRECTION_SYSTEM_MESSAGE = SystemMessagePromptTemplate.from_template(
    ERROR_CORRECTION_PROMPT)

ERROR_CORRECTION_PROMPT_TEMPLATE = ChatPromptTemplate(messages=[
    BASE_SYSTEM_MESSAGE_PROMPT_TEMPLATE,
    ERROR_CORRECTION_SYSTEM_MESSAGE,
])

DEFAULT_PROMPT_TEMPLATE = ChatPromptTemplate(messages=[
    BASE_SYSTEM_MESSAGE_PROMPT_TEMPLATE,
    *PROMPT_FOOTER_MESSAGES
])


def information_to_collect_prompt_template(
    form_tool: BaseTool,
    information_to_collect: str
):
    return SystemMessagePromptTemplate.from_template(dedent(
        f"""
        Help the user fill data for {form_tool.name}. Ask to provide the needed information.
        Now you MUST ask the user to provide a value for the field "{information_to_collect}".
        You MUST use the {form_tool.name} tool to update the stored data each time the user provides one or more values.
        """
    ).strip())


def ask_for_confirmation_prompt_template(
    form_tool: BaseTool
):
    return SystemMessagePromptTemplate.from_template(dedent(
        f"""
        Help the user fill data for {form_tool.name}. You have all the information you need.
        Show the user all of the information and ask for confirmation.
        If he agrees, call the {form_tool.name} tool one more time with confirm=True.
        If he doesn't or want to change something, call it with confirm=False.
        """
    ).strip())


class ModelFactory:

    @staticmethod
    def build_model(
        state: AgentState,
        tools: List[BaseTool] = []
    ):
        builder = ModelFactory.build_default_model
        if state.get("error"):
            builder = ModelFactory.build_error_model
        elif state.get("active_intent_tool"):
            builder = ModelFactory.build_intent_model

        return builder(state, tools)

    def build_llm(
        function_call: str = None
    ):
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            verbose=True,
            function_call={"name": function_call} if function_call else None
        )

    def build_default_model(
        state: AgentState,
        tools: List[BaseTool] = []
    ):
        return ModelFactory.__build_model_from_state_and_prompt(
            state=state,
            prompt=DEFAULT_PROMPT_TEMPLATE,
            tools=tools
        )

    def build_intent_model(
        state: AgentState,
        tools: List[BaseTool] = []
    ):

        form_tool = state.get("active_intent_tool")
        information_collected = re.sub("}", "}}", re.sub("{", "{{", str(
            {name: value for name, value in form_tool.form.__dict__.items() if value})))

        information_to_collect = form_tool.get_next_field_to_collect()
        if information_to_collect:
            message = information_to_collect_prompt_template(
                form_tool, information_to_collect)
        else:
            message = ask_for_confirmation_prompt_template(form_tool)

        return ModelFactory.__build_model_from_state_and_prompt(
            state=state,
            prompt=ChatPromptTemplate(
                messages=[
                    BASE_SYSTEM_MESSAGE_PROMPT_TEMPLATE,
                    message,
                    *PROMPT_FOOTER_MESSAGES
                ]
            ),
            tools=tools
        )

    def build_error_model(
        state: AgentState,
        tools: List[BaseTool] = []
    ):
        return ModelFactory.__build_model_from_state_and_prompt(
            state=state,
            prompt=ERROR_CORRECTION_PROMPT_TEMPLATE,
            tools=tools
        )

    def __build_model_from_state_and_prompt(
        state: AgentState,
        prompt: ChatPromptTemplate,
        tools: List[BaseTool] = []
    ):
        return create_openai_functions_agent(
            ModelFactory.build_llm(state.get("function_call")),
            tools,
            prompt=prompt
        )
