"""
This module contains classes and functions related to building models for conversation agents using OpenAI functions.

Classes:
- ModelFactory: A factory class for creating various types of models for a conversation agent.

Functions:
- information_to_collect_prompt_template: Generate a system message prompt template for collecting specific information using a form tool.
- ask_for_confirmation_prompt_template: Generate a confirmation prompt template for a form tool.

The module also includes functions for building different types of models based on agent state and tools, such as LLM models, default models, intent models, and error correction models. Each function specifies the arguments it takes, return types, and potential exceptions raised.
"""
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
    *PROMPT_FOOTER_MESSAGES
])

DEFAULT_PROMPT_TEMPLATE = ChatPromptTemplate(messages=[
    BASE_SYSTEM_MESSAGE_PROMPT_TEMPLATE,
    *PROMPT_FOOTER_MESSAGES
])


def information_to_collect_prompt_template(
    form_tool: BaseTool,
    information_to_collect: str
):
    """
    Generate a system message prompt template for collecting specific information using a form tool.
    
    Args:
        form_tool (BaseTool): The tool used for the form.
        information_to_collect (str): The specific information to collect from the user.
    
    Returns:
        SystemMessagePromptTemplate: The formatted prompt template for collecting information.
    """
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
    """Generate a confirmation prompt template for a form tool.
    
    Args:
        form_tool (BaseTool): An instance of the BaseTool class representing the form tool.
    
    Returns:
        SystemMessagePromptTemplate: A template for prompting the user to confirm the provided information for the form tool.
    
    The function generates a message template that helps the user fill data for a specific form tool. It presents all the necessary information and asks for confirmation from the user. If the user agrees, the form tool is called again with confirm=True. If the user does not agree or wants to change something, the form tool is called with confirm=False.
    """
    return SystemMessagePromptTemplate.from_template(dedent(
        f"""
        Help the user fill data for {form_tool.name}. You have all the information you need.
        Show the user all of the information and ask for confirmation.
        If he agrees, call the {form_tool.name} tool one more time with confirm=True.
        If he doesn't or want to change something, call it with confirm=False.
        """
    ).strip())


class ModelFactory:
    """
    ```plaintext
    ModelFactory: A factory class for building different types of models for a conversation agent.
    
    Methods:
        build_model: Builds a model based on the provided agent state and tools.
        build_llm: Build a ChatOpenAI instance for a Large Language Model (LLM).
        build_default_model: Builds a default model using the provided agent state and tools.
        build_intent_model: Build an intent model for the conversation agent based on the current state and tools provided.
        build_error_model: Build an error correction model using the given AgentState and optional list of tools.
        __build_model_from_state_and_prompt: Build a model for the agent based on the given state and chat prompt.
    ```
    """

    @staticmethod
    def build_model(
        state: AgentState,
        tools: List[BaseTool] = []
    ):
        """Builds a model based on the provided agent state and tools.
        
        Args:
            state (AgentState): The state of the agent containing information about the environment.
            tools (List[BaseTool], optional): List of tools to be used. Defaults to an empty list.
        
        Returns:
            The built model based on the state and selected tools.
        
        Raises:
            No specific exceptions are raised.
        """
        builder = ModelFactory.build_default_model
        if state.get("error"):
            builder = ModelFactory.build_error_model
        elif state.get("active_intent_tool"):
            builder = ModelFactory.build_intent_model

        return builder(state, tools)

    def build_llm(
        function_call: str = None
    ):
        """Build a ChatOpenAI instance for a Large Language Model (LLM).
        
        Args:
            function_call (str, optional): The function call to be included in the model input. Defaults to None.
        
        Returns:
            ChatOpenAI: A ChatOpenAI instance configured for the Large Language Model.
        """
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
        """
        Builds a default model using the provided agent state and tools.
        
        Args:
            state (AgentState): The state of the agent to build the model from.
            tools (List[BaseTool], optional): List of tools to be included in the model. Defaults to an empty list.
        
        Returns:
            A default model built using the agent state, default prompt template, and optional tools.
        """
        return ModelFactory.__build_model_from_state_and_prompt(
            state=state,
            prompt=DEFAULT_PROMPT_TEMPLATE,
            tools=tools
        )

    def build_intent_model(
        state: AgentState,
        tools: List[BaseTool] = []
    ):
        """Build an intent model for the conversation agent based on the current state and tools provided.
        
        Args:
            state (AgentState): The current state of the conversation agent.
            tools (List[BaseTool], optional): List of tools to be used. Defaults to an empty list.
        
        Returns:
            Model: The intent model constructed based on the provided state, tools, and prompts.
        
        Raises:
            ValueError: If there is an issue with the provided state or tools.
        
        This function builds an intent model using the active intent tool from the state. It collects information from the form tool, generates a prompt based on the next field to collect or asks for confirmation, and constructs a ChatPromptTemplate for the model. The intent model is created using ModelFactory class.
        """

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
        """Build an error correction model using the given AgentState and optional list of tools.
        
        Args:
            state (AgentState): The AgentState object containing necessary information for building the model.
            tools (List[BaseTool], optional): List of BaseTool instances to be used in the model. Defaults to an empty list.
        
        Returns:
            Model: A Model instance representing the error correction model.
        
        Note:
            This function utilizes a ModelFactory to construct the model based on the provided state and tools.
        """
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
        """Build a model for the agent based on the given state and chat prompt.
        
        Args:
            state (AgentState): The state of the agent.
            prompt (ChatPromptTemplate): The chat prompt template to use.
            tools (List[BaseTool], optional): List of BaseTool instances to be used. Defaults to an empty list.
        
        Returns:
            Agent: An agent created using the OpenAI functions agent with the specified model, tools, and prompt.
        """
        return create_openai_functions_agent(
            ModelFactory.build_llm(state.get("function_call")),
            tools,
            prompt=prompt
        )
