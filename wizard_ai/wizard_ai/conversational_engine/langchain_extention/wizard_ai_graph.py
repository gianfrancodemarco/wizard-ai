import logging
import pprint
import re
from datetime import datetime
from textwrap import dedent
from typing import Any, Sequence, Type

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import *
from langchain_core.messages import FunctionMessage
from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         MessagesPlaceholder,
                                         SystemMessagePromptTemplate)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from wizard_ai.conversational_engine.langchain_extention.form_tool import (
    AgentState, FormToolOutput, filter_active_tools)
from wizard_ai.conversational_engine.langchain_extention.tool_executor_with_state import \
    ToolExecutor

logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)


class MAIAssistantGraph(StateGraph):

    def __init__(
        self,
        tools: Sequence[Type[Any]] = [],
        on_tool_start: callable = None,
        on_tool_end: callable = None,
    ) -> None:
        super().__init__(AgentState)

        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.__build_graph()
        self._tools = tools

    @property
    def default_prompt(self):
        return ChatPromptTemplate(messages=[
            self.base_system_message,
            *self.prompt_footer
        ])

    @property
    def base_system_message(self):
        return SystemMessagePromptTemplate.from_template(dedent(f"""
            You are a personal assistant trying to help the user. You always answer in English. The current datetime is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
            Don't use any of your knowledge or information about the state of the world. If you need something, ask the user for it or use a tool to find or compute it.
        """))

    @property
    def prompt_footer(self):
        return [MessagesPlaceholder(
            variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate(prompt=PromptTemplate(
                template="{input}", input_variables=["input"])),
            MessagesPlaceholder(variable_name="agent_scratchpad")]

    @property
    def error_correction_prompt(self):
        return ChatPromptTemplate(messages=[
            self.base_system_message,
            SystemMessagePromptTemplate.from_template(dedent(f"""
                There was an error with your last action.
                Please fix it and try again.

                Error:
                {{error}}.

                """))
        ])

    def __build_graph(self):

        self.add_node("agent", self.call_agent)
        self.add_node("tool", self.call_tool)

        self.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tool": "tool",
                "error": "agent",
                "end": END
            }
        )

        self.add_conditional_edges(
            "tool",
            self.should_continue_after_tool,
            {
                "error": "agent",
                "continue": "agent",
                "end": END
            }
        )

        self.set_entry_point("agent")
        self.app = self.compile()

    def get_tools(self, state: AgentState):
        return filter_active_tools(self._tools[:], state)

    def get_tool_by_name(self, name: str, agent_state: AgentState):
        tools = self.get_tools(agent_state)
        return next((tool for tool in tools if tool.name == name), None)

    def get_tool_executor(self, state: AgentState):
        return ToolExecutor(self.get_tools(state))

    def get_llm(self):
        return ChatOpenAI(temperature=0, verbose=True)

    def get_model(self, state: AgentState):

        model = self.__get_default_model(state=state)

        if state.get("error"):
            model = self.__get_error_model(state=state)
        elif state.get("active_form_tool"):
            model = self.__get_intent_model(state=state)

        return model

    def __get_default_model(self, state: AgentState):
        return self.__get_model_from_state_and_prompt(
            state=state,
            prompt=self.default_prompt
        )

    def __get_intent_model(self, state: AgentState):

        form_tool = state.get("active_form_tool")
        information_collected = re.sub("}", "}}", re.sub("{", "{{", str(
            {name: value for name, value in form_tool.form.__dict__.items() if value})))
        information_to_collect = form_tool.get_next_field_to_collect(
            form_tool.form)

        ask_info = SystemMessagePromptTemplate.from_template(dedent(f"""
            You need to ask the user to provide the needed information.
            Now you MUST ask the user to provide a value for the field "{information_to_collect}".
            Use the {form_tool.name} tool to update the form each time the user provides one or more values.
        """))

        ask_confirm = SystemMessagePromptTemplate.from_template(dedent(f"""
            You have all the information you need.
            Show the user all of the information and ask for confirmation.
            If he agrees, call the {form_tool.name} tool one more time with all of the information.
        """))

        return self.__get_model_from_state_and_prompt(
            state=state,
            prompt=ChatPromptTemplate(
                messages=[
                    self.base_system_message,
                    ask_info if information_to_collect else ask_confirm,
                    *self.prompt_footer
                ]
            )
        )

    def __get_error_model(self, state: AgentState):
        return self.__get_model_from_state_and_prompt(
            state=state,
            prompt=ChatPromptTemplate(messages=[
                self.base_system_message,
                *self.prompt_footer,
                self.error_correction_prompt
            ])
        )

    def __get_model_from_state_and_prompt(
        self,
        state: AgentState,
        prompt: ChatPromptTemplate
    ):
        return create_openai_functions_agent(
            self.get_llm(),
            self.get_tools(state),
            prompt=prompt
        )

    def should_continue(self, state: AgentState):
        if state.get("error"):
            return "error"
        if isinstance(state.get("agent_outcome"), AgentFinish):
            return "end"
        elif isinstance(state.get("agent_outcome"), AgentAction):
            return "tool"

    def should_continue_after_tool(self, state: AgentState):
        if state.get("error"):
            return "error"

        action, result = state.get("intermediate_steps")[-1]
        tool = self.get_tool_by_name(action.tool, state)
        # If tool returns direct, stop here
        # TODO: the tool should be able to dinamically return if return direct
        # or not each time
        if tool and tool.return_direct:
            return "end"
        # Else let the agent use the tool response
        return "continue"

    # Define the function that calls the model
    def call_agent(self, state: AgentState):
        try:
            # Cap the number of intermediate steps in a prompt to 5
            if len(state.get("intermediate_steps")) > 5:
                state["intermediate_steps"] = state.get(
                    "intermediate_steps")[-5:]

            response = self.get_model(state).invoke(state)
            updates = {
                "agent_outcome": response,
                "error": None  # Reset the error
            }
        # TODO: if other exceptions are raised, we should handle them here
        except OutputParserException as e:
            updates = {"error": str(e)}
        finally:
            return updates

    def call_tool(self, state: AgentState):
        try:
            action = state.get("agent_outcome")
            self.on_tool_start(tool_name=action.tool,
                               tool_input=action.tool_input)

            # We call the tool_executor and get back a response
            response = self.get_tool_executor(state).invoke(action)

            # Allow the tool to update the state
            # If it does so, store the state_update for later and overwrite the response
            # with only the string output
            state_update = {}
            if isinstance(response, FormToolOutput):
                state_update = response.state_update
                response = response.output

            self.on_tool_end(tool_name=action.tool, tool_output=response)

            function_message = FunctionMessage(
                content=str(response),
                name=action.tool
            )

            updates = {
                **state_update,
                "intermediate_steps": [(action, function_message)]
            }

        except Exception as e:
            updates = {
                "intermediate_steps": [(action, FunctionMessage(
                    content=str(e),
                    name=action.tool
                ))],
                "error": str(e)
            }
        finally:
            return updates


if __name__ == "__main__":
    import os

    from wizard_ai.conversational_engine.langchain_extention.helpers import \
        StateGraphDrawer
    os.environ["OPENAI_API_KEY"] = "sk-..."
    graph = MAIAssistantGraph()
    StateGraphDrawer().draw(graph)