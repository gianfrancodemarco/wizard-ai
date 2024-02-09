import logging
import pprint
import re
from datetime import datetime
from textwrap import dedent
from typing import Any, Sequence, Type, Union

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
    AgentState, FormToolOutcome, filter_active_tools)
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
        """).strip())
    
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

    def parse_output(self, graph_output: dict) -> str:
        """
        Parses the final state of the graph.
        Theoretically, only one between tool_outcome and agent_outcome are set.
        Returns the str to be considered the output of the graph.
        """

        state = graph_output[END]

        output = None
        if state.get("tool_outcome"):
            output = state.get("tool_outcome").output
        elif state.get("agent_outcome"):
            output = state.get("agent_outcome").return_values["output"]
        
        return output

    def __build_graph(self):

        self.add_node("agent", self.call_agent)
        self.add_node("tool", self.call_tool)

        self.add_conditional_edges(
            "agent",
            self.should_continue_after_agent,
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

    def get_llm(self, function_call: str = None):
        return ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
            verbose=True,
            function_call={"name": function_call} if function_call else None
        )

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
        
        information_to_collect = form_tool.get_next_field_to_collect()
        if information_to_collect:
            message = SystemMessagePromptTemplate.from_template(dedent(
            f"""
            Help the user fill data for {form_tool.name}. Ask to provide the needed information.
            Now you MUST ask the user to provide a value for the field "{information_to_collect}".
            You MUST use the {form_tool.name} tool to update the stored data each time the user provides one or more values.
            """
            ).strip())
        else:
            message = SystemMessagePromptTemplate.from_template(dedent(
            f"""
            Help the user fill data for {form_tool.name}. You have all the information you need.
            Show the user all of the information and ask for confirmation.
            If he agrees, call the {form_tool.name} tool one more time with confirm=True.
            If he doesn't or want to change something, call it with confirm=False.
            """
            ).strip())
 
        return self.__get_model_from_state_and_prompt(
            state=state,
            prompt=ChatPromptTemplate(
                messages=[
                    self.base_system_message,
                    message,
                    *self.prompt_footer
                ]
            )
        )

    def __get_error_model(self, state: AgentState):
        return self.__get_model_from_state_and_prompt(
            state=state,
            prompt=ChatPromptTemplate(messages=[
                self.error_correction_prompt,
                *self.prompt_footer,
            ])
        )

    def __get_model_from_state_and_prompt(
        self,
        state: AgentState,
        prompt: ChatPromptTemplate
    ):
        return create_openai_functions_agent(
            self.get_llm(state.get("function_call")),
            self.get_tools(state),
            prompt=prompt
        )

    def should_continue_after_agent(self, state: AgentState):
        if state.get("error"):
            return "error"
        elif isinstance(state.get("agent_outcome"), AgentFinish):
            return "end"
        elif isinstance(state.get("agent_outcome"), AgentAction):
            return "tool"

    def should_continue_after_tool(self, state: AgentState):
        if state.get("error"):
            return "error"
        elif isinstance(state.get("tool_outcome"), FormToolOutcome) and state.get("tool_outcome").return_direct:
            return "end"
        else:
            return "continue"

    # Define the function that calls the model
    def call_agent(self, state: AgentState):
        try:
            # Cap the number of intermediate steps in a prompt to 5
            if len(state.get("intermediate_steps")) > 5:
                state["intermediate_steps"] = state.get(
                    "intermediate_steps")[-5:]

            agent_outcome = self.get_model(state).invoke(state)
            updates = {
                "agent_outcome": agent_outcome,
                "function_call": None, # Reset the function call
                "tool_outcome": None, # Reset the tool outcome
                "error": None  # Reset the error
            }
        # TODO: if other exceptions are raised, we should handle them here
        except OutputParserException as e:
            updates = {"error": str(e)}
        finally:
            return updates
        
    def _parse_tool_outcome(self, tool_output: Union[str, FormToolOutcome]):
        if isinstance(tool_output, str):
            return FormToolOutcome(
                state_update={},
                output=tool_output
            )
        elif isinstance(tool_output, FormToolOutcome):
            return tool_output
        else:
            raise ValueError(
                f"Tool returned an invalid output: {tool_output}. Must return a string or a FormToolOutcome.")


    def call_tool(self, state: AgentState):
        try:
            action = state.get("agent_outcome")
            tool = self.get_tool_by_name(action.tool, state)

            self.on_tool_start(tool=tool, tool_input=action.tool_input)
            tool_outcome = self._parse_tool_outcome(self.get_tool_executor(state).invoke(action))
            self.on_tool_end(tool=tool, tool_output=tool_outcome.output)

            updates = {
                **tool_outcome.state_update,
                "intermediate_steps": [(
                    action,
                    FunctionMessage(
                        content=str(tool_outcome.output),
                        name=action.tool
                    ))],
                "tool_outcome": tool_outcome,
                "agent_outcome": None,
                "error": None
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