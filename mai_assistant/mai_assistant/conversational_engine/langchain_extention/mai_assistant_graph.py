import logging
import pprint
import re
from textwrap import dedent
from typing import Any, Sequence, Type

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models.chat_models import *
from langchain_core.load.serializable import Serializable
from langchain_core.messages import FunctionMessage, SystemMessage
from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         MessagesPlaceholder,
                                         SystemMessagePromptTemplate)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from mai_assistant.conversational_engine.langchain_extention.tool_executor_with_state import ToolExecutorWithState, ToolExecutor
from mai_assistant.conversational_engine.langchain_extention.form_tool import (
    AgentState, FormTool, FormToolActivator, filter_active_tools)

logger = logging.getLogger(__name__)

# CONTEXT_UPDATE = "ContextUpdate"

pp = pprint.PrettyPrinter(indent=4)


# class AgentError(Serializable):
#     error: str


# class ChatOpenAIVerbose(ChatOpenAI):
#     def generate_prompt(
#         self,
#         prompts: List[PromptValue],
#         stop: Optional[List[str]] = None,
#         callbacks: Callbacks = None,
#         **kwargs: Any,
#     ) -> LLMResult:

#         logger.info(dedent(f"""

#             Generating prompt with 
                           
#             Prompt:   
#             {pp.pprint(prompts)}

#             Functions:
#             {pp.pprint(kwargs["functions"]) }


#         """))
#         prompt_messages = [p.to_messages() for p in prompts]
#         return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)


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
        self.default_prompt = hub.pull("hwchase17/openai-functions-agent")

    def get_tools(self, state: AgentState):
        return filter_active_tools(self._tools, state)

    def get_tool_by_name(self, name: str, agent_state: AgentState):
        tools = self.get_tools(agent_state)
        return next((tool for tool in tools if tool.name == name), None)

    def get_tool_executor(self, state: AgentState):
        return ToolExecutor(self.get_tools(state))

    def get_model(self, state: AgentState):

        variable_messages = [
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate(prompt=PromptTemplate(
                template="{input}", input_variables=["input"])),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]

        if state["error"]:
            messages = [
                SystemMessagePromptTemplate.from_template(dedent(f"""
                There was an error with your last action.
                Please fix it and try again.
                
                Error:
                {{error}}.

                """)),
                *variable_messages
            ]
            prompt = ChatPromptTemplate(messages=messages)

        elif state["active_form_tool"]:

            form_tool = state["active_form_tool"]
            #form = state["form"]
            information_collected = re.sub("}", "}}", re.sub("{", "{{", str(
                {name: value for name, value in form_tool.form.__dict__.items() if value})))
            information_to_collect = form_tool.get_next_field_to_collect(
                form_tool.form)
            
            ask_info = SystemMessagePromptTemplate.from_template(dedent(f"""
                You need to ask the user to provide the needed information.
                Now you MUST ask the user to provide a value for {information_to_collect}.
                When the user provides a value, use the {form_tool.name} tool to update the form.
            """))

            ask_confirm = SystemMessagePromptTemplate.from_template(dedent(f"""
                You have all the information you need. 
                Show the user all of the information and ask for confirmation.
                If he agrees, call the {form_tool.name} tool one more time with all of the information.
            """))

            messages = [
                SystemMessagePromptTemplate.from_template(dedent(f"""
                You are a personal assistant and you always answer in English.
                You are trying to to help the user fill data for {form_tool.name}.
                So far, you have collected the following information: {information_collected}
                """)),
                ask_info if information_to_collect else ask_confirm,                
                *variable_messages
            ]
            prompt = ChatPromptTemplate(messages=messages)
        else:
            prompt = self.default_prompt
            prompt.messages[0] = SystemMessagePromptTemplate.from_template(dedent(f"""
                You are a personal assistant trying to help the user. You always answer in English.
            """))

        llm = ChatOpenAI(temperature=0, verbose=True)
        model = create_openai_functions_agent(
            llm, self.get_tools(state=state), prompt)
        return model

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

    def should_continue(self, state: AgentState):
        if state['error']:
            return "error"
        if isinstance(state['agent_outcome'], AgentFinish):
            return "end"
        elif isinstance(state['agent_outcome'], AgentAction):
            return "tool"

    def should_continue_after_tool(self, state: AgentState):
        if state['error']:
            return "error"

        action, result = state['intermediate_steps'][-1]
        tool = self.get_tool_by_name(action.tool, state)
        # If tool returns direct, stop here
        # TODO: the tool should be able to dinamically return if return direct or not each time
        if tool and tool.return_direct:
            return "end"
        # Else let the agent use the tool response
        return "continue"

    # Define the function that calls the model
    def call_agent(self, state: AgentState):

        state = state.copy()

        # Cap the number of intermediate steps in a prompt to 5
        if len(state['intermediate_steps']) > 5:
            state['intermediate_steps'] = state['intermediate_steps'][-5:]

        import langchain_core
        try:
            response = self.get_model(state).invoke({
                "input": state["input"],
                "chat_history": state["chat_history"],
                "intermediate_steps": state["intermediate_steps"],
                "error": state["error"],
            })
        except langchain_core.exceptions.OutputParserException as e:
            return {"error": str(e)}

        return {
            "agent_outcome": response,
            "error": None
        }

    def call_tool(self, state: AgentState):

        action = state["agent_outcome"]

        try:
            self.on_tool_start(tool_name=action.tool,
                               tool_input=action.tool_input)

            # We call the tool_executor and get back a response
            response = self.get_tool_executor(state).invoke(action)

            # Allow the tool to update the state
            # If it does so, store the state_update for later and overwrite the response
            # with only the string output
            state_update = {}
            if isinstance(response, dict):
                assert "state_update" in response
                assert "output" in response
                state_update = response["state_update"]
                response = response["output"]

            self.on_tool_end(tool_name=action.tool, tool_output=response)

            function_message = FunctionMessage(
                content=str(response),
                name=action.tool
            )

            return {
                **state_update,
                "intermediate_steps": [(action, function_message)]
            }

        except Exception as e:

            function_message = FunctionMessage(
                content=str(e),
                name=action.tool
            )

            return {
                "intermediate_steps": [(action, function_message)],
                "error": str(e)
            }


if __name__ == "__main__":
    import os

    from mai_assistant.conversational_engine.langchain_extention.helpers import \
        StateGraphDrawer
    os.environ["OPENAI_API_KEY"] = "sk-..."
    graph = MAIAssistantGraph()
    StateGraphDrawer().draw(graph)
