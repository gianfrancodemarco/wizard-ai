import logging
from typing import Any, Sequence, Type

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.load.serializable import Serializable
from langchain_core.messages import FunctionMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.prompt import PromptTemplate
from textwrap import dedent
from mai_assistant.conversational_engine.langchain_extention.form_tool import (
    AgentState, FormTool, FormToolActivator)
from mai_assistant.conversational_engine.langchain_extention.intent_helpers import (
    filter_active_tools, make_optional_model)
import re

logger = logging.getLogger(__name__)

CONTEXT_UPDATE = "ContextUpdate"

class AgentError(Serializable):
    error: str


class MAIAssistantGraph(StateGraph):

    def __init__(
        self,
        # state: Type[AgentState] = AgentState(),
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
        # self.state = state
        
    def get_tools(self, state: AgentState):
        return filter_active_tools(self._tools, state)
    
    def get_tool_by_name(self, name: str, agent_state: AgentState):
        tools = self.get_tools(agent_state)
        return next((tool for tool in tools if tool.name == name), None)

    def get_tool_executor(self, state: AgentState):
        return ToolExecutor(self.get_tools(state))

    def get_model(self, state: AgentState):

        if state["active_form_tool"]:

            form_tool = state["active_form_tool"]
            form = state["form"]
            information_collected = re.sub("}", "}}", re.sub("{", "{{", str({name: value for name, value in form.__dict__.items() if value})))
            information_to_collect = form_tool.get_next_field_to_collect(state["form"])

            messages = [
                SystemMessagePromptTemplate.from_template(dedent(f"""
                You are a personal assistant trying to help the user fill data for {form_tool.name}.
                You need to ask the user to provide the needed information.
                So far, you have collected the following information: {information_collected}
                Now you MUST ask the user to provide a value for "{information_to_collect}".
                When the user provides a value, use the ContextUpdate tool to update the form.
                """)),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(template="{input}", input_variables=["input"])
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
            prompt = ChatPromptTemplate(messages=messages)
        else:
            # TODO: change when intentis active
            prompt = self.default_prompt
        
        llm = ChatOpenAI(temperature=0, verbose=True)
        model = create_openai_functions_agent(llm, self.get_tools(state=state), prompt)
        return model

    def __build_graph(self):

        self.add_node("agent", self.call_agent)
        self.add_node("tool", self.call_tool)
        self.add_node("activate_intent", self.activate_intent)

        self.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tool": "tool",
                "intent_activator_tool": "activate_intent",
                "error": "agent",
                "end": END
            }
        )

        self.add_edge("activate_intent", "tool")

        self.add_conditional_edges(
            "tool",
            self.should_continue_after_tool,
            {
                "continue": "agent",
                "end": END
            }
        )

        self.set_entry_point("agent")
        self.app = self.compile()

    def should_continue(self, state: AgentState):

        if isinstance(state['agent_outcome'], AgentFinish):
            return "end"
        
        elif isinstance(state['agent_outcome'], AgentAction):
            action = state['agent_outcome']
            tool = self.get_tool_by_name(action.tool, state)
        
            if isinstance(tool, FormToolActivator):
                logger.info(f"Activating form tool {tool.name}")
                return "intent_activator_tool"
            else:
                logger.info(f"Calling tool {tool.name}")
                return "tool"
        
        elif isinstance(state['agent_outcome'], AgentError):
            return "error"

    def activate_intent(self, state: AgentState):
        action = state["agent_outcome"]
        tool = self.get_tool_executor(state=state).tool_map[action.tool]
        return {
            # TODO: create a custom "SystemAction" class
            "agent_outcome": AgentAction(
                tool=CONTEXT_UPDATE,
                tool_input={"values": action.tool_input},
                log=""
            ),
            "active_form_tool": tool.form_tool,
            "form": make_optional_model(tool.form_tool.args_schema)()
        }

    def should_continue_after_tool(self, state: AgentState):
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
            })
        except langchain_core.exceptions.OutputParserException as e:
            # TODO: Dirty trick to handle the case where the agent response cannot be parsed
            # To be improved and handled with retries
            response = AgentError(error=str(e))
            error = FunctionMessage(
                content=f"Invalid function call, try again. \nError: {str(e)}",
                name="error"
            )
            return {
                "agent_outcome": response,
                "intermediate_steps": [(AgentAction("unknown", "unknown", "unknown"), error)]
            }

        return {"agent_outcome": response}

    def call_tool(self, state: AgentState):

        action = state["agent_outcome"]

        try:
            self.on_tool_start(tool_name=action.tool,
                               tool_input=action.tool_input)
            # We call the tool_executor and get back a response
            response = self.get_tool_executor(state).invoke(action)
            self.on_tool_end(tool_name=action.tool, tool_output=response)

            function_message = FunctionMessage(
                content=str(response),
                name=action.tool
            )

        except Exception as e:
            response = str(e)
            function_message = FunctionMessage(
                content=response,
                name=action.tool
            )

        return {"intermediate_steps": [(action, function_message)]}


if __name__ == "__main__":
    import os

    from mai_assistant.conversational_engine.langchain_extention.helpers import \
        StateGraphDrawer
    os.environ["OPENAI_API_KEY"] = "sk-..."
    graph = MAIAssistantGraph()
    StateGraphDrawer().draw(graph)