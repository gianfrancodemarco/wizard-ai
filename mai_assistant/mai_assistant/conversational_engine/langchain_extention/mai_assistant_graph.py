from typing import Any, Sequence, Type

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.messages import FunctionMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.load.serializable import Serializable

from mai_assistant.conversational_engine.langchain_extention.form_tool import \
    AgentState
from mai_assistant.conversational_engine.langchain_extention.intent_helpers import \
    filter_active_tools

class AgentError(Serializable):
    error: str

class MAIAssistantGraph(StateGraph):

    def __init__(
        self,
        state: Type[AgentState] = AgentState(),
        tools: Sequence[Type[Any]] = [],
        on_tool_start: callable = None,
        on_tool_end: callable = None,
    ) -> None:
        super().__init__(AgentState)

        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.__build_graph()
        
        self.tools = filter_active_tools(tools, state)
        self.tool_executor = ToolExecutor(self.tools)
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.llm = ChatOpenAI(temperature=0, verbose=True)
        self.model = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        
    def __build_graph(self):

        self.add_node("agent", self.call_agent)
        self.add_node("tool", self.call_tool)
        self.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tool",
                "error": "agent",
                "end": END
            }
        )

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
            return "continue"
        elif isinstance(state['agent_outcome'], AgentError):
            return "error"

    def should_continue_after_tool(self, state: AgentState):
        action, result = state['intermediate_steps'][-1]
        tool = next((tool for tool in self.tools if tool.name == action.tool), None)
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
            response = self.model.invoke({
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
            response = self.tool_executor.invoke(action)
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