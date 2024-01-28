import json
from typing import Any, Sequence, Type

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import FunctionMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from mai_assistant.conversational_engine.langchain_extention.form_tool import \
    AgentState
from mai_assistant.conversational_engine.langchain_extention.intent_helpers import \
    filter_active_tools


class MAIAssistantGraph(StateGraph):

    def __init__(
        self,
        state: Type[AgentState] = AgentState(),
        tools: Sequence[Type[Any]] = [],
        on_tool_start: callable = None,
        on_tool_end: callable = None,
    ) -> None:
        super().__init__(AgentState)

        self.tools = filter_active_tools(tools, state)
        self.tool_executor = ToolExecutor(self.tools)
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.__build_graph()
        
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.llm = ChatOpenAI(temperature=0, verbose=True)
        self.model = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        
    def __build_graph(self):

        # Define the two nodes we will cycle between
        self.add_node("agent", self.call_agent)
        self.add_node("tool", self.call_tool)

        # We now add a conditional edge
        self.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tool",
                # Otherwise we finish.
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

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.set_entry_point("agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.app = self.compile()

    # Define the function that determines whether to continue or not
    def should_continue(self, state: AgentState):
        
        if isinstance(state['agent_outcome'], AgentFinish):
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

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

        response = self.model.invoke({
            "input": state["input"],
            "chat_history": state["chat_history"],
            "intermediate_steps": state["intermediate_steps"],
        })

        # We return a list, because this will get added to the existing list
        return {"agent_outcome": response}

    # Define the function to execute tools
    def call_tool(self, state: AgentState):

        action = state["agent_outcome"]

        try:
            # We construct an ToolInvocation from the function_call
                # action = ToolInvocation(
                #     tool=action.tool,
                #     tool_input=action.tool_input
                # )
        # except Exception as e:
        #     error = str(e)
        #     function_message = FunctionMessage(
        #         content=f"Invalid function call, try again. \nError: {error}",
        #         name=action.additional_kwargs["function_call"]["name"]
        #     )
        #     return {"intermediate_steps": [(action, function_message)]}

        # try:
            self.on_tool_start(tool_name=action.tool,
                               tool_input=action.tool_input)
            # We call the tool_executor and get back a response
            response = self.tool_executor.invoke(action)
            self.on_tool_end(tool_name=action.tool, tool_output=response)

            function_message = FunctionMessage(
                content=str(response),
                name=action.tool,
                # additional_kwargs={
                #     **action.additional_kwargs,
                #     "return_direct": return_direct
                # }
            )

        except Exception as e:
            response = str(e)
            function_message = FunctionMessage(
                content=response,
                name=action.tool
            )

        # We return a list, because this will get added to the existing list
        return {"intermediate_steps": [(action, function_message)]}


if __name__ == "__main__":
    import os

    from mai_assistant.conversational_engine.langchain_extention.helpers import \
        StateGraphDrawer
    os.environ["OPENAI_API_KEY"] = "sk-..."
    graph = MAIAssistantGraph()
    StateGraphDrawer().draw(graph)