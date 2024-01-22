import getpass
import json
import operator
import os
from typing import Annotated, Sequence, Type, TypedDict

from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.callbacks.stdout import StdOutCallbackHandler

from mai_assistant.conversational_engine.tools.google.search import \
    GoogleSearch
from typing import Any

os.environ["OPENAI_API_KEY"] = "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"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class Graph(StateGraph):

    def __init__(
        self,
        tools: Sequence[Type[Any]] = [],
    ) -> None:
        super().__init__(AgentState)

        self.tools = tools
        self.tool_executor = ToolExecutor(self.tools)
        self.functions = [format_tool_to_openai_function(t) for t in self.tools]
        self.model = ChatOpenAI(temperature=0, verbose=True).bind_functions(self.functions)

        # Define the two nodes we will cycle between
        self.add_node("agent", self.call_model)
        self.add_node("action", self.call_tool)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.set_entry_point("agent")

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
                "continue": "action",
                # Otherwise we finish.
                "end": END
            }
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        self.add_edge('action', 'agent')

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.app = self.compile()

    # Define the function that determines whether to continue or not
    def should_continue(self, state):
        messages = state['messages']
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(self, state):
        messages = state['messages']
        response = self.model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function to execute tools
    def call_tool(self, state):
        messages = state['messages']
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}


# config = {
#     "callbacks": [
#         StdOutCallbackHandler()
#     ]
# }

# while True:
#     inputs = {"messages": [HumanMessage(content=input())]}
#     print(Graph().app.invoke(inputs, config=config)['messages'][-1])