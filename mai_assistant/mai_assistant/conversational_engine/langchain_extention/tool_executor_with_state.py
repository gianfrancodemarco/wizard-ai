from langgraph.prebuilt.tool_executor import *

from mai_assistant.conversational_engine.langchain_extention.form_tool import \
    AgentState


class ToolExecutorWithState(ToolExecutor):

    def _execute(
        self,
        tool_invocation: ToolInvocationInterface,
        agent_state: AgentState = None,
    ) -> Any:
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join(
                    [t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = tool.invoke(
                tool_invocation.tool_input,
                agent_state=agent_state)
            return output
