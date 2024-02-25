"""
This code snippet defines a `FormToolExecutor` class that extends `ToolExecutor` and implements the `_execute` and `_parse_tool_outcome` methods. 

The `_execute` method takes a `ToolInvocationInterface` object and an optional `AgentState` object as input parameters. It checks if the requested tool is in the tool map and then invokes the tool with the provided input. It then calls the `_parse_tool_outcome` method to parse the output and return it.

The `_parse_tool_outcome` method takes the output of the tool invocation and the tool itself as input parameters. It checks the type of the output and creates a `FormToolOutcome` object accordingly. If the output is a string, it creates a `FormToolOutcome` object with an empty state update, the output string, and the return_direct flag from the tool. If the output is already a `FormToolOutcome` object, it simply returns it. If the output is neither a string nor a `FormToolOutcome` object, it raises a `ValueError`.

The code seems well-structured and handles the tool outputs correctly based on their types.
"""
from langgraph.prebuilt.tool_executor import *

from wizard_ai.conversational_engine.form_agent.form_tool import (
    AgentState, FormToolOutcome)


class FormToolExecutor(ToolExecutor):

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
            output = self._parse_tool_outcome(output, tool)
            return output

    def _parse_tool_outcome(
        self,
        output: Union[str, FormToolOutcome],
        tool: BaseTool
    ):
        if isinstance(output, str):
            return FormToolOutcome(
                state_update={},
                output=output,
                return_direct=tool.return_direct,
            )
        elif isinstance(output, FormToolOutcome):
            return output
        else:
            raise ValueError(
                f"Tool returned an invalid output: {output}. Must return a string or a FormToolOutcome.")
