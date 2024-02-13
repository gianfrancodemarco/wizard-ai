"""
Execute tool invocations using the appropriate tools from the tool map.

This class inherits from ToolExecutor and provides functionality to execute a tool invocation by checking if the requested tool exists in the tool map. If the tool exists, it invokes the tool with the provided input and optional agent state. If the tool does not exist, it returns a message stating the invalid tool name and the list of available tool names.

Attributes:
    Inherits attributes from ToolExecutor.

Methods:
    _execute: Execute a tool invocation by invoking the appropriate tool from the tool map.
    _parse_tool_outcome: Parse the outcome of a tool execution.

Args:
    Inherits arguments from ToolExecutor.

Raises:
    No specific exceptions are raised within this class.
"""

"""
Method _execute:
    Execute a tool invocation using the appropriate tool from the tool map.

    Checks if the requested tool exists in the tool map. If the tool exists, it invokes the tool with the provided input and optional agent state.
    If the tool does not exist, it returns a message stating the invalid tool name and the list of available tool names.

Args:
    tool_invocation (ToolInvocationInterface): An object representing the tool invocation.
    agent_state (AgentState, optional): The state of the agent, defaults to None.

Returns:
    Any: The output of the tool invocation after parsing.

Raises:
    No specific exceptions are raised within this function.
"""

"""
Method _parse_tool_outcome:
    Parse the outcome of a tool execution.

Args:
    output (Union[str, IntentToolOutcome]): The outcome of the tool execution, which can be a string or an IntentToolOutcome.
    tool (BaseTool): The tool that was executed.

Returns:
    IntentToolOutcome: The parsed outcome of the tool execution.

Raises:
    ValueError: If the tool returns an invalid output that is not a string or an IntentToolOutcome.
"""
from langgraph.prebuilt.tool_executor import *

from wizard_ai.conversational_engine.intent_agent.intent_tool import (
    AgentState, IntentToolOutcome)


class IntentToolExecutor(ToolExecutor):
    """Execute tool invocations using the appropriate tools from the tool map.
    
    This class inherits from ToolExecutor and provides functionality to execute a tool invocation by checking if the requested tool exists in the tool map. If the tool exists, it invokes the tool with the provided input and optional agent state. If the tool does not exist, it returns a message stating the invalid tool name and the list of available tool names.
    
    Attributes:
        Inherits attributes from ToolExecutor.
    
    Methods:
        _execute: Execute a tool invocation by invoking the appropriate tool from the tool map.
        _parse_tool_outcome: Parse the outcome of a tool execution.
    
    Args:
        Inherits arguments from ToolExecutor.
    
    Raises:
        No specific exceptions are raised within this class.
    """

    def _execute(
        self,
        tool_invocation: ToolInvocationInterface,
        agent_state: AgentState = None,
    ) -> Any:
        """Execute a tool invocation using the appropriate tool from the tool map.
        
        Checks if the requested tool exists in the tool map. If the tool exists, it invokes the tool with the provided input and optional agent state.
        If the tool does not exist, it returns a message stating the invalid tool name and the list of available tool names.
        
        Args:
            tool_invocation (ToolInvocationInterface): An object representing the tool invocation.
            agent_state (AgentState, optional): The state of the agent, defaults to None.
        
        Returns:
            Any: The output of the tool invocation after parsing.
        
        Raises:
            No specific exceptions are raised within this function.
        """
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
        output: Union[str, IntentToolOutcome],
        tool: BaseTool
    ):
        """Parse the outcome of a tool execution.
        
        Args:
            output (Union[str, IntentToolOutcome]): The outcome of the tool execution, which can be a string or an IntentToolOutcome.
            tool (BaseTool): The tool that was executed.
        
        Returns:
            IntentToolOutcome: The parsed outcome of the tool execution.
        
        Raises:
            ValueError: If the tool returns an invalid output that is not a string or an IntentToolOutcome.
        """
        if isinstance(output, str):
            return IntentToolOutcome(
                state_update={},
                output=output,
                return_direct=tool.return_direct,
            )
        elif isinstance(output, IntentToolOutcome):
            return output
        else:
            raise ValueError(
                f"Tool returned an invalid output: {output}. Must return a string or a FormToolOutcome.")
