from typing import Sequence

from langchain_core.tools import BaseTool

from .context_reset import ContextReset
from .context_update import ContextUpdate
from .form_tool import (AgentState, FormTool,
                        FormToolActivator)


def filter_active_tools(
    tools: Sequence[BaseTool],
    context: AgentState
):
    """
    Form tools are replaced by their activators if they are not active.
    """

    base_tools = list(filter(lambda tool: not isinstance(
        tool, FormToolActivator) and not isinstance(tool, FormTool), tools))

    if context.get("active_form_tool") is None:
        activator_tools = [
            FormToolActivator(
                form_tool_class=tool.__class__,
                form_tool=tool,
                name=f"{tool.name}Activator",
                description=tool.description
            )
            for tool in tools
            if isinstance(tool, FormTool)
        ]
        tools = [
            *base_tools,
            *activator_tools
        ]
    else:
        # If a form_tool is active, remove the Activators and add the form
        # tool and the context update tool
        tools = [
            context.get("active_form_tool"),
            *base_tools,
            ContextUpdate(context=context),
            ContextReset(context=context)
        ]
    return tools
