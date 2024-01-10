import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain.agents import (BaseMultiActionAgent, BaseSingleActionAgent,
                              StructuredChatAgent)
from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import InvalidTool
from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                         CallbackManagerForChainRun)
from langchain.chains import LLMChain
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from .context_reset import ContextReset
from .context_update import ContextUpdate
from .form_tool import (FormStructuredChatExecutorContext, FormTool,
                        FormToolActivator)
from .prompts import FORMAT_INSTRUCTIONS, SUFFIX, MEMORY_PROMPTS, get_prefix

logger = logging.getLogger(__name__)


def make_optional_model(original_model: BaseModel) -> BaseModel:
    """
    Takes a Pydantic model and returns a new model with all attributes optional.
    """
    optional_attributes = {attr_name: (
        attr_type, None) for attr_name, attr_type in original_model.__annotations__.items()}

    # Define a custom Pydantic model with optional attributes
    new_class_name = original_model.__name__ + 'Optional'
    OptionalModel = create_model(
        new_class_name,
        **optional_attributes,
        __base__=original_model
    )
    OptionalModel.model_config["validate_assignment"] = True

    return OptionalModel


class FormStructuredChatExecutor(AgentExecutor):

    max_iterations: int = 5
    handle_parsing_errors = True

    form_agent: Optional[Union[BaseSingleActionAgent, BaseMultiActionAgent]]
    context: FormStructuredChatExecutorContext

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.context.active_form_tool:
            self._activate_form_agent()

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: LLM,
        tools: Sequence[BaseTool],
        context: FormStructuredChatExecutorContext,
        **kwargs
    ):
        if not context:
            raise ValueError("Context cannot be None")

        tools = cls.filter_active_tools(tools, context)

        llm_chain = LLMChain(
            llm=llm,
            prompt=StructuredChatAgent.create_prompt(
                tools=tools,
                memory_prompts=MEMORY_PROMPTS
            ),
            verbose=True
        )

        # TODO: deprecated, use create_structured_chat_agent
        agent = StructuredChatAgent(
            llm_chain=llm_chain,
            tools=tools,
            verbose=True
        )

        return cls(
            agent=agent,
            tools=tools,
            context=context,
            **kwargs
        )

    def _activate_form_agent(
        self
    ):
        self.form_agent = StructuredChatAgent.from_llm_and_tools(
            self.agent.llm_chain.llm,
            prefix=get_prefix(),
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            memory_prompts=MEMORY_PROMPTS,
            tools=FormStructuredChatExecutor.filter_active_tools(
                self.tools, self.context),
        )
        self.agent = self.form_agent

    def _update_inputs(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if isinstance(inputs, str):
            inputs = {"input": inputs}

        # If there is an active form tool, we need to update the inputs with
        # those expected by the new prompt
        if self.context.active_form_tool:
            tool = self.context.active_form_tool
            # information_to_collect = re.sub(
            #     "}", "}}", re.sub("{", "{{", str(tool.args)))
            information_to_collect = tool.get_next_field_to_collect(
                self.context)
            information_collected = re.sub("}", "}}", re.sub("{", "{{", str(
                {name: value for name, value in self.context.form.__dict__.items() if value})))

            inputs.update({
                "tool_name": tool.name,
                "information_to_collect": information_to_collect,
                "information_collected": information_collected
            })
        return inputs

    def prep_inputs(
            self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        inputs = self._update_inputs(inputs)
        return super().prep_inputs(inputs)

    @classmethod
    def filter_active_tools(
        cls,
        tools: Sequence[BaseTool],
        context: FormStructuredChatExecutorContext
    ):

        base_tools = list(filter(lambda tool: not isinstance(
            tool, FormToolActivator) and not isinstance(tool, FormTool), tools))

        if context.active_form_tool is None:
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
                context.active_form_tool,
                *base_tools,
                ContextUpdate(context=context),
                ContextReset(context=context)
            ]
        return tools
    # SYNC

    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        inputs = self._update_inputs(inputs)
        return super()._iter_next_step(
            name_to_tool_map=name_to_tool_map,
            color_mapping=color_mapping,
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            run_manager=run_manager
        )

    def _perform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""

            # We then call the tool on the tool input to get an observation
            is_form_tool_activator = isinstance(tool, FormToolActivator)
            if is_form_tool_activator:
                agent_action.tool_input = ""
                if self.context.active_form_tool != tool.form_tool:
                    self.context.active_form_tool = tool.form_tool
                    # Create a copy from the args_schema with all attributes optional, so that we can instantiate it in the context,
                    # provide partial updates, and still have all original
                    # validators
                    self.context.form = make_optional_model(
                        tool.form_tool.args_schema)()
                is_form_tool_complete = tool.form_tool.is_form_complete(
                    context=self.context
                )
                if not is_form_tool_complete:
                    self._activate_form_agent()
            # We called the tool and was completed, we can reset the context
            if isinstance(tool, FormTool) or isinstance(tool, ContextReset):
                self.context = FormStructuredChatExecutorContext()
                # self._restore_llm_chain()

            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    # ASYNC

    def _aiter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        inputs = self._update_inputs(inputs)
        return super()._aiter_next_step(
            name_to_tool_map=name_to_tool_map,
            color_mapping=color_mapping,
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            run_manager=run_manager
        )

    async def _aperform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""

            # We then call the tool on the tool input to get an observation
            is_form_tool_activator = isinstance(tool, FormToolActivator)
            if is_form_tool_activator:
                agent_action.tool_input = ""
                if self.context.active_form_tool != tool.form_tool:
                    self.context.active_form_tool = tool.form_tool
                    # Create a copy from the args_schema with all attributes optional, so that we can instantiate it in the context,
                    # provide partial updates, and still have all original
                    # validators
                    self.context.form = make_optional_model(
                        tool.form_tool.args_schema)()
                is_form_tool_complete = tool.form_tool.is_form_complete(
                    context=self.context
                )
                if not is_form_tool_complete:
                    self._activate_form_agent()
            # We called the tool and was completed, we can reset the context
            if isinstance(tool, FormTool) or isinstance(tool, ContextReset):
                self.context = FormStructuredChatExecutorContext()
                # self._restore_llm_chain()

            observation = await tool.arun(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                context=self.context,
                **tool_run_kwargs,
            )

        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)
