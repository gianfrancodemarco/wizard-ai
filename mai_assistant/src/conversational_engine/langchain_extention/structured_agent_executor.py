"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

import asyncio
import datetime
import logging
import re
from textwrap import dedent
from typing import (Any, AsyncIterator, Callable, Dict, Iterator, List,
                    Optional, Sequence, Tuple, Type, Union)

from langchain.agents import (AgentExecutor, BaseMultiActionAgent,
                              BaseSingleActionAgent, StructuredChatAgent)
from langchain.agents.agent import AgentExecutor, ExceptionTool
from langchain.agents.tools import InvalidTool
from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                         AsyncCallbackManagerForToolRun,
                                         CallbackManagerForChainRun,
                                         CallbackManagerForToolRun, Callbacks)
from langchain.chains.llm import LLMChain
from langchain.tools.base import BaseTool
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts.chat import ChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

class ToolDummyPayload(BaseModel):
    """
    We cannot pass directly the BaseModel class as args_schema as pydantic will raise errors,
    so we need to create a dummy class that inherits from BaseModel.
    """
    pass


class FormTool(BaseTool):
    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
        **kwargs
    ) -> str:
        pass

    def activate(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
    ):
        """
        Function called when the tool is activated.
        """

        pass

    async def aactivate(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
    ):
        pass

    # TODO: using context.form as dict is wrong. We need to use a Pydantic model to enable validation etc
    async def aupdate(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
    ):
        pass

    async def ais_form_complete(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
    ) -> bool:
        """
        The default implementation checks if all values except optional ones are set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            if field_info.is_required():
                if not getattr(context.form, field_name):
                    return False
        return True

    def get_tool_start_message(self, input: dict) -> str:
        return "Creating form\n"
    
    def get_information_to_collect(self) -> str:
        return str(list(self.args.keys()))


class FormToolActivator(BaseTool):
    args_schema: Type[BaseModel] = ToolDummyPayload
    form_tool_class: Type[FormTool]
    form_tool: FormTool

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
        **kwargs
    ) -> str:
        return f"Entered in {self.form_tool.name} context"

    def _parse_input(self, tool_input: str | Dict) -> str | Dict[str, Any]:
        """FormToolActivator shouldn't have any input, so we ovveride the default implementation."""
        return {}


class FormStructuredChatExecutorContext(BaseModel):
    active_form_tool: Optional[FormTool] = None
    form: BaseModel = None



class ContextUpdatePayload(BaseModel):
    values: Dict[str, Any]

class ContextReset(BaseTool):
    name = "ContextReset"
    description = """Call this tool when the user doesn't want to fill the form anymore."""
    args_schema: Type[BaseModel] = ToolDummyPayload

    context: Optional[FormStructuredChatExecutorContext] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        self.context.active_form_tool = None
        self.context.form = None
        return "Context reset. Form cleared. Ask the user what he wants to do next."

class ContextUpdate(BaseTool):
    name = "ContextUpdate"
    description = """Useful to store the information given by the user."""
    args_schema: Type[BaseModel] = ContextUpdatePayload

    context: Optional[FormStructuredChatExecutorContext] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        for key, value in kwargs['values'].items():
            setattr(self.context.form, key, value)
        return "Context updated"

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

    # Adding the dynamically created class to the global scope so that it can be pickled
    # https://stackoverflow.com/a/39529149/8458431
    globals()[new_class_name] = OptionalModel
    # Validators are not working !!!

    return OptionalModel


class FormStructuredChatExecutor(AgentExecutor):

    context: FormStructuredChatExecutorContext
    original_params: Dict[str, Any]
    memory_prompts: List[ChatMessagePromptTemplate]

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
            # If a form_tool is active, remove the Activators and add the form tool and the context update tool
            tools = [
                context.active_form_tool,
                *base_tools,
                ContextUpdate(context=context),
                ContextReset(context=context)
            ]
        return tools

    @classmethod
    def from_tools_and_builders(cls,
        llm_chain_builder: Callable[[Sequence[BaseTool]], LLMChain],
        agent_builder: Callable[[LLMChain, Sequence[BaseTool]], Union[BaseSingleActionAgent, BaseMultiActionAgent]],
        tools: Sequence[BaseTool],
        context: FormStructuredChatExecutorContext,
        memory_prompts: List[ChatMessagePromptTemplate],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from a list of tools. The tools will be used to create the LLMChain and the agent."""

        original_params = {
            "llm_chain_builder": llm_chain_builder,
            "agent_builder": agent_builder,
            "tools": tools
        }
        tools = cls.filter_active_tools(tools, context)
        llm_chain = llm_chain_builder(tools)
        agent = agent_builder(llm_chain, tools)

        instance = cls(
            agent=agent,
            tools=tools,
            context=context,
            callbacks=callbacks,
            original_params=original_params,
            memory_prompts=memory_prompts,
            **kwargs,
        )
        if context.active_form_tool is not None:
            instance._update_llm_chain()
        return instance

    def _restore_llm_chain(self) -> LLMChain:
        """Restore the llm_chain to the original state."""
        tools = self.filter_active_tools(self.original_params["tools"], self.context)
        self.agent.llm_chain = self.original_params["llm_chain_builder"](
            tools
        )

    def _update_llm_chain(self) -> LLMChain:
        """After the a form tool is activated, we need to update the llm_chain to include the new prompt."""

        tool = self.context.active_form_tool
        # information_to_collect = re.sub(
        #     "}", "}}", re.sub("{", "{{", str(tool.args)))
        information_to_collect = tool.get_information_to_collect()
        information_collected = re.sub("}", "}}", re.sub("{", "{{", str(
            {name: value for name, value in self.context.form.__dict__.items() if value})))

        prefix = dedent(f"""
            Today is: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            You are a personal assistant. The user is trying to fill data for {tool.name} and you need to help him.

            Kindly ask the user to provide the next missing information using the Final Answer tool.

            You have access to the following tools:
        """)

        FORMAT_INSTRUCTIONS = dedent("""
            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{{{
            "action": $TOOL_NAME,
            "action_input": $INPUT
            }}}}
            ```

            Follow this format:

            Question: human input to the assistant

            (repeat the following Thought/Action/Observation N times)
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            Thought: I know what to respond
            Action:
            ```
            {{{{
            "action": "Final Answer",
            "action_input": "Final response to human"
            }}}}                                     
            ```
        """)

        suffix = dedent(f"""
            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
            Thought:

            The information you need to collect is the following:
            {information_to_collect}

            The information you have collected so far is:
            {information_collected}

            When you have all the NEEDED information, call {tool.name} with the input data.
        """)

        prompt = StructuredChatAgent.create_prompt(
            prefix=prefix,
            suffix=suffix,
            memory_prompts=self.memory_prompts,
            format_instructions=FORMAT_INSTRUCTIONS,
            tools=self.filter_active_tools(
                self.original_params["tools"], self.context),
            input_variables=["input"]
        )

        self.agent.llm_chain = LLMChain(
            llm=self.agent.llm_chain.llm,
            prompt=prompt,
            verbose=True
        )

    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(
                intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError(
                    "Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
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
            yield AgentStep(action=agent_action, observation=observation)

    async def _aiter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(
                intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError(
                    "Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        for agent_action in actions:
            yield agent_action

        async def _aperform_agent_action(
            agent_action: AgentAction,
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
                    if self.context.active_form_tool != tool.form_tool:
                        self.context.active_form_tool = tool.form_tool
                        await tool.form_tool.aactivate(
                            run_manager=run_manager.get_child() if run_manager else None,
                            context=self.context
                        )
                        # Create a copy from the args_schema with all attributes optional, so that we can instantiate it in the context,
                        # provide partial updates, and still have all original validators
                        self.context.form = make_optional_model(
                            tool.form_tool.args_schema)()
                    await tool.form_tool.aupdate(
                        agent_action.tool_input,
                        run_manager=run_manager.get_child() if run_manager else None,
                        context=self.context
                    )
                    is_form_tool_complete = await tool.form_tool.ais_form_complete(
                        run_manager=run_manager.get_child() if run_manager else None,
                        context=self.context
                    )
                    if not is_form_tool_complete:
                        self._update_llm_chain()

                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    context=self.context,
                    **tool_run_kwargs,
                )

                # We called the tool and was completed, we can reset the context
                if isinstance(tool, FormTool) or isinstance(tool, ContextReset):
                    self.context = FormStructuredChatExecutorContext()
                    self._restore_llm_chain()

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

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        # TODO This could yield each result as it becomes available
        for chunk in result:
            yield chunk
