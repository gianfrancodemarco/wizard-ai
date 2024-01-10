from .agent import Agent

class LLamaAgent(Agent):
    pass

# # Chain components
# # 1. Memory


# from langchain.agents import AgentType, initialize_agent

# from mai_assistant.src.tools.calculator import Calculator
# from mai_assistant.src.tools.random_number_generator import \
#     RandomNumberGenerator


# def get_memory_chain(memory: BaseChatMemory):
#     return RunnablePassthrough.assign(
#         history=RunnableLambda(
#             memory.load_memory_variables) | itemgetter("history")
#     )


# # 2. Prompt
# prompt = PromptTemplate.from_template("""
# You are a professional personal assistant who helps people with their daily tasks.

# Previous conversation:
# {history}

# [INST]{question}[/INST]
# AI:
# """)


# # 3. LLM model
# LLM_MODEL = os.environ.get('LLM_MODEL')
# # check if llm model is valid
# if LLM_MODEL not in LLM_MODELS.values():
#     raise ValueError(f"LLM_MODEL must be one of {LLM_MODELS.values()}")
# llm = LLMClientFactory.create(
#     LLM_MODEL,
#     url=os.environ.get('LLM_URL')
# )


# tools = [
#     Calculator(),
#     RandomNumberGenerator()
# ]
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True)


# import datetime
# import os
# from textwrap import dedent

# from langchain.agents.structured_chat.base import *
# from langchain.memory.chat_memory import BaseMemory
# from langchain_core.prompts.chat import ChatMessagePromptTemplate

# from mai_assistant.src.clients.llm import LLM_MODELS, LLMClientFactory
# from mai_assistant.src.conversational_engine.tools import *

# from .agent import Agent


# class GPTAgent(Agent):

#     def get_llm(self):
#         return LLMClientFactory.create(
#             LLM_MODELS.GPT3_5_TURBO.value,
#             url=os.environ.get('LLM_URL')
#         )

#     def get_prefix(self):
#         """
#         We use a function here to avoid the prefix being cached in the module, so that the current time is always up to date.
#         """

#         return dedent(f"""
#             Respond to the human as helpfully and accurately as possible.
#             If the user request is not clear, ask for clarification (using the final answer tool).
#             Current time is: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#             \n
#             You have access to the following tools:"""
#                     )


#     def get_memory_prompts(self):
#         memory_prompts = [
#             ChatMessagePromptTemplate.from_template(
#                 role="Previous conversation",
#                 template="""
#             \n\n
#         {history}
#             \n\n
#             """
#             )
#         ]


#     def create_prompt(
#         self,
#         tools: Sequence[BaseTool],
#         prefix: str = get_prefix(),
#         suffix: str = SUFFIX,
#         human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
#         format_instructions: str = FORMAT_INSTRUCTIONS,
#         input_variables: Optional[List[str]] = None,
#         memory_prompts: Optional[List[BasePromptTemplate]] = None,
#     ) -> BasePromptTemplate:
#         """
#         Custom prompt with a slightly different positioning of memory and prompt suffix w.r.t StructuredChatAgent.create_prompt
#         """

#         tool_strings = []
#         for tool in tools:
#             args_schema = re.sub("}", "}}", re.sub("{", "{{", str(tool.args)))
#             tool_strings.append(
#                 f"{tool.name}: {tool.description}, args: {args_schema}")
#         formatted_tools = "\n".join(tool_strings)
#         tool_names = ", ".join([tool.name for tool in tools])
#         format_instructions = format_instructions.format(tool_names=tool_names)
#         template = "\n\n".join(
#             [prefix, formatted_tools, format_instructions])
#         if input_variables is None:
#             input_variables = ["input", "agent_scratchpad"]
#         _memory_prompts = memory_prompts or []
#         messages = [
#             SystemMessagePromptTemplate.from_template(template),
#             *_memory_prompts,
#             SystemMessagePromptTemplate.from_template(suffix),
#             ChatMessagePromptTemplate.from_template(
#                 role="Input",
#                 template=human_message_template),
#         ]
#         return ChatPromptTemplate(
#             input_variables=input_variables,
#             messages=messages)