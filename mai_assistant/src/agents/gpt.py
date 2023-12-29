import os

from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseMemory

from mai_assistant.src.llm_client import LLM_MODELS, LLMClientFactory
from mai_assistant.src.tools import Calculator, RandomNumberGenerator, Search

from langchain.agents.structured_chat.base import *
from langchain_core.prompts.chat import ChatMessagePromptTemplate

PREFIX = """Respond to the human as helpfully and accurately as possible.
If the user request is not clear, ask for clarification (using the final answer tool).
You have access to the following tools:"""

memory_prompts = [
    ChatMessagePromptTemplate.from_template(
        role="Previous conversation",
        template="""
    \n\n
{history}
    \n\n
    """
    )
]


def create_prompt(
    tools: Sequence[BaseTool],
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    memory_prompts: Optional[List[BasePromptTemplate]] = None,
) -> BasePromptTemplate:
    """
    Custom prompt with a slightly different positioning of memory and prompt suffix w.r.t StructuredChatAgent.create_prompt
    """


    tool_strings = []
    for tool in tools:
        args_schema = re.sub("}", "}}", re.sub("{", "{{", str(tool.args)))
        tool_strings.append(
            f"{tool.name}: {tool.description}, args: {args_schema}")
    formatted_tools = "\n".join(tool_strings)
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = format_instructions.format(tool_names=tool_names)
    template = "\n\n".join(
        [prefix, formatted_tools, format_instructions])
    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]
    _memory_prompts = memory_prompts or []
    messages = [
        SystemMessagePromptTemplate.from_template(template),
        *_memory_prompts,
        SystemMessagePromptTemplate.from_template(suffix),
        ChatMessagePromptTemplate.from_template(
            role="Question",
            template=human_message_template),
    ]
    return ChatPromptTemplate(input_variables=input_variables, messages=messages)


class GPTAgent():

    def __init__(self, memory: BaseMemory):

        self.memory = memory

        self.tools = [
            Calculator(),
            RandomNumberGenerator(),
            Search()
        ]

        self.llm = LLMClientFactory.create(
            LLM_MODELS.GPT3_5_TURBO.value,
            url=os.environ.get('LLM_URL')
        )

        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=create_prompt(
                tools=self.tools, memory_prompts=memory_prompts),
            verbose=True
        )

        self.agent = StructuredChatAgent(
            llm_chain=self.llm_chain,
            tools=self.tools,
            verbose=True
        )

        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
