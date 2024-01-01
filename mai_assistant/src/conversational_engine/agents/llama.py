# Chain components
# 1. Memory


from mai_assistant.src.tools.random_number_generator import RandomNumberGenerator
from mai_assistant.src.tools.calculator import Calculator
from langchain.agents import AgentType, initialize_agent


def get_memory_chain(memory: BaseChatMemory):
    return RunnablePassthrough.assign(
        history=RunnableLambda(
            memory.load_memory_variables) | itemgetter("history")
    )


# 2. Prompt
prompt = PromptTemplate.from_template("""
You are a professional personal assistant who helps people with their daily tasks.

Previous conversation:
{history}

[INST]{question}[/INST]
AI:
""")


# 3. LLM model
LLM_MODEL = os.environ.get('LLM_MODEL')
# check if llm model is valid
if LLM_MODEL not in LLM_MODELS.values():
    raise ValueError(f"LLM_MODEL must be one of {LLM_MODELS.values()}")
llm = LLMClientFactory.create(
    LLM_MODEL,
    url=os.environ.get('LLM_URL')
)


tools = [
    Calculator(),
    RandomNumberGenerator()
]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)
