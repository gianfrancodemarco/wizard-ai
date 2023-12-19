import logging
import os
import pickle
from operator import itemgetter

from fastapi import APIRouter
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from mai_assistant.src.dependencies import RedisClient
from mai_assistant.src.llm_client import LLM_MODELS, LLMClientFactory

logger = logging.getLogger(__name__)

# Chain components
# 1. Memory
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


class ChatPayload(BaseModel):
    conversation_id: str
    question: str


chat_router = APIRouter()

def start_chain_logger(input):
    logging.info("\n\n---------------------------------- STARTING CHAIN ----------------------------------\n")
    return input 

def prompt_logger(prompt):
    logging.info(f"Calling LLM with prompt:")
    logging.info(prompt.text)
    return prompt

def response_logger(response):
    logging.info(f"Received response:")
    logging.info(response)
    return response

def end_chain_logger(input):
    logging.info("\n\n----------------------------------- ENDING CHAIN ----------------------------------\n\n")
    return input 


@chat_router.post("/chat")
def chat(data: ChatPayload, redis_client: RedisClient):

    memory = redis_client.get(data.conversation_id)
    if memory is not None:
        memory = pickle.loads(memory)
        logging.info("Loaded memory from redis")
    else:
        memory = ConversationBufferWindowMemory(k=3, memory_key="history")

    # make this verbose
    chain = start_chain_logger | \
        get_memory_chain(memory) | \
            prompt | prompt_logger | \
                llm\
                      | response_logger |\
                      end_chain_logger
    input = {"question": data.question}
    answer = chain.invoke(input)
    memory.save_context(input, {"history": answer})
    
    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")
    
    return {"answer": answer}