import logging
import os
import pickle
from operator import itemgetter

import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from mai_assistant.app.llm_client import LLM_MODELS, LLMClientFactory

load_dotenv('mai_assistant/.env')

REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
redis_client = redis.Redis(password=REDIS_PASSWORD)

# Chain components
# 1. Memory

def get_memory_chain(memory: BaseChatMemory):
    return RunnablePassthrough.assign(
        history=RunnableLambda(
            memory.load_memory_variables) | itemgetter("history")
    )


# 2. Prompt
prompt = PromptTemplate.from_template("""
You are a personal assistant who helps people with their daily tasks.

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

# Add stream and file handlers to logger. Use basic config
# to avoid adding duplicate handlers when reloading server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("langchain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


class ChatPayload(BaseModel):
    conversation_id: str
    question: str


@app.delete("/conversations/delete")
def delete_conversations():
    """Delete all conversations"""
    redis_client.flushdb()
    return {"success": True}


@app.post("/chat")
def chat(data: ChatPayload):

    memory = redis_client.get(data.conversation_id)
    if memory is not None:
        memory = pickle.loads(memory)
        logger.info("Loaded memory from redis")
    else:
        memory = ConversationBufferWindowMemory(k=3, memory_key="history")

    chain = get_memory_chain(memory) | prompt | llm
    answer = chain.invoke({"question": data.question})

    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")
    logger.info(f"Answer: {answer}")

    return {"answer": answer}