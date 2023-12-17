import logging
import os
import pickle
from operator import itemgetter

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from mai_assistant.src.llm_client import LLM_MODELS, LLMClientFactory

load_dotenv('mai_assistant/.env')

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