import logging
import os
import pickle

import redis
from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from mai_assistant.app.llm_client import LLM_MODELS, LLMClientFactory
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv('mai_assistant/.env')

REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
redis_client = redis.Redis(password=REDIS_PASSWORD)

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


prompt = PromptTemplate.from_template("""
You are a personal assistant who helps people with their daily tasks.

Previous conversation:
{chat_history}

Human: {question}
AI: 
""")

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
        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    conversation = LLMChain(
        # llm=OpenAI(),
        llm=LLMClientFactory.create(
            #LLM_MODELS.GPT3_5_TURBO.value,
            LLM_MODELS.LLAMA_2_7B_CHAT_HF.value,
            url=os.environ.get('LLM_URL')
        ),
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    answer = conversation.predict(question=data.question)
    redis_client.set(data.conversation_id, pickle.dumps(memory))
    logger.info("Saved memory to redis")
    logger.info(f"Answer: {answer}")

    return {"answer": answer}