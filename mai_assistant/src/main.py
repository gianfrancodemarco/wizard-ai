import logging

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv('mai_assistant/.env')

from mai_assistant.src.controllers.chat import chat_router
from mai_assistant.src.controllers.redis import redis_router

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

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

app.include_router(redis_router)
app.include_router(chat_router)
