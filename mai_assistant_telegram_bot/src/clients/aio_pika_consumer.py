import asyncio
import json
import os
from typing import Coroutine
import aio_pika

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST')
RABBITMQ_PORT = os.environ.get('RABBITMQ_PORT', 5672)
RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD')
RABBITMQ_USER = os.environ.get('RABBITMQ_USER')

# Convert port to int if it is a string (Due to the fact that Kubernetes automatically populates some env variables from the services)
if type(RABBITMQ_PORT) == str and ':' in RABBITMQ_PORT:
    RABBITMQ_PORT = int(RABBITMQ_PORT.split(':')[-1])

class AioPikaConsumer:
    def __init__(
        self,
        # coroutine
        on_message_callback: Coroutine[dict, None, None],
        queue_name: str
    ):
        self.queue_name = queue_name
        self.on_message_callback = on_message_callback
        self.connection = None
        self.lock = asyncio.Lock()

    async def on_message(self, message):
        async with message.process():
            async with self.lock:
                body = json.loads(message.body.decode())
                print(f" [x] Received {body}")
                await self.on_message_callback(body)
                # Your processing logic here
                # For example, you can call an async function:
                # await process_message(body)

    async def setup_consumer(self):
        self.connection = await aio_pika.connect_robust(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            login=RABBITMQ_USER,
            password=RABBITMQ_PASSWORD,
        )
        channel = await self.connection.channel()

        queue = await channel.declare_queue(self.queue_name, durable=True)

        await queue.consume(self.on_message)

    async def run_consumer(self):
        await self.setup_consumer()