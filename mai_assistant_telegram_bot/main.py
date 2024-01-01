import asyncio
import logging
import os
import sys

from mai_assistant_telegram_bot.src.clients import (get_rabbitmq_consumer, get_redis_client)
from mai_assistant_telegram_bot.src.constants import MessageQueues

from mai_assistant_telegram_bot.src.bot.bot import MaiAssistantTelegramBot
from mai_assistant_telegram_bot.src.consumer import MAIAssistantConsumer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


redis_client = get_redis_client()

# Add stream and file handlers to logger. Use basic config
# to avoid adding duplicate handlers when reloading server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":

    # Init the bot
    bot = MaiAssistantTelegramBot()

    # Create a consumer for RabbitMQ that handles update from MAIAssistant
    mai_assistant_consumer = MAIAssistantConsumer(bot=bot.telegram_bot)
    rabbitmq_consumer = get_rabbitmq_consumer(
        queue_name=MessageQueues.MAI_ASSISTANT_OUT.value,
        on_message_callback=mai_assistant_consumer.on_message_callback
    )

    # Start the RabbitMQ consumer
    asyncio.get_event_loop().create_task(rabbitmq_consumer.run_consumer())

    # Start the bot at last because it is a blocking operation
    bot.start()