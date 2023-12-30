
import asyncio
import json
import logging
import os
import sys

from telegram import Bot, Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from mai_assistant_telegram_bot.src.clients import (MAIAssistantClient,
                                                    get_rabbitmq_consumer,
                                                    get_rabbitmq_producer,
                                                    get_redis_client)
from mai_assistant_telegram_bot.src.constants import (Emojis, MessageQueues,
                                                      MessageType)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


TOKEN = os.getenv("TELEGRAM_API_TOKEN")
bot = Bot(token=TOKEN)
mai_assistant_client = MAIAssistantClient()



async def on_message_callback(message: str) -> None:
    """Callback to be called when a message is received from the RabbitMQ queue."""

    if "type" not in message:
        logging.error(f"Received message without type: {message}")
        return

    if message["type"] == MessageType.TOOL_START.value:

        sent_message = await bot.send_message(
            chat_id=message["chat_id"],
            text=f"""{Emojis.LOADING.value} {message["content"]}"""
        )

        redis_client.hset(
            f"telegram.{message['chat_id']}",
            "last_tool_start_message",
            json.dumps({
                "content": message["content"],
                "message_id": sent_message.message_id
            })
        )

    elif message["type"] == MessageType.TOOL_END.value:

        last_tool_start_message = redis_client.hget(
            f"telegram.{message['chat_id']}",
            "last_tool_start_message"
        )

        if not last_tool_start_message:
            logging.error(
                f"Received tool end message without tool start message: {message}")
            return

        last_tool_start_message = json.loads(last_tool_start_message)

        await bot.edit_message_text(
            chat_id=message["chat_id"],
            message_id=last_tool_start_message["message_id"],
            text=f"""{Emojis.DONE.value} {last_tool_start_message["content"]}"""
        )

        redis_client.hdel(
            f"telegram.{message['chat_id']}",
            "last_tool_start_message"
        )

    elif message["type"] == MessageType.TEXT.value:
        await bot.send_message(
            chat_id=message["chat_id"],
            text=message["content"]
        )

redis_client = get_redis_client()
rabbitmq_consumer = get_rabbitmq_consumer(
    queue_name=MessageQueues.MAI_ASSISTANT_OUT.value,
    on_message_callback=on_message_callback
)
rabbitmq_producer = get_rabbitmq_producer()

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


async def reset_conversation_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history."""
    mai_assistant_client.reset_conversation(
        chat_id=str(update.message.chat_id)
    )
    await update.message.reply_text("Conversation history cleared.")


async def login_to_google_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Login to Google."""
    login_url = mai_assistant_client.login_to_google(
        chat_id=str(update.message.chat_id)
    )
    login_text = f"[Login to Google]({login_url})"
    await update.message.reply_markdown(f"Please complete the login process at the following URL:\n{login_text}")


async def text_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:

    logging.info(f"Message received: {update}")

    await bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING.value
    )

    rabbitmq_producer.publish(
        queue=MessageQueues.MAI_ASSISTANT_IN.value,
        message=json.dumps({
            "type": MessageType.TEXT.value,
            "chat_id": str(update.message.chat_id),
            "content": update.message.text
        })
    )


async def post_init(application: Application) -> None:

    await application.bot.set_my_commands([
        ("reset", "Clears the conversation history."),
        ("login_to_google", "Login to Google.")
    ])


def start_bot() -> None:
    """Start the bot."""
    logging.info("Starting the telegram bot")

    application = Application.builder()\
        .bot(bot)\
        .post_init(post_init)\
        .build()

    # Add command handler to application
    application.add_handler(CommandHandler(
        "reset", reset_conversation_handler))
    application.add_handler(CommandHandler(
        "login_to_google", login_to_google_handler))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, text_handler))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(close_loop=False)

    logging.info("Telegram bot started")


if __name__ == "__main__":

    # Start the RabbitMQ consumer
    asyncio.get_event_loop().create_task(rabbitmq_consumer.run_consumer())

    # Start the bot
    start_bot()
