
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from mai_assistant_telegram_bot.src.clients.mai_assistant import \
    MAIAssistantClient

TOKEN = os.getenv("TELEGRAM_API_TOKEN")

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


async def text_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:

    logging.info(f"Message received: {update}")

    await update._bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING.value
    )
    response = MAIAssistantClient().chat(
        conversation_id=str(update.message.chat_id),
        message=update.message.text
    )
    await update.message.reply_text(response["answer"])

def start_bot() -> None:

    """Start the bot."""
    logging.info("Starting the telegram bot")
    # Create the Application and pass it your bot's token.
    application = Application.builder()\
        .token(TOKEN)\
        .build()

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(close_loop=False)
    logging.info("Telegram bot started")

if __name__ == "__main__":
    start_bot()