
import json
import logging
import os
import sys

from telegram import Bot, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from mai_assistant_telegram_bot.src.clients.mai_assistant import \
    MAIAssistantClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from telegram import Update, Bot
from telegram.constants import ChatAction


TOKEN = os.getenv("TELEGRAM_API_TOKEN")
bot = Bot(token=TOKEN)

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

    await bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING.value
    )
    response = MAIAssistantClient().chat(
        conversation_id=str(update.message.chat_id),
        message=update.message.text
    )
    await update.message.reply_text(response["answer"])


async def text_handler_websocket(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:

    logging.info(f"Message received: {update}")

    async with MAIAssistantClient().chat_ws() as websocket:

        # Ask the question to the MAI Assistant
        await websocket.send(json.dumps({
            "conversation_id": str(update.message.chat_id),
            "question": update.message.text
        }))

        answer = None
        while answer is None:
            
            await bot.send_chat_action(
                chat_id=update.message.chat_id,
                action=ChatAction.TYPING.value
            )

            response = json.loads(await websocket.recv())
            # The message is a tool usage
            if response["type"] == "tool":
                await update.message.reply_text(f"""✔️ {response["content"]}""")

            # The message is the final answer
            elif response["type"] == "answer":
                answer = response["answer"]
                await update.message.reply_text(response["answer"])


def start_bot() -> None:
    """Start the bot."""
    logging.info("Starting the telegram bot")
    # Create the Application and pass it your bot's token.

    application = Application.builder()\
        .bot(bot)\
        .build()

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, text_handler_websocket))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(close_loop=False)

    logging.info("Telegram bot started")


if __name__ == "__main__":
    start_bot()
