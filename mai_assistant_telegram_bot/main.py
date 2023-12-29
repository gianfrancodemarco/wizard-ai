
from mai_assistant_telegram_bot.src.constants import Emojis, MessageType
from telegram.constants import ChatAction
import json
import logging
import os
import sys

from telegram import Bot, Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from mai_assistant_telegram_bot.src.clients.mai_assistant import \
    MAIAssistantClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


TOKEN = os.getenv("TELEGRAM_API_TOKEN")
bot = Bot(token=TOKEN)
mai_assistant_client = MAIAssistantClient()

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
        conversation_id=str(update.message.chat_id)
    )
    await update.message.reply_text("Conversation history cleared.")


async def text_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:

    logging.info(f"Message received: {update}")

    await bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.TYPING.value
    )
    response = mai_assistant_client.chat(
        conversation_id=str(update.message.chat_id),
        message=update.message.text
    )
    await update.message.reply_text(response["answer"])


async def text_handler_websocket(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:

    logging.info(f"Message received: {update}")

    async with mai_assistant_client.chat_ws() as websocket:

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

            mai_assistant_update = json.loads(await websocket.recv())

            if mai_assistant_update["type"] == MessageType.TOOL_START.value:
                # We keep the tool_start_update because the tool_end_update will not have all the information
                tool_start_update = mai_assistant_update
                tool_start_telegram_message_ref = await update.message.reply_text(f"""{Emojis.LOADING.value} {tool_start_update["content"]}""")
            elif mai_assistant_update["type"] == MessageType.TOOL_END.value:
                await tool_start_telegram_message_ref.edit_text(f"""{Emojis.DONE.value} {tool_start_update["content"]}""")
            elif mai_assistant_update["type"] == MessageType.ANSWER.value:
                answer = mai_assistant_update["answer"]
                await update.message.reply_text(mai_assistant_update["answer"])


async def post_init(application: Application) -> None:

    await application.bot.set_my_commands([
        ("reset", "Clears the conversation history.")
    ])


def start_bot() -> None:
    """Start the bot."""
    logging.info("Starting the telegram bot")
    # Create the Application and pass it your bot's token.

    application = Application.builder()\
        .bot(bot)\
        .post_init(post_init)\
        .build()

    # Add command handler to application
    application.add_handler(CommandHandler(
        "reset", reset_conversation_handler))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, text_handler_websocket))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(close_loop=False)

    logging.info("Telegram bot started")


if __name__ == "__main__":
    start_bot()
