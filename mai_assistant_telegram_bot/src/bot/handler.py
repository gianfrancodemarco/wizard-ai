import json
import logging

from telegram import Bot, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from mai_assistant_telegram_bot.src.clients import (MAIAssistantClient,
                                                    get_rabbitmq_consumer,
                                                    get_rabbitmq_producer,
                                                    get_redis_client)
from mai_assistant_telegram_bot.src.constants import MessageQueues, MessageType


class Handler:
    """
    Class that handles Telegram events.
    """

    def __init__(
        self,
        bot: Bot,
    ) -> None:
        self.bot = bot
        self.mai_assistant_client = MAIAssistantClient()
        self.rabbitmq_producer = get_rabbitmq_producer()

    async def reset_conversation_handler(
        self,
        update: Update,
        _: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Clears the conversation history."""
        self.mai_assistant_client.reset_conversation(
            chat_id=str(update.message.chat_id)
        )
        await update.message.reply_text("Conversation history cleared.")

    async def login_to_google_handler(
        self,
        update: Update,
        _: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Login to Google."""
        login_url = self.mai_assistant_client.login_to_google(
            chat_id=str(update.message.chat_id)
        )
        login_text = f"[Login to Google]({login_url})"
        await update.message.reply_markdown(f"Please complete the login process at the following URL:\n{login_text}")

    async def text_handler(
        self,
        update: Update,
        _: ContextTypes.DEFAULT_TYPE
    ) -> None:

        logging.info(f"Message received: {update}")

        await self.bot.send_chat_action(
            chat_id=update.message.chat_id,
            action=ChatAction.TYPING.value
        )

        self.rabbitmq_producer.publish(
            queue=MessageQueues.MAI_ASSISTANT_IN.value,
            message=json.dumps({
                "type": MessageType.TEXT.value,
                "chat_id": str(update.message.chat_id),
                "content": update.message.text
            })
        )
