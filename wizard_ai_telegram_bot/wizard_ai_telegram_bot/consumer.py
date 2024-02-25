import json
import logging

from telegram import Bot
from telegram.constants import ParseMode

from wizard_ai_telegram_bot.clients import get_redis_client
from wizard_ai_telegram_bot.constants import Emojis, MessageType


class MAIAssistantConsumer:
    """
    Class that consumes messages generated by the Wizard AI and sends them to the Telegram user.
    """

    def __init__(
        self,
        bot: Bot
    ) -> None:
        self.bot = bot
        self.redis_client = get_redis_client()

    def _sanitize_text_for_telegram(
        self,
        message: str
    ) -> str:
        """Sanitizes a message to be sent to Telegram."""
        return message

    async def on_message_callback(
        self,
        message: str
    ) -> None:
        """Callback to be called when a message is received from the RabbitMQ queue."""

        if "type" not in message:
            logging.error(f"Received message without type: {message}")
            return

        message_processors = {
            MessageType.TOOL_START.value: self.__process_tool_start_message,
            MessageType.TOOL_END.value: self.__process_tool_end_message,
            MessageType.TEXT.value: self.__process_text_message
        }

        if message["type"] not in message_processors:
            logging.error(f"Received message with unknown type: {message}")
            return

        await message_processors[message["type"]](message)

    async def __process_tool_start_message(
        self,
        message: str
    ) -> None:
        """Processes a tool start message."""

        text = self._sanitize_text_for_telegram(
            f"""{Emojis.LOADING.value} {message["content"]}""")

        # TODO: text is too long
        sent_message = await self.bot.send_message(
            chat_id=message["chat_id"],
            text=text,
            parse_mode=ParseMode.HTML
        )

        self.redis_client.hset(
            f"telegram.{message['chat_id']}",
            "last_tool_start_message",
            json.dumps({
                "content": message["content"],
                "message_id": sent_message.message_id
            })
        )

    async def __process_tool_end_message(
        self,
        message: str
    ) -> None:
        """Processes a tool end message."""

        last_tool_start_message = self.redis_client.hget(
            f"telegram.{message['chat_id']}",
            "last_tool_start_message"
        )

        if not last_tool_start_message:
            logging.error(
                f"Received tool end message without tool start message: {message}")
            return

        last_tool_start_message = json.loads(last_tool_start_message)

        text = self._sanitize_text_for_telegram(
            f"""{Emojis.DONE.value} {last_tool_start_message["content"]}""")
        await self.bot.edit_message_text(
            chat_id=message["chat_id"],
            message_id=last_tool_start_message["message_id"],
            text=text,
            parse_mode=ParseMode.HTML
        )

        self.redis_client.hdel(
            f"telegram.{message['chat_id']}",
            "last_tool_start_message"
        )

    async def __process_text_message(
        self,
        message: str
    ) -> None:
        """Processes a text message."""

        text = self._sanitize_text_for_telegram(message["content"])
        await self.bot.send_message(
            chat_id=message["chat_id"],
            text=text,
            parse_mode=ParseMode.HTML
        )
