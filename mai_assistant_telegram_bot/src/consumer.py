import json
import logging
import re

from telegram import Bot
from telegram.constants import ParseMode

from mai_assistant_telegram_bot.src.clients import get_redis_client
from mai_assistant_telegram_bot.src.constants import Emojis, MessageType


class MAIAssistantConsumer:

    def __init__(
        self,
        bot: Bot
    ) -> None:
        self.bot = bot
        self.redis_client = get_redis_client()

    async def on_message_callback(
        self,
        message: str
    ) -> None:
        """Callback to be called when a message is received from the RabbitMQ queue."""

        if "type" not in message:
            logging.error(f"Received message without type: {message}")
            return

        if message["type"] == MessageType.TOOL_START.value:

            sent_message = await self.bot.send_message(
                chat_id=message["chat_id"],
                text=f"""{Emojis.LOADING.value} {message["content"]}""",
                parse_mode=ParseMode.MARKDOWN_V2
            )

            self.redis_client.hset(
                f"telegram.{message['chat_id']}",
                "last_tool_start_message",
                json.dumps({
                    "content": message["content"],
                    "message_id": sent_message.message_id
                })
            )

        elif message["type"] == MessageType.TOOL_END.value:

            last_tool_start_message = self.redis_client.hget(
                f"telegram.{message['chat_id']}",
                "last_tool_start_message"
            )

            if not last_tool_start_message:
                logging.error(
                    f"Received tool end message without tool start message: {message}")
                return

            last_tool_start_message = json.loads(last_tool_start_message)

            await self.bot.edit_message_text(
                chat_id=message["chat_id"],
                message_id=last_tool_start_message["message_id"],
                text=f"""{Emojis.DONE.value} {last_tool_start_message["content"]}""",
                parse_mode=ParseMode.MARKDOWN_V2
            )

            self.redis_client.hdel(
                f"telegram.{message['chat_id']}",
                "last_tool_start_message"
            )

        # TODO: Find a better way to escape Telegram markdown characters
        elif message["type"] == MessageType.TEXT.value:
            await self.bot.send_message(
                chat_id=message["chat_id"],
                text=re.sub(r'([|{\[\]*_~}+)(#>!=\-.])',
                            r'\\\1', message["content"]),
                parse_mode=ParseMode.MARKDOWN_V2
            )
