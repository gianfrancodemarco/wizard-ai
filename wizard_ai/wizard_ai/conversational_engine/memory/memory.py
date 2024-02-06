import logging
import pickle
from typing import Dict, Union

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory

from wizard_ai.constants import RedisKeys
from wizard_ai.conversational_engine.langchain_extention.form_tool import (
    AgentState, FormTool)

logger = logging.getLogger(__name__)


class StoredAgentState:
    memory: BaseChatMemory
    active_form_tool: Union[Dict, FormTool]

    def __init__(
        self,
        memory: BaseChatMemory,
        active_form_tool: Union[Dict, FormTool]
    ) -> None:
        self.memory = memory
        self.active_form_tool = active_form_tool


def get_stored_agent_state(
    redis_client: redis.Redis,
    chat_id: str
) -> StoredAgentState:
    stored_agent_state = redis_client.hget(
        chat_id,
        RedisKeys.AGENT_STATE.value
    )

    if stored_agent_state is not None:
        stored_agent_state: StoredAgentState = pickle.loads(stored_agent_state)
        logger.info("Loaded agent state from redis")
    else:
        stored_agent_state = StoredAgentState(
            memory=ConversationBufferWindowMemory(
                k=5,
                memory_key="history",
                human_prefix="Human",
                ai_prefix="Answer",
                input_key="messages",
                return_messages=True
            ),
            active_form_tool=None
        )
    return stored_agent_state


def store_agent_state(
    redis_client: redis.Redis,
    chat_id: str,
    agent_state: AgentState
):
    stored_agent_state = StoredAgentState(
        memory=agent_state.memory,
        active_form_tool=agent_state.active_form_tool
    )

    redis_client.hset(
        chat_id,
        RedisKeys.AGENT_STATE.value,
        pickle.dumps(stored_agent_state)
    )
