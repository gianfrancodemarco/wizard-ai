"""
This code defines classes and functions related to storing and retrieving the state of an agent in a chat application using Redis as a backend. Here's a brief summary of what the code does:

1. `StoredAgentState`: This class represents the state of an agent, including its chat memory and active form tool. It has methods to serialize the state into a pickle format and deserialize it back.

2. `get_stored_agent_state`: This function retrieves the stored agent state from Redis based on a given chat ID. If the state exists in Redis, it loads it using `StoredAgentState.from_pickle` method, otherwise creates a new state.

3. `store_agent_state`: This function stores the agent state into Redis for a specific chat ID. It creates a new `StoredAgentState` object with the provided agent state information and serializes it before storing in Redis.

Overall, this code provides a mechanism to save and retrieve the state of an agent within a conversation using Redis as a data store.
"""
import logging
import os
import pickle
from typing import Dict, Optional, Union

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory

from wizard_ai.constants import RedisKeys
from wizard_ai.conversational_engine.form_agent.form_tool import AgentState, FormTool

logger = logging.getLogger(__name__)

HISTORY_LENGTH = os.getenv("HISTORY_LENGTH", 20)


class StoredAgentState:
    memory: Optional[BaseChatMemory]
    active_form_tool: Optional[Union[Dict, FormTool]]

    def __init__(
        self,
        memory: Optional[BaseChatMemory] = None,
        active_form_tool: Union[Dict, FormTool] = None
    ) -> None:

        if memory is None:
            memory = ConversationBufferWindowMemory(
                k=HISTORY_LENGTH,
                memory_key="history",
                human_prefix="Human",
                ai_prefix="Answer",
                input_key="messages",
                return_messages=True
            )

        self.memory = memory
        self.active_form_tool = active_form_tool

    def to_pickle(self):
        if self.active_form_tool:
            self.active_form_tool.form = self.active_form_tool.form.model_dump_json()
            self.active_form_tool.args_schema = self.active_form_tool.args_schema_
            self.active_form_tool.args_schema_ = None
        return pickle.dumps(self)

    @staticmethod
    def from_pickle(pickled: str):
        stored_agent_state = pickle.loads(pickled)
        if stored_agent_state.active_form_tool:
            stored_agent_state.active_form_tool.args_schema_ = stored_agent_state.active_form_tool.args_schema
            stored_agent_state.active_form_tool.init_state()
        return stored_agent_state


def get_stored_agent_state(
    redis_client: redis.Redis,
    chat_id: str
) -> StoredAgentState:
    stored_agent_state = redis_client.hget(
        chat_id,
        RedisKeys.AGENT_STATE.value
    )

    if stored_agent_state is not None:
        stored_agent_state: StoredAgentState = StoredAgentState.from_pickle(
            stored_agent_state)
        logger.info("Loaded agent state from redis")
    else:
        stored_agent_state = StoredAgentState()
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
        stored_agent_state.to_pickle()
    )
