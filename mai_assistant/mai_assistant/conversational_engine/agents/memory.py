import json
import logging
import pickle
from typing import Dict, Union

import redis
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import BaseModel

from mai_assistant.constants import RedisKeys
from mai_assistant.conversational_engine.langchain_extention.form_tool import \
    FormTool
from mai_assistant.conversational_engine.langchain_extention.structured_agent_executor import (
    AgentState, make_optional_model)
from mai_assistant.constants import RedisKeys
logger = logging.getLogger(__name__)


class StoredAgentState:
    memory: BaseChatMemory
    form: BaseModel
    active_form_tool: Union[Dict, FormTool]

    def __init__(
        self,
        memory: BaseChatMemory,
        form: BaseModel,
        active_form_tool: Union[Dict, FormTool]
    ) -> None:
        self.memory = memory
        self.form = form
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
        active_form_tool = stored_agent_state.active_form_tool
        if active_form_tool:
            loaded = json.loads(stored_agent_state.form)
            loaded = {key: value for key, value in loaded.items() if value}
            context_form_class = make_optional_model(active_form_tool.args_schema)
            stored_agent_state.form = context_form_class(**loaded)
    else:
        stored_agent_state = StoredAgentState(
            memory=ConversationBufferWindowMemory(
                k=3,
                memory_key="history",
                human_prefix="Human",
                ai_prefix="Answer",
                input_key="messages",
                return_messages=True
            ),
            form=None,
            active_form_tool=None
        )
    return stored_agent_state

def store_agent_state(
    redis_client: redis.Redis,
    chat_id: str,
    agent_state: AgentState
):
    
    # this shouldn't be here
    if agent_state.active_form_tool:
        agent_state.form = agent_state.form.model_dump_json()

    stored_agent_state = StoredAgentState(
        memory=agent_state.memory,
        form=agent_state.form,
        active_form_tool=agent_state.active_form_tool
    )

    redis_client.hset(
        chat_id,
        RedisKeys.AGENT_STATE.value,        
        pickle.dumps(stored_agent_state)
    )

# def get_stored_memory(
#         redis_client: redis.Redis,
#         chat_id: str
# ) -> BaseChatMemory:
#     memory = redis_client.hget(
#         chat_id,
#         "memory"
#     )
#     if memory is not None:
#         memory = pickle.loads(memory)
#         logger.info("Loaded memory from redis")
#     else:
#         memory = ConversationBufferWindowMemory(
#             k=3,
#             memory_key="history",
#             human_prefix="Human",
#             ai_prefix="Answer",
#             input_key="messages",
#             return_messages=True
#         )
#     return memory


# def get_stored_context(
#     redis_client: redis.Redis,
#     chat_id: str
# ) -> dict:
#     context = redis_client.hget(
#         chat_id,
#         "context"
#     )
#     if context is not None:
#         context: AgentState = pickle.loads(context)
#         active_form_tool = context.get("active_form_tool")

#         if active_form_tool:
#             loaded = json.loads(context.form)
#             loaded = {key: value for key, value in loaded.items() if value}
#             context_form_class = make_optional_model(active_form_tool.args_schema)
#             context.form = context_form_class(**loaded)
#         logger.info("Loaded context from redis")
#     else:
#         context = AgentState(
            
#         )
#     return context
