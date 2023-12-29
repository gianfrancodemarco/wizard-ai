from enum import Enum


class MessageType(Enum):
    ANSWER = "answer"
    TOOL_START = "TOOL_START"
    TOOL_END = "TOOL_END"
