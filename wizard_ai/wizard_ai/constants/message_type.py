from enum import Enum

class MessageType(Enum):
    """This class represents an enumeration of different message types. It is used to define and store the possible values for a message type."""
    TEXT = 'TEXT'
    TOOL_START = 'TOOL_START'
    TOOL_END = 'TOOL_END'