from enum import Enum

class MessageQueues(Enum):
    """The MessageQueues class is an enumeration that represents different types of message queues. It allows for easy referencing and handling of different message queue types in code."""
    wizard_ai_IN = 'wizard_ai_in'
    wizard_ai_OUT = 'wizard_ai_out'