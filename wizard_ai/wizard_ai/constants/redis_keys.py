from enum import Enum

class RedisKeys(Enum):
    '''"""
This is an enumeration class that represents the keys used in Redis.
RedisKeys is a subclass of the built-in enum.Enum class, which provides a way to define enumerations in Python.
Enumerations are a set of symbolic names (members) bound to unique, immutable values.
In this case, the members of RedisKeys are the keys used in Redis.
Each member has a name and a value.
The name is a string that represents the key, and the value is an auto-incrementing integer that uniquely identifies the key.
By using an enumeration, we can define a fixed set of keys and ensure that only those keys are used, preventing typos or inconsistencies in key names.
"""'''
    AGENT_STATE = 'AGENT_STATE'
    GOOGLE_CREDENTIALS = 'GOOGLE_CREDENTIALS'
    GOOGLE_STATE_TOKEN = 'GOOGLE_STATE_TOKEN'