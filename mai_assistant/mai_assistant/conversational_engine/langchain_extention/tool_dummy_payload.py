from pydantic import BaseModel


class ToolDummyPayload(BaseModel):
    """
    We cannot pass directly the BaseModel class as args_schema as pydantic will raise errors,
    so we need to create a dummy class that inherits from BaseModel.
    """
