
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

# TODO: Delete this
def make_optional_model(original_model: BaseModel) -> BaseModel:
    """
    Takes a Pydantic model and returns a new model with all attributes optional.
    """
    optional_attributes = {attr_name: (
        attr_type, None) for attr_name, attr_type in original_model.__annotations__.items()}

    # Define a custom Pydantic model with optional attributes
    new_class_name = original_model.__name__ + 'Optional'
    OptionalModel = create_model(
        new_class_name,
        **optional_attributes,
        __base__=original_model
    )
    OptionalModel.model_config["validate_assignment"] = True

    return OptionalModel
