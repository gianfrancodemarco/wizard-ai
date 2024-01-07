from pydantic import BaseModel, create_model, field_validator
from datetime import datetime

from pydantic import BaseModel, Field

from pydantic import field_validator
from pydantic import BaseModel, field_validator
from dateutil.parser import parse
class YourModel(BaseModel):
    attr1: int
    attr2: str

    @field_validator('attr1')
    def attr1_is_positive(cls, v):
        if v <= 0:
            raise ValueError('attr1 must be positive')
        return v

# Create a new Pydantic model with optional attributes
def make_optional_model(original_model):
    optional_attributes = {attr_name: (attr_type, None) for attr_name, attr_type in original_model.__annotations__.items()}

    # Define a custom Pydantic model with optional attributes
    OptionalModel = create_model(
        original_model.__name__ + 'Optional',
        **optional_attributes,
        __base__=original_model
    )

    #Validators are not working !!!

    return OptionalModel

# Create a new Pydantic model with optional attributes
OptionalModel = make_optional_model(YourModel)

# Now, OptionalModel has 


class Test(BaseModel):
    pass


class CreateCalendarEventPayload(BaseModel):
    summary: str = Field(
        description="Title of the event",
    )
    description: str = Field(
        description="Description of the event",
    )
    start: datetime = Field(
        description="Start date of the event",
    )
    end: datetime = Field(
        description="End date of the event",
    )

    @field_validator("start", "end", mode="before")
    def parse_date(cls, v, values):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return parse(v)

#CreateCalendarEventPayload(summary="1", description="1", start="2023-10-10 00:00:00", end="2023-10-10 00:00:00")
#CreateCalendarEventPayload(summary="1", description="1", start="2023-10-10 00:00:00", end="10/10/2023 00:00:00")
create_calendar_event_payload_optional = make_optional_model(CreateCalendarEventPayload)
create_calendar_event_payload_optional(summary="1", description="1", start=" 00:00:00", end="10/10/2023 00:00:00")