from typing import Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator, model_validator

from wizard_ai.conversational_engine.form_agent import FormTool


class OnlinePurchasePayload(BaseModel):

    item: Literal["watch", "shoes", "phone", "book"] = Field(
        description="Item to purchase"
    )

    ebook: Optional[bool] = Field(
        description="Whether the book is an ebook"
    )

    quantity: int = Field(
        description="Quantity of items to purchase, between 1 and 10"
    )

    region: str = Field(
        description="Region to ship the item"
    )

    province: Optional[str] = Field(
        description="Province to ship the item"
    )

    address: str = Field(
        description="Address to ship the item"
    )

    @field_validator("quantity")
    def validate_quantity(cls, v):
        if v < 1 or v > 10:
            raise ValueError("Quantity must be between 1 and 10")
        return v
    
    @field_validator("region")
    def validate_region(cls, v):
        if v not in ["puglia", "sicilia", "toscana"]:
            raise ValueError("Region must be one of puglia, sicilia, toscana")
        return v
    
    @model_validator(mode="after")
    def validate_province(cls, model: "OnlinePurchasePayload"):
        if model.region and model.province:

            if model.region == "puglia" and model.province not in ["bari", "brindisi", "foggia", "lecce", "taranto"]:
                raise ValueError("province must be one of bari, brindisi, foggia, lecce, taranto")
            
            if model.region == "sicilia" and model.province not in ["agrigento", "caltanissetta", "catania", "enna", "messina", "palermo", "ragusa", "siracusa", "trapani"]:
                raise ValueError("province must be one of agrigento, caltanissetta, catania, enna, messina, palermo, ragusa, siracusa, trapani")
            
            if model.region == "toscana" and model.province not in ["arezzo", "firenze", "grosseto", "livorno", "lucca", "massa-carrara", "pisa", "pistoia", "prato", "siena"]:
                raise ValueError("province must be one of arezzo, firenze, grosseto, livorno, lucca, massa-carrara, pisa, pistoia, prato, siena")
        return model

    @model_validator(mode="after")
    def validate_ebook(cls, model: "OnlinePurchasePayload"):
        if model.item == "book" and model.ebook == None:
            raise ValueError("Ebook must be set for books")
        return model

class OnlinePurchase(FormTool):
    name = "OnlinePurchase"
    description = """Purchase an item from an online store"""
    args_schema: Type[BaseModel] = OnlinePurchasePayload


    def _run_when_complete(
        self,
    ) -> str:
        return "OK"
    
    def get_next_field_to_collect(
        self,
        **kwargs
    ) -> str:
        """
        The default implementation returns the first field that is not set.
        """
        if not self.form.item:
            return "item"
        
        if self.form.item == "book":
            if self.form.ebook == None:
                return "ebook"
            if self.form.ebook == True:
                return None # No more fields to collect for an ebook
            
        if not self.form.quantity:
            return "quantity"
        
        if not self.form.region:
            return "region"
        
        if not self.form.province:
            return "province"
        
        if not self.form.address:
            return "address"
        
        return None
    
    def is_form_filled(self) -> bool:
        if not self.form.item:
            return False
        if self.form.item == "book":
            if self.form.ebook == None:
                return False
        if not self.form.quantity:
            return False
        if not self.form.region:
            return False
        if not self.form.province:
            return False
        if not self.form.address:
            return False
        return True