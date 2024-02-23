from datetime import datetime
from typing import Dict, Union

from wizard_ai.conversational_engine.tools import OnlinePurchase

from .form_tool_for_evaluation import FormToolForEvaluation


class OnlinePurchaseEvaluation(OnlinePurchase, FormToolForEvaluation):    
    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        """
        Use library faker to generate random data for the form.
        """

        import faker

        fake = faker.Faker()
        
        item = fake.random_element(elements=("Watch", "Shoes", "Phone", "Book"))
        ebook = None
        quantity = None
        region = None
        province = None
        address = None

        if item == "Book":
            ebook = fake.random_element(elements=(True, False))

        if item != "Book" or ebook == False:
            quantity = fake.random_int(min=1, max=10)
            region = fake.random_element(elements=("Puglia", "Sicilia", "Toscana"))
            if region == "Puglia":
                province = fake.random_element(elements=("Bari", "Brindisi", "Foggia", "Lecce", "Taranto"))
            if region == "Sicilia":
                province = fake.random_element(elements=("Agrigento", "Caltanissetta", "Catania", "Enna", "Messina", "Palermo", "Ragusa", "Siracusa", "Trapani"))
            if region == "Toscana":
                province = fake.random_element(elements=("Arezzo", "Firenze", "Grosseto", "Livorno", "Lucca", "Massa-Carrara", "Pisa", "Pistoia", "Prato", "Siena"))
            address = fake.address()
            
        return {
            "item": item,
            "ebook": ebook,
            "quantity": quantity,
            "region": region,
            "province": province,
            "address": address
        }
            
        