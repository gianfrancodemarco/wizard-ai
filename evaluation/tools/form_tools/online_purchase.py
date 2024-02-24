from datetime import datetime
from typing import Dict, Union

from wizard_ai.conversational_engine.tools import OnlinePurchase


class OnlinePurchaseEvaluation(OnlinePurchase):    

    def _run_when_complete(
        self,
        *args,
        **kwargs
    ) -> str:
        return "OK"

    def get_random_payload(self) -> Dict[str, Union[str, datetime]]:
        """
        Use library faker to generate random data for the form.
        """

        import faker

        fake = faker.Faker()
        
        item = fake.random_element(elements=("watch", "shoes", "phone", "book"))
        ebook = None
        quantity = None
        region = None
        province = None
        address = None

        if item == "book":
            ebook = fake.random_element(elements=(True, False))

        if item != "book" or ebook == False:
            quantity = fake.random_int(min=1, max=10)
            region = fake.random_element(elements=("puglia", "sicilia", "toscana"))
            if region == "puglia":
                province = fake.random_element(elements=("bari", "brindisi", "foggia", "lecce", "taranto"))
            if region == "sicilia":
                province = fake.random_element(elements=("agrigento", "caltanissetta", "catania", "enna", "messina", "palermo", "ragusa", "siracusa", "trapani"))
            if region == "toscana":
                province = fake.random_element(elements=("arezzo", "firenze", "grosseto", "livorno", "lucca", "massa-carrara", "pisa", "pistoia", "prato", "siena"))
            address = fake.address()
            
        return {
            "item": item,
            "ebook": ebook,
            "quantity": quantity,
            "region": region,
            "province": province,
            "address": address
        }
            
        