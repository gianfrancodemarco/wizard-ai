from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Union


class FormToolForEvaluation(ABC):
    def _run_when_complete(
        self,
        *args
    ) -> str:
        return "OK"

    @abstractmethod
    def get_random_payload(
        self
    ) -> Dict[str, Union[str, datetime]]:
        raise NotImplementedError