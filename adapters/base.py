from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from executor import OrderRequest


class TransientError(Exception):
    pass


class BaseAdapter(ABC):
    mode = "PAPER"

    def __init__(self, logger=None) -> None:
        self.log = logger

    @abstractmethod
    def place_order(self, req: OrderRequest) -> Dict[str, Any]:
        raise NotImplementedError
