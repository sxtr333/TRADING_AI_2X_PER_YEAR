from __future__ import annotations

from typing import Any, Dict

from adapters.base import BaseAdapter
from executor import OrderRequest


class LiveStubAdapter(BaseAdapter):
    mode = "LIVE_STUB"

    def place_order(self, req: OrderRequest) -> Dict[str, Any]:
        raise NotImplementedError("Implement locally")
