from __future__ import annotations

import time
from typing import Any, Dict

from adapters.base import BaseAdapter
from executor import OrderRequest


class PaperAdapter(BaseAdapter):
    mode = "PAPER"

    def place_order(self, req: OrderRequest) -> Dict[str, Any]:
        # Paper: no network. Simulate success with a stub order id.
        order_id = f"PAPER-{int(time.time()*1000)}"
        return {
            "status": "filled",
            "order_id": order_id,
            "symbol": req.symbol,
            "side": req.side,
            "qty": req.qty,
            "type": req.type,
            "limit_price": req.limit_price,
            "client_order_id": req.client_order_id,
        }
