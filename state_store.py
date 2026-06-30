from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict

from executor import OrderRequest


class StateStore:
    def __init__(self, path: str = "executor_state.json", logger=None) -> None:
        self.path = path
        self.log = logger
        self._ensure_state()

    def _ensure_state(self) -> None:
        if not os.path.exists(self.path):
            self._atomic_write(
                {
                    "date": self._today(),
                    "orders_today": 0,
                    "positions": {},
                    "last_order_ts": None,
                    "orders": [],
                }
            )

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _read(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="state_", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def orders_today_count(self) -> int:
        st = self._read()
        if st.get("date") != self._today():
            st["date"] = self._today()
            st["orders_today"] = 0
            self._atomic_write(st)
        return int(st.get("orders_today", 0))

    def get_position(self, symbol: str) -> float:
        st = self._read()
        return float(st.get("positions", {}).get(symbol, 0.0))

    def record_order(self, req: OrderRequest, record: Dict[str, Any]) -> None:
        st = self._read()
        if st.get("date") != self._today():
            st["date"] = self._today()
            st["orders_today"] = 0
            st["orders"] = []

        st["orders_today"] = int(st.get("orders_today", 0)) + 1
        st["last_order_ts"] = datetime.now(timezone.utc).isoformat()

        # update positions
        positions = st.get("positions", {})
        cur = float(positions.get(req.symbol, 0.0))
        delta = req.qty if req.side.upper() == "BUY" else -req.qty
        positions[req.symbol] = cur + delta
        st["positions"] = positions

        # append order record
        orders = st.get("orders", [])
        orders.append(record)
        st["orders"] = orders

        self._atomic_write(st)
