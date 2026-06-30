from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from executor import OrderRequest
from state_store import StateStore


class RiskViolation(Exception):
    pass


@dataclass
class RiskLimits:
    max_orders_per_day: int
    max_position_per_symbol: float
    daily_loss_limit: float


class RiskManager:
    def __init__(self, pnl_path: str = "pnl_today.json") -> None:
        self.pnl_path = pnl_path
        self.limits = RiskLimits(
            max_orders_per_day=int(os.environ.get("MAX_ORDERS_PER_DAY", "50")),
            max_position_per_symbol=float(os.environ.get("MAX_POSITION_PER_SYMBOL", "10")),
            daily_loss_limit=float(os.environ.get("DAILY_LOSS_LIMIT", "-100.0")),
        )

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _load_pnl(self) -> float:
        try:
            with open(self.pnl_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") != self._today():
                return 0.0
            return float(data.get("pnl", 0.0))
        except FileNotFoundError:
            return 0.0
        except Exception:
            return 0.0

    def check(self, req: OrderRequest, state: StateStore) -> None:
        # orders per day
        orders_today = state.orders_today_count()
        if orders_today >= self.limits.max_orders_per_day:
            raise RiskViolation(f"MAX_ORDERS_PER_DAY exceeded: {orders_today} >= {self.limits.max_orders_per_day}")

        # per-symbol position cap
        current_pos = state.get_position(req.symbol)
        projected = current_pos + (req.qty if req.side.upper() == "BUY" else -req.qty)
        if abs(projected) > self.limits.max_position_per_symbol:
            raise RiskViolation(
                f"MAX_POSITION_PER_SYMBOL exceeded: abs({projected}) > {self.limits.max_position_per_symbol}"
            )

        # daily loss limit
        pnl = self._load_pnl()
        if pnl <= self.limits.daily_loss_limit:
            raise RiskViolation(f"DAILY_LOSS_LIMIT breached: pnl={pnl} <= {self.limits.daily_loss_limit}")
