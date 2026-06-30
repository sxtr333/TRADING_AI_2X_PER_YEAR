from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Dict, Optional

from risk import RiskManager, RiskViolation
from state_store import StateStore
from adapters.base import BaseAdapter, TransientError
from adapters.paper import PaperAdapter


@dataclass
class OrderRequest:
    symbol: str
    side: str  # BUY/SELL
    qty: float
    type: str  # MARKET/LIMIT
    limit_price: Optional[float] = None
    client_order_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Executor:
    def __init__(
        self,
        adapter: Optional[BaseAdapter] = None,
        state_path: str = "executor_state.json",
        pnl_path: str = "pnl_today.json",
        logger: Optional[Logger] = None,
        dry_run: bool = False,
        require_confirm: bool = False,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        self.adapter = adapter or PaperAdapter(logger=logger)
        self.state = StateStore(state_path, logger=logger)
        self.risk = RiskManager(pnl_path=pnl_path)
        self.log = logger
        self.dry_run = dry_run
        self.require_confirm = require_confirm
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

    def _log(self, msg: str) -> None:
        if self.log:
            self.log.info(msg)
        else:
            print(msg, file=sys.stderr)

    def _check_kill_switch(self) -> None:
        if os.environ.get("EXECUTOR_KILL_SWITCH") == "1":
            raise RiskViolation("Kill-switch active via EXECUTOR_KILL_SWITCH=1")
        if os.path.exists("KILL_SWITCH"):
            raise RiskViolation("Kill-switch file present: ./KILL_SWITCH")

    def _confirm(self, req: OrderRequest) -> None:
        if not self.require_confirm:
            return
        prompt = f"CONFIRM ORDER? {req.side} {req.qty} {req.symbol} ({req.type}) -> type YES: "
        answer = input(prompt).strip().upper()
        if answer != "YES":
            raise RiskViolation("Order not confirmed by user")

    def place_order(self, req: OrderRequest) -> Dict[str, Any]:
        self._check_kill_switch()

        # normalize
        req.side = req.side.upper()
        req.type = req.type.upper()

        # risk checks
        self.risk.check(req, self.state)

        # confirmation
        self._confirm(req)

        # dry-run = no adapter calls
        if self.dry_run or self.adapter.mode == "DRY_RUN":
            self._log(f"[DRY_RUN] order: {req}")
            return self._record_state(req, status="dry_run")

        # retry on transient errors
        attempt = 0
        while True:
            try:
                resp = self.adapter.place_order(req)
                return self._record_state(req, status="accepted", extra=resp)
            except TransientError as exc:
                attempt += 1
                self._log(f"[WARN] transient error: {exc} (attempt {attempt})")
                if attempt > self.max_retries:
                    raise
                time.sleep(self.retry_backoff_sec * attempt)

    def _record_state(self, req: OrderRequest, status: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "ts": now,
            "status": status,
            "order": asdict(req),
        }
        if extra:
            record["adapter_response"] = extra

        # update state
        self.state.record_order(req, record)
        self._log(f"[STATE] recorded order status={status} symbol={req.symbol} side={req.side} qty={req.qty}")
        return record
