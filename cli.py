from __future__ import annotations

import argparse
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from executor import Executor, OrderRequest
from adapters.paper import PaperAdapter
from adapters.live_stub import LiveStubAdapter


def build_logger() -> logging.Logger:
    logger = logging.getLogger("executor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            "executor.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str)
    p.add_argument("--side", type=str)
    p.add_argument("--qty", type=float)
    p.add_argument("--type", dest="order_type", type=str, default="MARKET")
    p.add_argument("--limit-price", type=float, default=None)
    p.add_argument("--client-order-id", type=str, default=None)
    p.add_argument("--meta", type=str, default=None)
    p.add_argument("--order-json", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-confirm", action="store_true")
    p.add_argument("--adapter", choices=["paper", "live_stub"], default="paper")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = build_logger()

    if args.adapter == "paper":
        adapter = PaperAdapter(logger=logger)
    else:
        adapter = LiveStubAdapter(logger=logger)

    executor = Executor(
        adapter=adapter,
        logger=logger,
        dry_run=args.dry_run,
        require_confirm=args.require_confirm,
    )

    if args.order_json:
        data = json.loads(Path(args.order_json).read_text(encoding="utf-8"))
        req = OrderRequest(**data)
    else:
        if not args.symbol or not args.side or args.qty is None:
            raise SystemExit("symbol/side/qty are required unless --order-json is used")
        meta = json.loads(args.meta) if args.meta else {}
        req = OrderRequest(
            symbol=args.symbol,
            side=args.side,
            qty=args.qty,
            type=args.order_type,
            limit_price=args.limit_price,
            client_order_id=args.client_order_id,
            meta=meta,
        )

    result = executor.place_order(req)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
