#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Callable, Dict, Any

from fastapi.routing import APIRoute

from serve_fastapi import create_app


def get_route(app, path: str, method: str) -> Callable[..., Dict[str, Any]]:
    for route in app.router.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route.endpoint
    raise RuntimeError(f"Route not found: {method} {path}")


def check_timegrid(points: list[dict], step_sec: int) -> None:
    if len(points) < 2:
        return
    times = [int(p["time"]) for p in points]
    deltas = [b - a for a, b in zip(times, times[1:])]
    if any(d != step_sec for d in deltas):
        raise AssertionError(f"timegrid step mismatch: expected {step_sec}, got {sorted(set(deltas))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test for forecast endpoints.")
    ap.add_argument("--features", required=True, help="Features parquet for forecast endpoints")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--model-h20", default="model_battle_itransformer.keras")
    ap.add_argument("--stats-h20", default="norm_stats_battle_itransformer.npz")
    ap.add_argument("--model-multi", default="model_15m_itransformer_tb_multi.keras")
    ap.add_argument("--stats-multi", default="norm_stats_15m_itransformer_tb_multi.npz")
    args = ap.parse_args()

    app = create_app(
        model_h20_path=args.model_h20,
        seq_len=args.seq_len,
        stats_h20_path=args.stats_h20,
        model_multi_path=args.model_multi,
        stats_multi_path=args.stats_multi,
        features_path=args.features,
    )

    forecast = get_route(app, "/forecast", "GET")
    forecast_multi = get_route(app, "/forecast_multi", "GET")

    res_h20 = forecast(interval="h20", hours=5)
    if not res_h20["points"]:
        raise AssertionError("forecast h20 returned empty points")
    if res_h20["step_min"] != 15:
        raise AssertionError(f"forecast h20 step_min mismatch: {res_h20['step_min']}")
    check_timegrid(res_h20["points"], 15 * 60)

    res_h80 = forecast_multi(interval="h80")
    if not res_h80["points"]:
        raise AssertionError("forecast_multi h80 returned empty points")
    if res_h80["step_min"] != 15:
        raise AssertionError(f"forecast_multi h80 step_min mismatch: {res_h80['step_min']}")
    check_timegrid(res_h80["points"], 15 * 60)

    print("OK: forecast endpoints respond with 15m timegrid.")


if __name__ == "__main__":
    main()
