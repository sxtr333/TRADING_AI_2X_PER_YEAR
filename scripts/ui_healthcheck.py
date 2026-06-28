#!/usr/bin/env python3
import json
import sys
import time
import urllib.request
from urllib.parse import urlencode

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

def get(path, params=None, timeout=15):
    url = BASE + path
    if params:
        url += "?" + urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "healthcheck"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8")
        return json.loads(data)


def main():
    out = {"base": BASE, "ok": True, "checks": {}}
    try:
        # candles
        candles = get("/candles", {"limit": 2})
        out["checks"]["candles"] = {
            "ok": bool(candles.get("candles")),
            "last": candles.get("candles", [])[-1] if candles.get("candles") else None,
        }
        # forecast h20
        forecast = get("/forecast", {"interval": "h20"})
        out["checks"]["forecast_h20"] = {
            "ok": bool(forecast.get("points")),
            "base_time": forecast.get("base_time"),
            "base_price": forecast.get("base_price"),
        }
        # forecast h80/h160
        f80 = get("/forecast_multi", {"interval": "h80"})
        out["checks"]["forecast_h80"] = {"ok": bool(f80.get("points"))}
        f160 = get("/forecast_multi", {"interval": "h160"})
        out["checks"]["forecast_h160"] = {"ok": bool(f160.get("points"))}
        # trades live
        trades_live = get("/trades", {"file": "trades_live.csv"})
        out["checks"]["trades_live"] = {
            "ok": bool(trades_live.get("trades")),
            "count": len(trades_live.get("trades", [])),
        }
        # trades fallback
        trades_hist = get("/trades", {"file": "trades_meta_best_2025.csv"})
        out["checks"]["trades_hist"] = {
            "ok": bool(trades_hist.get("trades")),
            "count": len(trades_hist.get("trades", [])),
        }
        # news
        news = get("/news", {"limit": 5})
        out["checks"]["news"] = {
            "ok": bool(news.get("items")),
            "count": len(news.get("items", [])),
            "top_title": (news.get("items") or [{}])[0].get("title"),
        }
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
