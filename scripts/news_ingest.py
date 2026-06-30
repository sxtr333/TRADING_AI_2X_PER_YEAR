#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd
import requests


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _canon_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        query = [(k, v) for k, v in parse_qsl(parsed.query) if not k.lower().startswith("utm_")]
        new_query = urlencode(query, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)).lower()
    except Exception:
        return url.lower()


def _to_iso(ts: Optional[dt.datetime]) -> str:
    if ts is None:
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat()


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _normalize_item(
    source: str,
    source_id: str,
    title: str,
    url: str,
    published_at: Optional[dt.datetime],
    lang: str = "en",
    sentiment: Optional[float] = None,
    votes: Optional[int] = None,
    category: str = "news",
    impact: Optional[str] = None,
    currency: str = "BTC",
    raw: Optional[Dict] = None,
) -> Dict:
    canon = _canon_url(url)
    dedup_key = canon or f"{source}:{source_id}"
    return {
        "id": f"{source}:{source_id}",
        "source": source,
        "title": title or "",
        "url": url or "",
        "canonical_url": canon,
        "published_at": _to_iso(published_at),
        "ingested_at": _to_iso(_utc_now()),
        "lang": lang or "en",
        "sentiment": sentiment,
        "votes": votes,
        "category": category,
        "impact": impact,
        "currency": currency,
        "dedup_key": dedup_key,
        "raw_json": json.dumps(raw or {}, ensure_ascii=False),
    }


def _cryptopanic(token: str, currency: str, max_items: int) -> List[Dict]:
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": token, "currencies": currency, "public": "true"}
    items: List[Dict] = []
    while url and len(items) < max_items:
        resp = requests.get(url, params=params, headers={"accept": "application/json", "user-agent": "Mozilla/5.0"}, timeout=20)
        if resp.status_code == 404:
            raise ValueError("cryptopanic api returned 404 (likely API path changed or key/plan not supported)")
        resp.raise_for_status()
        data = resp.json()
        for row in data.get("results", []):
            published = None
            if row.get("published_at"):
                published = dt.datetime.fromisoformat(row["published_at"].replace("Z", "+00:00"))
            votes = row.get("votes", {}) or {}
            score = _safe_float(votes.get("positive"), 0.0) - _safe_float(votes.get("negative"), 0.0)
            items.append(
                _normalize_item(
                    source="cryptopanic",
                    source_id=str(row.get("id", "")),
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    published_at=published,
                    lang=row.get("language", "en"),
                    sentiment=score if votes else None,
                    votes=_safe_int(votes.get("important", 0)) + _safe_int(votes.get("liked", 0)),
                    category="news",
                    impact=row.get("kind"),
                    currency=currency,
                    raw=row,
                )
            )
            if len(items) >= max_items:
                break
        url = data.get("next")
        params = {}
    return items


def _cryptopanic_rss(max_items: int) -> List[Dict]:
    url = "https://cryptopanic.com/news/rss/"
    resp = requests.get(url, headers={"user-agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    from xml.etree import ElementTree as ET
    root = ET.fromstring(resp.text)
    items = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub = item.findtext("pubDate", default="")
        published = None
        if pub:
            try:
                published = dt.datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                published = None
        items.append(
            _normalize_item(
                source="cryptopanic_rss",
                source_id=link or title,
                title=title,
                url=link,
                published_at=published,
                lang="en",
                sentiment=None,
                votes=None,
                category="news",
                impact=None,
                currency="BTC",
                raw={"rss": True},
            )
        )
    return items


def _cryptocompare(api_key: str, currency: str, max_items: int) -> List[Dict]:
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    params = {"lang": "EN", "categories": currency}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    items = []
    for row in data.get("Data", [])[:max_items]:
        published = dt.datetime.fromtimestamp(int(row.get("published_on", 0)), tz=dt.timezone.utc)
        items.append(
            _normalize_item(
                source="cryptocompare",
                source_id=str(row.get("id", "")),
                title=row.get("title", ""),
                url=row.get("url", ""),
                published_at=published,
                lang=row.get("lang", "EN").lower(),
                sentiment=None,
                votes=None,
                category=row.get("categories", "news"),
                impact=None,
                currency=currency,
                raw=row,
            )
        )
    return items


def _binance_announcements(currency: str, max_items: int) -> List[Dict]:
    url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    payload = {
        "type": 1,
        "catalogId": 48,
        "pageNo": 1,
        "pageSize": max_items,
        "lang": "en",
    }
    headers = {
        "user-agent": "Mozilla/5.0",
        "accept": "application/json, text/plain, */*",
        "origin": "https://www.binance.com",
        "referer": "https://www.binance.com/en/support/announcement",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    items = []
    articles = data.get("data", {}).get("articles", []) or data.get("data", {}).get("articleList", [])
    for row in articles[:max_items]:
        ts = row.get("releaseDate") or row.get("publishTime")
        published = dt.datetime.fromtimestamp(int(ts) / 1000, tz=dt.timezone.utc) if ts else None
        items.append(
            _normalize_item(
                source="binance",
                source_id=str(row.get("id", "")),
                title=row.get("title", ""),
                url=f"https://www.binance.com/en/support/announcement/{row.get('code', '')}",
                published_at=published,
                lang="en",
                sentiment=None,
                votes=None,
                category="announcement",
                impact=row.get("category", ""),
                currency=currency,
                raw=row,
            )
        )
    return items


def _coinmarketcal(api_key: str, currency: str, max_items: int) -> List[Dict]:
    url = "https://developers.coinmarketcal.com/v1/events"
    headers = {"x-api-key": api_key}
    # coinmarketcal expects coin ids, not symbols (e.g., bitcoin)
    coin_id = None
    try:
        coins = requests.get("https://developers.coinmarketcal.com/v1/coins", headers=headers, timeout=20).json()
        for row in coins.get("body", []):
            if str(row.get("symbol", "")).upper() == currency.upper():
                coin_id = row.get("id")
                break
    except Exception:
        coin_id = None
    params = {"coins": coin_id or currency.lower(), "max": max_items}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("events", []) or data.get("data", [])
    items = []
    for row in events[:max_items]:
        ts = row.get("date_event") or row.get("date")
        published = None
        if ts:
            published = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        items.append(
            _normalize_item(
                source="coinmarketcal",
                source_id=str(row.get("id", "")),
                title=row.get("title", ""),
                url=row.get("link", "") or row.get("proof", ""),
                published_at=published,
                lang="en",
                sentiment=None,
                votes=_safe_int(row.get("votes", {}).get("positive", 0)),
                category="event",
                impact=row.get("impact"),
                currency=currency,
                raw=row,
            )
        )
    return items


def _load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.DataFrame()


def _save_df(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    elif out_path.endswith(".jsonl"):
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    else:
        raise SystemExit("Unsupported output format, use .parquet or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize crypto news into a single store.")
    parser.add_argument("--out", default="data/news/news.parquet", help="Output file (.parquet or .jsonl)")
    parser.add_argument("--currency", default="BTC", help="Main currency filter (BTC)")
    parser.add_argument("--max-items", type=int, default=200, help="Max items per source")
    parser.add_argument("--start", default=None, help="ISO start timestamp (UTC) for filtering")
    parser.add_argument("--end", default=None, help="ISO end timestamp (UTC) for filtering")
    args = parser.parse_args()

    items: List[Dict] = []
    currency = args.currency.upper()

    cryptopanic_key = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
    cryptocompare_key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
    coinmarketcal_key = os.getenv("COINMARKETCAL_API_KEY", "").strip()

    if cryptopanic_key:
        try:
            cp_items = _cryptopanic(cryptopanic_key, currency, args.max_items)
            items.extend(cp_items)
            print(f"[cryptopanic] items={len(cp_items)}")
        except Exception as exc:
            print(f"[cryptopanic] failed: {exc}")
            try:
                rss_items = _cryptopanic_rss(args.max_items)
                items.extend(rss_items)
                print(f"[cryptopanic_rss] items={len(rss_items)}")
            except Exception as rss_exc:
                print(f"[cryptopanic_rss] failed: {rss_exc}")
    else:
        print("[cryptopanic] skipped (no CRYPTOPANIC_API_KEY)")

    try:
        cc_items = _cryptocompare(cryptocompare_key, currency, args.max_items)
        items.extend(cc_items)
        print(f"[cryptocompare] items={len(cc_items)}")
    except Exception as exc:
        print(f"[cryptocompare] failed: {exc}")

    try:
        b_items = _binance_announcements(currency, args.max_items)
        items.extend(b_items)
        print(f"[binance] items={len(b_items)}")
    except Exception as exc:
        print(f"[binance] failed: {exc}")

    if coinmarketcal_key:
        try:
            cmc_items = _coinmarketcal(coinmarketcal_key, currency, args.max_items)
            items.extend(cmc_items)
            print(f"[coinmarketcal] items={len(cmc_items)}")
        except Exception as exc:
            print(f"[coinmarketcal] failed: {exc}")
    else:
        print("[coinmarketcal] skipped (no COINMARKETCAL_API_KEY)")

    if not items:
        print("No items fetched.")
        return

    start_ts = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")) if args.start else None
    end_ts = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00")) if args.end else None

    def within_range(published_iso: str) -> bool:
        if not published_iso:
            return False
        try:
            ts = dt.datetime.fromisoformat(published_iso)
        except Exception:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        if start_ts and ts < start_ts:
            return False
        if end_ts and ts > end_ts:
            return False
        return True

    df_new = pd.DataFrame(items)
    if start_ts or end_ts:
        df_new = df_new[df_new["published_at"].apply(within_range)]
    df_old = _load_existing(args.out)
    df = pd.concat([df_old, df_new], ignore_index=True)
    if "dedup_key" in df.columns:
        df = df.sort_values("published_at", ascending=False).drop_duplicates("dedup_key", keep="first")
    df = df.sort_values("published_at", ascending=False)
    _save_df(df, args.out)
    print(f"Saved {len(df)} news items to {args.out}")


if __name__ == "__main__":
    main()
