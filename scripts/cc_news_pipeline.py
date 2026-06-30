#!/usr/bin/env python3
"""
CC-NEWS pipeline: stream WARC, filter on-the-fly, save to parquet with checkpoints.

Defaults:
  - output_dir: /mnt/data/cc-news
  - checkpoints: /mnt/data/cc-news/checkpoints
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from warcio.archiveiterator import ArchiveIterator


CC_BASE = "https://data.commoncrawl.org"


class TextExtractor(HTMLParser):
    def __init__(self, max_len: int = 5000) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._max_len = max_len

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if len("".join(self._parts)) >= self._max_len:
            return
        self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join([p for p in self._parts if p])


def _month_range(start: str, end: str) -> List[str]:
    s = dt.datetime.strptime(start, "%Y-%m")
    e = dt.datetime.strptime(end, "%Y-%m")
    months = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return months


def _fetch_warc_paths(month: str) -> List[str]:
    year, mon = month.split("-")
    url = f"{CC_BASE}/crawl-data/CC-NEWS/{year}/{mon}/warc.paths.gz"
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch warc.paths.gz for {month}: {resp.status_code}")
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        lines = gz.read().decode("utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _domain(url: str) -> str:
    try:
        dom = urlparse(url).netloc.lower()
        if dom.startswith("www."):
            dom = dom[4:]
        return dom
    except Exception:
        return ""


def _load_checkpoint(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    return done


def _append_checkpoint(path: str, warc_path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(warc_path + "\n")


def _save_parquet(rows: List[dict], out_dir: str, month: str, part: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    fname = f"ccnews_{month}_part{part:04d}.parquet"
    out_path = os.path.join(out_dir, fname)
    df.to_parquet(out_path, index=False)
    return out_path


def _next_part(out_dir: str, month: str) -> int:
    if not os.path.isdir(out_dir):
        return 1
    prefix = f"ccnews_{month}_part"
    max_part = 0
    for name in os.listdir(out_dir):
        if not name.startswith(prefix) or not name.endswith(".parquet"):
            continue
        m = re.search(r"part(\d{4})\.parquet$", name)
        if not m:
            continue
        try:
            max_part = max(max_part, int(m.group(1)))
        except ValueError:
            continue
    return max_part + 1


def _parse_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title[:300]


def _extract_text(html: str, max_len: int) -> str:
    parser = TextExtractor(max_len=max_len)
    parser.feed(html)
    return parser.text()


def _iter_warc_records(url: str, retries: int = 3, backoff: float = 10.0) -> Iterable[Tuple[str, str, str]]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=180)
            resp.raise_for_status()
            with gzip.GzipFile(fileobj=resp.raw) as gz:
                for rec in ArchiveIterator(gz):
                    if rec.rec_type != "response":
                        continue
                    target = rec.rec_headers.get_header("WARC-Target-URI") or ""
                    date = rec.rec_headers.get_header("WARC-Date") or ""
                    try:
                        payload = rec.content_stream().read()
                    except Exception:
                        continue
                    yield target, date, payload
            return
        except Exception as exc:
            last_err = exc
            print(f"[warn] warc fetch failed (attempt {attempt}/{retries}) {url} -> {exc}", flush=True)
            if attempt < retries:
                import time
                time.sleep(backoff)
    if last_err:
        raise last_err


def _compile_kw_patterns(keywords: List[str]) -> List[re.Pattern]:
    patterns = []
    for kw in keywords:
        if not kw:
            continue
        patterns.append(re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
    return patterns


def _compile_ticker_patterns(tickers: List[str]) -> List[re.Pattern]:
    patterns = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        patterns.append(re.compile(rf"(\$)?\b{re.escape(t)}\b", re.IGNORECASE))
    return patterns


def _compile_url_patterns(keywords: List[str]) -> List[re.Pattern]:
    patterns = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        patterns.append(re.compile(rf"(^|[\W_]){re.escape(kw)}([\W_]|$)", re.IGNORECASE))
    return patterns


def _has_kw(patterns: List[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/data/cc-news", help="Output directory for parquet")
    ap.add_argument("--checkpoint", default="/mnt/data/cc-news/checkpoints/processed.txt", help="Checkpoint file")
    ap.add_argument("--start-month", required=True, help="YYYY-MM")
    ap.add_argument("--end-month", required=True, help="YYYY-MM")
    ap.add_argument("--max-warc", type=int, default=0, help="Limit WARC files per month (0=no limit)")
    ap.add_argument("--warc-sample", default="first", choices=["first", "even", "random"],
                    help="How to sample WARC files when max-warc is set")
    ap.add_argument("--warc-seed", type=int, default=42, help="Seed for random WARC sampling")
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Max bytes read per record")
    ap.add_argument("--max-text", type=int, default=6000, help="Max extracted text length")
    ap.add_argument("--text-window", type=int, default=1000, help="Text window (prefix) for keyword matching")
    ap.add_argument("--min-body-chars", type=int, default=200, help="Minimum body length")
    ap.add_argument("--flush", type=int, default=1000, help="Rows per parquet chunk")
    ap.add_argument("--log-every", type=int, default=500, help="Print progress every N records")
    ap.add_argument("--keywords", default="bitcoin,btc,crypto,cryptocurrency,blockchain,ethereum",
                    help="Comma-separated keywords")
    ap.add_argument("--macro-keywords", default="etf,sec,regulation,macro,fomc,rate,interest,cpi,ppi,inflation,employment,jobs,treasury,bank,liquidity,credit,default,stress,volatility",
                    help="Comma-separated macro/market keywords (used for title gate)")
    ap.add_argument("--infra-keywords", default="binance,coinbase,kraken,bybit,okx,bitstamp,gemini,metamask,ledger,trezor,stablecoin,usdt,usdc,usde,usdd,bridge,layer2,l2,dex,amm,oracle,staking",
                    help="Comma-separated infrastructure keywords")
    ap.add_argument("--event-keywords", default="listing,delisting,futures,perpetual,airdrop,unlock,upgrade,hardfork,exploit,hack,outage,breach,sanction,investigation,indictment,settlement,funding round,raise,acquisition,merger",
                    help="Comma-separated event keywords")
    ap.add_argument("--url-keywords", default="crypto,bitcoin,btc,ethereum,eth,blockchain,web3,stablecoin,digital-asset",
                    help="Comma-separated URL path keywords")
    ap.add_argument("--score-min", type=int, default=2, help="Minimum score to keep record")
    ap.add_argument("--gold-score", type=int, default=3, help="Score threshold for gold tier")
    ap.add_argument("--silver-score", type=int, default=2, help="Score threshold for silver tier")
    ap.add_argument("--enable-silver", action="store_true", help="Keep silver tier in addition to gold")
    ap.add_argument("--require-anchor", action="store_true", help="Require crypto anchor in title or URL")
    ap.add_argument("--aux-keywords", default="etf,sec,regulation,macro,fomc,rate,interest,cpi,inflation,stablecoin,bank,treasury",
                    help="Comma-separated auxiliary keywords (optional signal)")
    ap.add_argument("--exclude-keywords", default="football,soccer,nba,nfl,mlb,nhl,tennis,cricket,goal.com,match,score,fixtures,weather,forecast,temperature,climate,crime,police,arrested,murder,shooting,accident,celebrity,entertainment,horoscope,astrology,gossip,lottery,gambling,coupon,betting,casino",
                    help="Comma-separated exclude keywords")
    ap.add_argument("--block-domains", default="goal.com,einpresswire.com,news.livedoor.com,infobae.com,mexc.com,mexc.fm,apolyton.net,kenyan-post.com",
                    help="Comma-separated domain blocklist")
    ap.add_argument("--domains", default="",
                    help="Comma-separated domain whitelist (optional)")
    ap.add_argument("--title-only", action="store_true", help="Match keywords only in title")
    ap.add_argument("--require-title", action="store_true", help="Require keyword hit in title")
    args = ap.parse_args()

    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    macro_keywords = [k.strip().lower() for k in args.macro_keywords.split(",") if k.strip()]
    infra_keywords = [k.strip().lower() for k in args.infra_keywords.split(",") if k.strip()]
    event_keywords = [k.strip().lower() for k in args.event_keywords.split(",") if k.strip()]
    url_keywords = [k.strip().lower() for k in args.url_keywords.split(",") if k.strip()]
    aux_keywords = [k.strip().lower() for k in args.aux_keywords.split(",") if k.strip()]
    exclude_keywords = [k.strip().lower() for k in args.exclude_keywords.split(",") if k.strip()]
    kw_patterns = _compile_kw_patterns(keywords)
    macro_patterns = _compile_kw_patterns(macro_keywords)
    infra_patterns = _compile_kw_patterns(infra_keywords)
    event_patterns = _compile_kw_patterns(event_keywords)
    url_patterns = _compile_url_patterns(url_keywords)
    aux_patterns = _compile_kw_patterns(aux_keywords)
    exclude_patterns = _compile_kw_patterns(exclude_keywords)
    ticker_patterns = _compile_ticker_patterns(
        ["BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "MATIC", "LTC", "BCH", "XLM", "XMR", "USDT", "USDC", "DAI"]
    )
    domains = {d.strip().lower() for d in args.domains.split(",") if d.strip()}
    block_domains = {d.strip().lower() for d in args.block_domains.split(",") if d.strip()}
    months = _month_range(args.start_month, args.end_month)

    done = _load_checkpoint(args.checkpoint)
    rows: List[dict] = []

    for month in months:
        part = _next_part(args.out_dir, month)
        print(f"[info] month={month} fetching warc.paths.gz", flush=True)
        warc_paths = _fetch_warc_paths(month)
        print(f"[info] month={month} warc_paths={len(warc_paths)}", flush=True)
        if args.max_warc > 0:
            if args.warc_sample == "first":
                warc_paths = warc_paths[: args.max_warc]
            elif args.warc_sample == "random":
                import random
                random.seed(args.warc_seed)
                warc_paths = random.sample(warc_paths, min(args.max_warc, len(warc_paths)))
            else:  # even
                n = min(args.max_warc, len(warc_paths))
                if n <= 1:
                    warc_paths = warc_paths[:n]
                else:
                    step = (len(warc_paths) - 1) / (n - 1)
                    idxs = [round(i * step) for i in range(n)]
                    warc_paths = [warc_paths[i] for i in idxs]
        for wp in warc_paths:
            if wp in done:
                continue
            url = f"{CC_BASE}/{wp}"
            seen = 0
            kept = 0
            print(f"[info] warc_start {wp}", flush=True)
            try:
                for target, date, payload in _iter_warc_records(url, retries=3, backoff=10.0):
                    if not target:
                        continue
                    seen += 1
                    dom = _domain(target)
                    if dom in block_domains:
                        continue
                    if domains and dom not in domains:
                        continue
                    payload = payload[: args.max_bytes]
                    try:
                        html = payload.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    title = _parse_title(html)
                    text = _extract_text(html, max_len=args.max_text)
                    if len(text) < args.min_body_chars:
                        continue
                    if args.title_only:
                        blob = title.lower()
                    else:
                        window = text[: args.text_window]
                        blob = f"{title} {window}".lower()
                    if exclude_patterns and _has_kw(exclude_patterns, blob):
                        continue
                    title_blob = title.lower()
                    primary_in_title = _has_kw(kw_patterns, title_blob) if kw_patterns else False
                    macro_in_title = _has_kw(macro_patterns, title_blob) if macro_patterns else False
                    if args.require_title and not (primary_in_title or macro_in_title):
                        continue
                    primary_body_hit = _has_kw(kw_patterns, blob) if kw_patterns else False
                    infra_hit = _has_kw(infra_patterns, blob) if infra_patterns else False
                    event_hit = _has_kw(event_patterns, blob) if event_patterns else False
                    url_hit = _has_kw(url_patterns, target.lower()) if url_patterns else False
                    ticker_hit = False  # avoid body tickers (too noisy)
                    ticker_in_title = _has_kw(ticker_patterns, title_blob)
                    score = 0
                    primary_hit = primary_in_title or url_hit
                    if primary_hit:
                        score += 2
                    if primary_in_title:
                        score += 2
                    if ticker_in_title and (primary_in_title or url_hit or infra_hit):
                        score += 1
                    if macro_in_title:
                        score += 1
                    if infra_hit:
                        score += 1
                    if event_hit:
                        score += 1
                    if url_hit:
                        score += 1
                    crypto_anchor = (
                        primary_in_title
                        or url_hit
                        or (primary_body_hit and (infra_hit or event_hit))
                    )
                    quality_tier = None
                    if score >= args.gold_score and (not args.require_anchor or crypto_anchor):
                        quality_tier = "gold"
                    elif args.enable_silver:
                        silver_anchor = primary_in_title or url_hit or primary_body_hit
                        if score >= args.silver_score and silver_anchor:
                            quality_tier = "silver"
                    if quality_tier is None:
                        continue
                    if score < args.score_min:
                        continue
                    if aux_patterns and _has_kw(aux_patterns, blob):
                        pass
                    # if aux keywords not present, still allow primary keyword matches
                    rows.append(
                        {
                            "url": target,
                            "domain": _domain(target),
                            "published_at": date,
                            "title": title,
                            "body": text,
                            "source": "cc-news",
                            "score": score,
                            "quality_tier": quality_tier,
                        }
                    )
                    kept += 1
                    if len(rows) >= args.flush:
                        out_path = _save_parquet(rows, args.out_dir, month, part)
                        print(f"[info] saved {out_path} rows={len(rows)}", flush=True)
                        rows = []
                        part += 1
                    if args.log_every > 0 and seen % args.log_every == 0:
                        print(f"[info] progress warc={wp} seen={seen} kept={kept}", flush=True)
            except Exception as exc:
                print(f"[warn] WARC failed: {wp} -> {exc}", flush=True)
            print(f"[info] warc_done {wp} seen={seen} kept={kept}", flush=True)
            _append_checkpoint(args.checkpoint, wp)
            done.add(wp)

        # End-of-month flush so each month gets its own part file
        if rows:
            out_path = _save_parquet(rows, args.out_dir, month, part)
            print(f"[info] saved {out_path} rows={len(rows)}", flush=True)
            rows = []
            part += 1

    # Final safety flush (should usually be empty due to per-month flush above)
    if rows:
        out_path = _save_parquet(rows, args.out_dir, months[-1], _next_part(args.out_dir, months[-1]))
        print(f"[info] saved {out_path} rows={len(rows)}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
