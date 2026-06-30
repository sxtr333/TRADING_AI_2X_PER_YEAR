#!/usr/bin/env python3
"""
Compute news sentiment using an ensemble of:
  - LedgerBERT (crypto market sentiment, EN)
  - XLM-R (multilingual sentiment)

Outputs a parquet with columns:
  url, published_at, title, sentiment_ledger, sentiment_xlmr, sentiment
"""
from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _load_news(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {input_path}")
        frames = [pd.read_parquet(f) for f in files]
        news = pd.concat(frames, ignore_index=True)
    else:
        news = pd.read_parquet(input_path)
    # normalize required columns
    for col in ("url", "title", "body", "published_at"):
        if col not in news.columns:
            news[col] = None
    return news


def _text_for_row(row: pd.Series, max_chars: int) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()
    if title and body:
        text = f"{title}. {body}"
    else:
        text = title or body
    return text[:max_chars]


def _is_english(text: str) -> bool:
    if not text:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if alpha == 0:
        return False
    latin = sum("a" <= ch.lower() <= "z" for ch in text)
    return (latin / alpha) >= 0.6


def _label_to_sentiment(label: str, score: float) -> float:
    l = label.lower()
    if "pos" in l or "bull" in l:
        return float(score)
    if "neg" in l or "bear" in l:
        return -float(score)
    if "neutral" in l:
        return 0.0
    # fallback
    return 0.0


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _resolve_device(device: str) -> int:
    if device.lower() == "cpu":
        return -1
    if device.lower() == "cuda":
        return 0
    try:
        return int(device)
    except ValueError:
        return -1


def _write_cache(cache: Dict[str, Tuple[float, float, float]], cache_path: Path) -> None:
    rows = []
    for url, (s_l, s_x, s) in cache.items():
        rows.append(
            {
                "url": url,
                "sentiment_ledger": s_l,
                "sentiment_xlmr": s_x,
                "sentiment": s,
            }
        )
    pd.DataFrame(rows).to_parquet(cache_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet file or directory with parquet news")
    ap.add_argument("--output", required=True, help="Output parquet path")
    ap.add_argument("--cache", default="", help="Optional cache parquet (by url)")
    ap.add_argument("--max-chars", type=int, default=2000, help="Max chars per news")
    ap.add_argument("--max-length", type=int, default=512, help="Max token length for transformers")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    ap.add_argument("--device", default="cuda", help="cuda|cpu|index")
    ap.add_argument("--save-every", type=int, default=1000, help="Checkpoint cache every N rows")
    ap.add_argument("--model-ledger", default="ExponentialScience/LedgerBERT-Market-Sentiment")
    ap.add_argument("--model-ledger-revision", default="", help="Optional HF revision for LedgerBERT")
    ap.add_argument("--model-xlm", default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    ap.add_argument("--model-xlm-revision", default="", help="Optional HF revision for XLM-R sentiment")
    ap.add_argument("--require-xlmr", action="store_true", help="Fail if XLM-R cannot be loaded")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache) if args.cache else None
    revision_ledger = args.model_ledger_revision or None
    revision_xlm = args.model_xlm_revision or None

    news = _load_news(input_path)
    news = news.dropna(subset=["url"]).drop_duplicates(subset=["url"])
    news["text"] = news.apply(lambda r: _text_for_row(r, args.max_chars), axis=1)

    cache: Dict[str, Tuple[float, float, float]] = {}
    if cache_path and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        for _, row in cached.iterrows():
            cache[str(row["url"])] = (
                float(row.get("sentiment_ledger", 0.0)),
                float(row.get("sentiment_xlmr", 0.0)),
                float(row.get("sentiment", 0.0)),
            )

    to_score = news[~news["url"].isin(cache.keys())].copy()

    if not to_score.empty:
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        # Allow online model download by default; can be forced offline with NEWS_HF_OFFLINE=1.
        if "HF_HUB_OFFLINE" not in os.environ:
            os.environ["HF_HUB_OFFLINE"] = "1" if os.getenv("NEWS_HF_OFFLINE", "0") == "1" else "0"
        from transformers import pipeline

        device = _resolve_device(args.device)
        # local_files_only is passed to pipeline directly to avoid duplicate args
        model_kwargs_ledger = {"use_safetensors": True}
        # Cardiff XLM-R checkpoints are often published as pytorch_model.bin only.
        model_kwargs_xlm = {"use_safetensors": False}
        xlm_path = Path(args.model_xlm)
        if xlm_path.exists() and xlm_path.is_dir():
            if not (xlm_path / "model.safetensors").exists():
                model_kwargs_xlm["use_safetensors"] = False
        pipe_ledger = pipeline(
            "text-classification",
            model=args.model_ledger,
            device=device,
            truncation=True,
            framework="pt",
            model_kwargs=model_kwargs_ledger,
            revision=revision_ledger,
        )
        pipe_xlmr = None
        try:
            pipe_xlmr = pipeline(
                "text-classification",
                model=args.model_xlm,
                device=device,
                truncation=True,
                framework="pt",
                model_kwargs=model_kwargs_xlm,
                revision=revision_xlm,
            )
        except Exception as exc:
            print(f"[warn] XLM-R pipeline unavailable, falling back to LedgerBERT only: {exc}")
            if args.require_xlmr:
                raise RuntimeError("XLM-R required but failed to load") from exc

        # process batches
        sentiments_ledger: List[float] = []
        sentiments_xlmr: List[float] = []
        sentiments_ens: List[float] = []

        texts = to_score["text"].tolist()
        is_en = [ _is_english(t) for t in texts ]

        processed = 0
        next_save = args.save_every if cache_path else None
        for batch_idx, batch in enumerate(_batched(texts, args.batch_size)):
            batch_start = batch_idx * args.batch_size
            batch_en = [t for t, e in zip(batch, is_en[batch_idx * args.batch_size : batch_idx * args.batch_size + len(batch)]) if e]
            batch_all = batch

            # XLM-R for all (if available)
            out_xlmr = pipe_xlmr(batch_all, truncation=True, max_length=args.max_length) if pipe_xlmr else []
            # LedgerBERT only for EN subset
            out_ledger = {}
            if batch_en:
                out_ledger_list = pipe_ledger(batch_en, truncation=True, max_length=args.max_length)
                j = 0
                for i, t in enumerate(batch_all):
                    if _is_english(t):
                        out_ledger[i] = out_ledger_list[j]
                        j += 1

            batch_ledger: List[float] = []
            batch_xlmr: List[float] = []
            batch_ens: List[float] = []
            for i, t in enumerate(batch_all):
                if pipe_xlmr:
                    r_x = out_xlmr[i]
                    s_x = _label_to_sentiment(r_x["label"], r_x["score"])
                else:
                    s_x = 0.0
                s_l = 0.0
                if i in out_ledger:
                    r_l = out_ledger[i]
                    s_l = _label_to_sentiment(r_l["label"], r_l["score"])
                # ensemble: if EN, blend; else XLM-R only
                if pipe_xlmr:
                    if _is_english(t):
                        s = 0.65 * s_l + 0.35 * s_x
                    else:
                        s = s_x
                else:
                    s = s_l
                sentiments_ledger.append(s_l)
                sentiments_xlmr.append(s_x)
                sentiments_ens.append(s)
                batch_ledger.append(s_l)
                batch_xlmr.append(s_x)
                batch_ens.append(s)

            processed += len(batch_all)
            if cache_path:
                for j in range(len(batch_all)):
                    url = str(to_score["url"].iloc[batch_start + j])
                    cache[url] = (
                        float(batch_ledger[j]),
                        float(batch_xlmr[j]),
                        float(batch_ens[j]),
                    )
            if cache_path and next_save is not None and processed >= next_save:
                _write_cache(cache, cache_path)
                next_save += args.save_every

        to_score["sentiment_ledger"] = sentiments_ledger
        to_score["sentiment_xlmr"] = sentiments_xlmr
        to_score["sentiment"] = sentiments_ens

        for _, row in to_score.iterrows():
            cache[str(row["url"])] = (
                float(row["sentiment_ledger"]),
                float(row["sentiment_xlmr"]),
                float(row["sentiment"]),
            )
        if cache_path:
            _write_cache(cache, cache_path)

    # build output from cache (preserve original order)
    out_rows = []
    for _, row in news.iterrows():
        url = str(row["url"])
        s_l, s_x, s = cache.get(url, (0.0, 0.0, 0.0))
        out_rows.append(
            {
                "url": url,
                "published_at": row.get("published_at"),
                "title": row.get("title"),
                "sentiment_ledger": s_l,
                "sentiment_xlmr": s_x,
                "sentiment": s,
            }
        )

    out_df = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    if cache_path:
        cache_df = out_df.copy()
        cache_df.to_parquet(cache_path, index=False)


if __name__ == "__main__":
    main()
