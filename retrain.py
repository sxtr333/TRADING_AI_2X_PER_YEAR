"""
Simple retrain/fine-tune pipeline:
- Downloads/uses existing data (assumes bybit_public_trades.py already run)
- Builds features (optionally with aux/spot)
- Splits by time and fine-tunes the model on the latest window
Designed to be cron-friendly; customize paths/dates as needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import pandas as pd

from build_features import build_features
from trading_keras_core import default_feature_list
from bybit_public_trades import download_timeframes
from bybit_aux import main as aux_main


def fine_tune(
    features_path: Path,
    model_out: Path,
    seq_len: int = 256,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    warmup_steps: int = 200,
    cosine: bool = True,
):
    from train_keras import main as train_main
    # Reuse train_keras via CLI call to keep behavior consistent
    args = [
        "python",
        "train_keras.py",
        "--features",
        str(features_path),
        "--seq-len",
        str(seq_len),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--warmup-steps",
        str(warmup_steps),
        "--model-out",
        str(model_out),
    ]
    if cosine:
        args.append("--cosine")
    subprocess.run(args, check=True)


def slice_recent(input_path: Path, days: int) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if df.empty:
        return df
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on recent data.")
    parser.add_argument("--input", required=True, help="Base OHLCV parquet (e.g., data/BTCUSDT_1m.parquet)")
    parser.add_argument("--aux", help="Aux file for OI/funding/liq/etc")
    parser.add_argument("--spot", help="Spot file for basis")
    parser.add_argument("--output-features", default="data/BTCUSDT_1m_features_ft.parquet")
    parser.add_argument("--model-out", default="model_finetuned.keras")
    parser.add_argument("--days", type=int, default=30, help="Use last N days for fine-tune")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--download", action="store_true", help="If set, download latest kline data before slicing (requires net).")
    parser.add_argument("--start", help="Start date for download (ISO)")
    parser.add_argument("--end", help="End date for download (ISO)")
    args = parser.parse_args()

    if args.download:
        if not args.start or not args.end:
            raise ValueError("Provide --start and --end for download range.")
        download_timeframes(
            symbol="BTCUSDT",
            timeframes=["1m"],
            start_ms=int(pd.to_datetime(args.start, utc=True).timestamp() * 1000),
            end_ms=int(pd.to_datetime(args.end, utc=True).timestamp() * 1000),
            out_dir=Path("data"),
        )

    # Slice recent input
    recent_df = slice_recent(Path(args.input), args.days)
    if recent_df.empty:
        raise ValueError("No data for the specified window.")
    temp_path = Path("data/_recent.parquet")
    recent_df.to_parquet(temp_path, index=False)

    build_features(
        temp_path,
        Path(args.output_features),
        aux_path=Path(args.aux) if args.aux else None,
        spot_path=Path(args.spot) if args.spot else None,
        horizon=1,
        target_mode="log_return",
    )

    fine_tune(
        Path(args.output_features),
        Path(args.model_out),
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        cosine=True,
    )


if __name__ == "__main__":
    main()
