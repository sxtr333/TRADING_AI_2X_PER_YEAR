#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.api.types import is_datetime64_any_dtype


def build_text(row: pd.Series) -> str:
    symbol = str(row.get("symbol", "") or "")
    direction = str(row.get("direction", "") or "")
    text = str(row.get("text", "") or "")
    return f"{symbol} {direction} {text}".strip()


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (y_prob >= t).astype(np.int32)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Binary signal-quality training with purged time split")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=30000)
    ap.add_argument("--seq-len", type=int, default=220)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--purge-days", type=int, default=14)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--lstm-units", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--dense-units", type=int, default=96)
    ap.add_argument("--num-units", type=int, default=24)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--ignore-sample-weight", action="store_true")
    ap.add_argument("--include-neutral-as-bad", action="store_true")
    ap.add_argument("--train-neg-pos-ratio", type=float, default=0.0)
    ap.add_argument("--vision-features", default="")
    ap.add_argument("--directional-features", default="")
    ap.add_argument("--crowd-priors", default="")
    ap.add_argument("--allow-leaky-directional", action="store_true")
    args = ap.parse_args()

    if args.val_frac <= 0 or args.test_frac <= 0 or (args.val_frac + args.test_frac) >= 0.6:
        raise ValueError("bad val/test fractions")

    df = pd.read_parquet(args.data).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df["timestamp_utc"].notna()].copy()

    if args.include_neutral_as_bad:
        df = df[df["quality_label"].isin(["good", "bad", "neutral"])].copy()
        df["y"] = (df["quality_label"] == "good").astype(np.int32)
    else:
        df = df[df["quality_label"].isin(["good", "bad"])].copy()
        df["y"] = (df["quality_label"] == "good").astype(np.int32)

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if len(df) < 1000:
        raise ValueError("too few rows after filtering")

    df["text_in"] = df.apply(build_text, axis=1)
    df["direction_long"] = (df["direction"].astype(str).str.lower() == "long").astype(np.float32)
    df["parse_confidence"] = pd.to_numeric(df.get("parse_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    df["text_len"] = df["text"].astype(str).str.len().astype(np.float32)
    df["is_candidate_f"] = df.get("is_candidate", False).astype(np.float32)
    df["year"] = df["timestamp_utc"].dt.year.astype(int)

    if args.vision_features:
        vf = pd.read_parquet(args.vision_features).copy()
        vf["message_id"] = vf["message_id"].astype(str)
        vf["timestamp_utc"] = vf["timestamp_utc"].astype(str)
        df["message_id"] = df["message_id"].astype(str)
        df["timestamp_utc_str"] = df["timestamp_utc"].astype(str)
        key_cols = ["message_id", "timestamp_utc"]

        # Keep only model-relevant columns from either legacy OCR schema
        # or new VLM strict_s3 schema.
        ocr_schema_cols = [
            "photo_exists",
            "img_w",
            "img_h",
            "img_aspect",
            "gray_mean",
            "gray_std",
            "edge_mean",
            "edge_std",
            "ocr_len",
            "ocr_digit_count",
            "ocr_letter_count",
            "ocr_price_count",
            "ocr_has_usdt",
            "ocr_has_long",
            "ocr_has_short",
            "ocr_has_tp",
            "ocr_has_sl",
        ]
        vlm_schema_cols = [
            "vlm_has_trade_setup",
            "vlm_has_drawn_levels",
            "vlm_has_arrow",
            "vlm_explicit_price_count",
            "vlm_explicit_price_min",
            "vlm_explicit_price_max",
            "vlm_risk_flag_count",
            "vlm_pattern_count",
            "vlm_confidence",
            "vlm_visual_bias",
            "vlm_error",
        ]
        feat_cols = [c for c in (ocr_schema_cols + vlm_schema_cols) if c in vf.columns]

        vf = vf.drop_duplicates(subset=key_cols, keep="first")
        df = df.merge(vf[key_cols + feat_cols], left_on=["message_id", "timestamp_utc_str"], right_on=key_cols, how="left")
        if "timestamp_utc_x" in df.columns:
            df = df.rename(columns={"timestamp_utc_x": "timestamp_utc"})
        if "timestamp_utc_y" in df.columns:
            df = df.drop(columns=["timestamp_utc_y"])
        if "timestamp_utc" in df.columns and not is_datetime64_any_dtype(df["timestamp_utc"]):
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        # Legacy OCR numeric cleanup.
        for c in [x for x in ocr_schema_cols if x in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

        # New VLM features -> numeric model inputs.
        if "vlm_has_trade_setup" in df.columns:
            df["vlm_has_trade_setup_f"] = pd.to_numeric(df["vlm_has_trade_setup"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_has_drawn_levels" in df.columns:
            df["vlm_has_drawn_levels_f"] = pd.to_numeric(df["vlm_has_drawn_levels"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_has_arrow" in df.columns:
            df["vlm_has_arrow_f"] = pd.to_numeric(df["vlm_has_arrow"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_explicit_price_count" in df.columns:
            df["vlm_explicit_price_count"] = pd.to_numeric(df["vlm_explicit_price_count"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_explicit_price_min" in df.columns:
            df["vlm_explicit_price_min"] = pd.to_numeric(df["vlm_explicit_price_min"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_explicit_price_max" in df.columns:
            df["vlm_explicit_price_max"] = pd.to_numeric(df["vlm_explicit_price_max"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_risk_flag_count" in df.columns:
            df["vlm_risk_flag_count"] = pd.to_numeric(df["vlm_risk_flag_count"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_pattern_count" in df.columns:
            df["vlm_pattern_count"] = pd.to_numeric(df["vlm_pattern_count"], errors="coerce").fillna(0.0).astype(np.float32)
        if "vlm_confidence" in df.columns:
            df["vlm_confidence"] = pd.to_numeric(df["vlm_confidence"], errors="coerce").fillna(0.0).astype(np.float32)

        if "vlm_visual_bias" in df.columns:
            vb = df["vlm_visual_bias"].fillna("unclear").astype(str).str.lower().str.strip()
            for cls in ["bullish", "bearish", "neutral", "unclear"]:
                df[f"vlm_bias_{cls}"] = (vb == cls).astype(np.float32)

        if "vlm_error" in df.columns:
            df["vlm_error_flag"] = df["vlm_error"].notna().astype(np.float32)

    if args.directional_features:
        dmf = pd.read_parquet(args.directional_features).copy()
        dmf["message_id"] = dmf["message_id"].astype(str)
        dmf["timestamp_utc"] = dmf["timestamp_utc"].astype(str)
        df["message_id"] = df["message_id"].astype(str)
        df["timestamp_utc_str"] = df["timestamp_utc"].astype(str)
        dkey = ["message_id", "timestamp_utc"]

        dcols_base = [
            "direction_move_status",
            "bars_to_anchor",
            "signed_ret_24h_pct",
            "max_favorable_24h_pct",
            "max_adverse_24h_pct",
            "is_profitable_24h",
            "error_reason_24h",
            "signed_ret_72h_pct",
            "max_favorable_72h_pct",
            "max_adverse_72h_pct",
            "is_profitable_72h",
            "error_reason_72h",
            "signed_ret_168h_pct",
            "max_favorable_168h_pct",
            "max_adverse_168h_pct",
            "is_profitable_168h",
            "error_reason_168h",
        ]
        dcols = [c for c in dcols_base if c in dmf.columns]
        dmf = dmf.drop_duplicates(subset=dkey, keep="first")
        df = df.merge(
            dmf[dkey + dcols],
            left_on=["message_id", "timestamp_utc_str"],
            right_on=dkey,
            how="left",
        )
        if "timestamp_utc_x" in df.columns:
            df = df.rename(columns={"timestamp_utc_x": "timestamp_utc"})
        if "timestamp_utc_y" in df.columns:
            df = df.drop(columns=["timestamp_utc_y"])
        if "timestamp_utc" in df.columns and not is_datetime64_any_dtype(df["timestamp_utc"]):
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    if args.crowd_priors:
        cpf = pd.read_parquet(args.crowd_priors).copy()
        cpf["message_id"] = cpf["message_id"].astype(str)
        cpf["timestamp_utc"] = cpf["timestamp_utc"].astype(str)
        df["message_id"] = df["message_id"].astype(str)
        df["timestamp_utc_str"] = df["timestamp_utc"].astype(str)
        ckey = ["message_id", "timestamp_utc"]
        crowd_cols = [c for c in cpf.columns if c not in ckey]
        cpf = cpf.drop_duplicates(subset=ckey, keep="first")
        df = df.merge(
            cpf[ckey + crowd_cols],
            left_on=["message_id", "timestamp_utc_str"],
            right_on=ckey,
            how="left",
        )
        if "timestamp_utc_x" in df.columns:
            df = df.rename(columns={"timestamp_utc_x": "timestamp_utc"})
        if "timestamp_utc_y" in df.columns:
            df = df.drop(columns=["timestamp_utc_y"])
        if "timestamp_utc" in df.columns and not is_datetime64_any_dtype(df["timestamp_utc"]):
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        # Numeric normalize for crowd priors.
        for c in crowd_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    # Directional numeric cleanup (works for merged input or when columns already in --data).
    directional_numeric_cols = [
        "bars_to_anchor",
        "signed_ret_24h_pct",
        "max_favorable_24h_pct",
        "max_adverse_24h_pct",
        "is_profitable_24h",
        "signed_ret_72h_pct",
        "max_favorable_72h_pct",
        "max_adverse_72h_pct",
        "is_profitable_72h",
        "signed_ret_168h_pct",
        "max_favorable_168h_pct",
        "max_adverse_168h_pct",
        "is_profitable_168h",
    ]
    for c in directional_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    if "direction_move_status" in df.columns:
        dms = df["direction_move_status"].fillna("unknown").astype(str).str.lower().str.strip()
        for cls in ["ok", "partial", "out_of_range", "invalid_direction", "invalid_time", "unknown"]:
            df[f"dms_{cls}"] = (dms == cls).astype(np.float32)

    for h in [24, 72, 168]:
        rc = f"error_reason_{h}h"
        if rc in df.columns:
            rr = df[rc].fillna("unknown").astype(str).str.lower().str.strip()
            for cls in [
                "good",
                "wrong_direction_or_deep_drawdown",
                "no_follow_through",
                "gave_back_profit",
                "insufficient_data",
                "unknown",
            ]:
                df[f"{rc}_{cls}"] = (rr == cls).astype(np.float32)

    n = len(df)
    val_start_idx = int(n * (1.0 - args.test_frac - args.val_frac))
    test_start_idx = int(n * (1.0 - args.test_frac))
    t_val_start = df.loc[val_start_idx, "timestamp_utc"]
    t_test_start = df.loc[test_start_idx, "timestamp_utc"]
    purge = pd.Timedelta(days=int(args.purge_days))

    train = df[df["timestamp_utc"] < (t_val_start - purge)].copy()
    val = df[(df["timestamp_utc"] >= (t_val_start + purge)) & (df["timestamp_utc"] < (t_test_start - purge))].copy()
    test = df[df["timestamp_utc"] >= (t_test_start + purge)].copy()
    if min(len(train), len(val), len(test)) < 100:
        raise ValueError("split too small after purge; lower purge-days or fractions")

    if float(args.train_neg_pos_ratio) > 0:
        pos = train[train["y"] == 1]
        neg = train[train["y"] == 0]
        max_neg = int(len(pos) * float(args.train_neg_pos_ratio))
        if len(pos) > 0 and len(neg) > max_neg:
            neg = neg.sample(n=max_neg, random_state=42)
            train = pd.concat([pos, neg], ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)

    xnum_cols = ["direction_long", "parse_confidence", "text_len", "is_candidate_f"]
    if args.vision_features:
        for c in [
            "photo_exists",
            "img_w",
            "img_h",
            "img_aspect",
            "gray_mean",
            "gray_std",
            "edge_mean",
            "edge_std",
            "ocr_len",
            "ocr_digit_count",
            "ocr_letter_count",
            "ocr_price_count",
            "ocr_has_usdt",
            "ocr_has_long",
            "ocr_has_short",
            "ocr_has_tp",
            "ocr_has_sl",
            "vlm_has_trade_setup_f",
            "vlm_has_drawn_levels_f",
            "vlm_has_arrow_f",
            "vlm_explicit_price_count",
            "vlm_explicit_price_min",
            "vlm_explicit_price_max",
            "vlm_risk_flag_count",
            "vlm_pattern_count",
            "vlm_confidence",
            "vlm_bias_bullish",
            "vlm_bias_bearish",
            "vlm_bias_neutral",
            "vlm_bias_unclear",
            "vlm_error_flag",
        ]:
            if c in df.columns:
                xnum_cols.append(c)

    if args.allow_leaky_directional:
        for c in [
            "bars_to_anchor",
            "signed_ret_24h_pct",
            "max_favorable_24h_pct",
            "max_adverse_24h_pct",
            "is_profitable_24h",
            "signed_ret_72h_pct",
            "max_favorable_72h_pct",
            "max_adverse_72h_pct",
            "is_profitable_72h",
            "signed_ret_168h_pct",
            "max_favorable_168h_pct",
            "max_adverse_168h_pct",
            "is_profitable_168h",
            "dms_ok",
            "dms_partial",
            "dms_out_of_range",
            "dms_invalid_direction",
            "dms_invalid_time",
            "dms_unknown",
            "error_reason_24h_good",
            "error_reason_24h_wrong_direction_or_deep_drawdown",
            "error_reason_24h_no_follow_through",
            "error_reason_24h_gave_back_profit",
            "error_reason_24h_insufficient_data",
            "error_reason_24h_unknown",
            "error_reason_72h_good",
            "error_reason_72h_wrong_direction_or_deep_drawdown",
            "error_reason_72h_no_follow_through",
            "error_reason_72h_gave_back_profit",
            "error_reason_72h_insufficient_data",
            "error_reason_72h_unknown",
            "error_reason_168h_good",
            "error_reason_168h_wrong_direction_or_deep_drawdown",
            "error_reason_168h_no_follow_through",
            "error_reason_168h_gave_back_profit",
            "error_reason_168h_insufficient_data",
            "error_reason_168h_unknown",
        ]:
            if c in df.columns:
                xnum_cols.append(c)

    for c in [
        "crowd_samples_prev_all",
        "crowd_samples_prev_long",
        "crowd_samples_prev_short",
        "crowd_h24_winrate_all",
        "crowd_h24_avg_signed_ret_all",
        "crowd_h24_err_wrong_all",
        "crowd_h24_err_nofollow_all",
        "crowd_h24_err_giveback_all",
        "crowd_h24_samples_dir",
        "crowd_h24_winrate_dir",
        "crowd_h24_avg_signed_ret_dir",
        "crowd_h24_err_wrong_dir",
        "crowd_h24_err_nofollow_dir",
        "crowd_h24_err_giveback_dir",
        "crowd_h24_dir_prior_ready",
        "crowd_h72_winrate_all",
        "crowd_h72_avg_signed_ret_all",
        "crowd_h72_err_wrong_all",
        "crowd_h72_err_nofollow_all",
        "crowd_h72_err_giveback_all",
        "crowd_h72_samples_dir",
        "crowd_h72_winrate_dir",
        "crowd_h72_avg_signed_ret_dir",
        "crowd_h72_err_wrong_dir",
        "crowd_h72_err_nofollow_dir",
        "crowd_h72_err_giveback_dir",
        "crowd_h72_dir_prior_ready",
        "crowd_h168_winrate_all",
        "crowd_h168_avg_signed_ret_all",
        "crowd_h168_err_wrong_all",
        "crowd_h168_err_nofollow_all",
        "crowd_h168_err_giveback_all",
        "crowd_h168_samples_dir",
        "crowd_h168_winrate_dir",
        "crowd_h168_avg_signed_ret_dir",
        "crowd_h168_err_wrong_dir",
        "crowd_h168_err_nofollow_dir",
        "crowd_h168_err_giveback_dir",
        "crowd_h168_dir_prior_ready",
    ]:
        if c in df.columns:
            xnum_cols.append(c)

    def make_weights(part: pd.DataFrame) -> np.ndarray:
        if (not args.ignore_sample_weight) and ("sample_weight" in part.columns):
            base = pd.to_numeric(part["sample_weight"], errors="coerce").fillna(1.0).to_numpy(np.float32)
        else:
            base = np.ones((len(part),), dtype=np.float32)
        # class balancing
        c0 = max(1, int((part["y"] == 0).sum()))
        c1 = max(1, int((part["y"] == 1).sum()))
        w0 = len(part) / (2.0 * c0)
        w1 = len(part) / (2.0 * c1)
        cw = np.where(part["y"].to_numpy() == 1, w1, w0).astype(np.float32)
        # year flattening
        yc = part["year"].value_counts().to_dict()
        yw = part["year"].map(lambda y: 1.0 / yc[int(y)]).to_numpy(np.float32)
        yw = yw / np.mean(yw)
        w = base * cw * yw
        w = w / np.mean(w)
        return w.astype(np.float32)

    x_text_train = train["text_in"].astype(str).to_numpy()
    x_text_val = val["text_in"].astype(str).to_numpy()
    x_text_test = test["text_in"].astype(str).to_numpy()
    x_num_train = train[xnum_cols].to_numpy(dtype=np.float32)
    x_num_val = val[xnum_cols].to_numpy(dtype=np.float32)
    x_num_test = test[xnum_cols].to_numpy(dtype=np.float32)
    y_train = train["y"].to_numpy(dtype=np.float32)
    y_val = val["y"].to_numpy(dtype=np.float32)
    y_test = test["y"].to_numpy(dtype=np.float32)
    w_train = make_weights(train)
    w_val = make_weights(val)

    text_vec = tf.keras.layers.TextVectorization(
        max_tokens=args.max_tokens,
        output_mode="int",
        output_sequence_length=args.seq_len,
        standardize="lower_and_strip_punctuation",
    )
    text_vec.adapt(tf.data.Dataset.from_tensor_slices(x_text_train).batch(1024))

    reg = tf.keras.regularizers.l2(float(args.l2))
    text_in = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    num_in = tf.keras.Input(shape=(len(xnum_cols),), dtype=tf.float32, name="num")

    x = text_vec(text_in)
    x = tf.keras.layers.Embedding(args.max_tokens, int(args.embed_dim))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(int(args.lstm_units), kernel_regularizer=reg, recurrent_regularizer=reg)
    )(x)
    x = tf.keras.layers.Dropout(float(args.dropout))(x)
    y = tf.keras.layers.Dense(int(args.num_units), activation="relu", kernel_regularizer=reg)(num_in)
    y = tf.keras.layers.BatchNormalization()(y)
    z = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(int(args.dense_units), activation="relu", kernel_regularizer=reg)(z)
    z = tf.keras.layers.Dropout(float(args.dropout))(z)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(z)

    model = tf.keras.Model(inputs={"text": text_in, "num": num_in}, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5),
    ]

    hist = model.fit(
        {"text": x_text_train, "num": x_num_train},
        y_train,
        sample_weight=w_train,
        validation_data=({"text": x_text_val, "num": x_num_val}, y_val, w_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=cb,
    )

    val_prob = model.predict({"text": x_text_val, "num": x_num_val}, verbose=0).reshape(-1)
    test_prob = model.predict({"text": x_text_test, "num": x_num_test}, verbose=0).reshape(-1)
    best_t, best_f1_val = best_f1_threshold(y_val.astype(np.int32), val_prob)
    yhat = (test_prob >= best_t).astype(np.int32)
    yt = y_test.astype(np.int32)
    tp = int(((yhat == 1) & (yt == 1)).sum())
    fp = int(((yhat == 1) & (yt == 0)).sum())
    tn = int(((yhat == 0) & (yt == 0)).sum())
    fn = int(((yhat == 0) & (yt == 1)).sum())
    acc = float((tp + tn) / len(yt))
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    summary = {
        "data": args.data,
        "rows_total_after_target_filter": int(len(df)),
        "target_positive_good": int((df["y"] == 1).sum()),
        "target_negative_bad_or_badplusneutral": int((df["y"] == 0).sum()),
        "split_rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "split_time": {
            "train_min": str(train["timestamp_utc"].min()),
            "train_max": str(train["timestamp_utc"].max()),
            "val_min": str(val["timestamp_utc"].min()),
            "val_max": str(val["timestamp_utc"].max()),
            "test_min": str(test["timestamp_utc"].min()),
            "test_max": str(test["timestamp_utc"].max()),
            "purge_days": int(args.purge_days),
        },
        "best_threshold_on_val_f1": best_t,
        "best_val_f1": best_f1_val,
        "test_at_best_threshold": {
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "history_last": {k: float(v[-1]) for k, v in hist.history.items()},
        "xnum_cols": xnum_cols,
        "vision_features": args.vision_features or None,
        "directional_features": args.directional_features or None,
        "crowd_priors": args.crowd_priors or None,
        "allow_leaky_directional": bool(args.allow_leaky_directional),
        "model_out": str(model_out),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
