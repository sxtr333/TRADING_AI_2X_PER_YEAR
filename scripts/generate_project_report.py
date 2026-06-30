#!/usr/bin/env python3
import csv
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/vitamind/my_project/model6')
OUT_TXT = ROOT / 'PROJECT_REPORT.txt'
OUT_PDF = ROOT / 'PROJECT_REPORT.pdf'


def read_file(path, max_chars=None):
    p = Path(path)
    if not p.exists():
        return ''
    data = p.read_text(errors='ignore')
    if max_chars:
        return data[:max_chars]
    return data


def read_pdf_text(path: Path, max_chars=4000):
    import subprocess
    if not path.exists():
        return ''
    try:
        out = subprocess.check_output(['pdftotext', str(path), '-'], text=True)
    except Exception:
        return ''
    return out[:max_chars]


def list_dirs(base: Path):
    return sorted([p for p in base.iterdir() if p.is_dir()])


def file_count(p: Path):
    if not p.exists():
        return 0
    return sum(1 for _ in p.rglob('*') if _.is_file())


def list_files(p: Path, max_items=40):
    if not p.exists():
        return []
    items = sorted([f.name for f in p.iterdir() if f.is_file()])
    return items[:max_items]


def list_subdirs(p: Path, max_items=40):
    if not p.exists():
        return []
    items = sorted([f.name for f in p.iterdir() if f.is_dir()])
    return items[:max_items]


def wf_metrics(path: Path):
    if not path.exists():
        return None
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def read_log_snippet(path: Path, keywords, max_lines=40):
    if not path.exists():
        return ''
    lines = path.read_text(errors='ignore').splitlines()
    hits = []
    for line in lines:
        if any(k in line for k in keywords):
            hits.append(line)
    return '\n'.join(hits[:max_lines])

def parse_eval_rows(path: Path):
    if not path.exists():
        return []
    lines = path.read_text(errors='ignore').splitlines()
    rows = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        if not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        # Heuristic: horizon + n + mae + rmse + mape + smape + dir_acc
        rows.append(s)
    return rows


def wrap(text, width=110):
    return '\n'.join(textwrap.fill(line, width=width, replace_whitespace=False) for line in text.splitlines())


def build_report():
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')

    project_context = read_file(ROOT / 'PROJECT_CONTEXT.md')
    architecture = read_file(ROOT / 'ARCHITECTURE.md')
    config_ref = read_file(ROOT / 'CONFIG_REFERENCE.md')
    code_index = read_file(ROOT / 'CODE_INDEX.json', max_chars=6000)
    chat_summary = read_pdf_text(ROOT / 'chat_summary.pdf', max_chars=4000)

    wf_tb = wf_metrics(ROOT / 'reports/wf_tb.csv')
    wf_battle = wf_metrics(ROOT / 'reports/wf_battle.csv')
    wf_moderate = wf_metrics(ROOT / 'reports/wf_itransformer_moderate_plus.csv')

    new_models_dir = ROOT / 'new_models'
    new_models = list_subdirs(new_models_dir)

    dirs = list_dirs(ROOT)

    fastapi_err = read_log_snippet(ROOT / 'logs/fastapi.log', ['Traceback', 'TypeError', 'PydanticDeprecatedSince20'])
    live_update_err = read_log_snippet(ROOT / 'logs/live_update.log', ['Traceback', 'failed', 'FileNotFoundError', '403', '404', 'Unsupported output format'])
    cloudflared_err = read_log_snippet(ROOT / 'logs/cloudflared.log', ['ERR', 'error', 'Unauthorized', 'Cannot determine', 'ping_group_range'])

    eval_logs = {
        'v2 (rolling-news, docker)': ROOT / 'logs/eval_news_xlmr_v2_docker.log',
        'v3 (rolling-news, line metrics)': ROOT / 'logs/eval_news_xlmr_v3_line.log',
        'v4 price-weight (h20 line)': ROOT / 'logs/eval_news_xlmr_v4_pw_h20_line.log',
        'v4 price-weight (h80 line)': ROOT / 'logs/eval_news_xlmr_v4_pw_h80_line.log',
        'v4 price-weight (h160 line)': ROOT / 'logs/eval_news_xlmr_v4_pw_h160_line.log',
        'v5 dR + per-head scale (line)': ROOT / 'logs/eval_news_xlmr_v5_dr_scale_line.log',
        'v6 dR + per-head scale + 8 nodes (training log only)': ROOT / 'logs/train_news_xlmr_v6_dr_scale_8nodes.log',
    }

    report = []
    report.append('MODEL6 PROJECT DEEP REPORT')
    report.append(f'Generated (UTC): {now}')
    report.append('')
    report.append('TL;DR FOR NEW CODEX (EXEC SUMMARY)')
    report.append('')
    report.append('- Project: BTCUSDT 15m ML forecasting + FastAPI + web UI in html/aladin_from_image.html.')
    report.append('- Core horizon: h20 (≈5h). Multi-horizon: h80/h160 (≈20h/40h).')
    report.append('- Production models (older baseline): model_battle_itransformer.keras + norm_stats_battle_itransformer.npz (h20)')
    report.append('  and model_15m_itransformer_tb_multi.keras + norm_stats_15m_itransformer_tb_multi.npz (multi).')
    report.append('- Latest experiments: news-based models in new_models/2026-01-13..2026-01-14.')
    report.append('- Best logged news model so far: v5 (dR + per-head scale) in eval_news_xlmr_v5_dr_scale_line.log.')
    report.append('- v6 (dR + per-head scale + 8 nodes) trained, eval blocked by Keras3 loader issues in eval_price_quality.py.')
    report.append('- UI: Binance klines for candles; buttons 15m/1h/4h/1d; PREDICT -> horizon -> ПРОГНОЗ.')
    report.append('- Backend: serve_fastapi.py exposes /forecast, /forecast_multi, /news; expects 15m grid.')
    report.append('- Live update pipeline: scripts/live_update_once.sh (news->sentiment->OHLCV->features).')
    report.append('- One command to run site: scripts/run_servers.sh (or run_live_site.sh for auto-updates).')
    report.append('- Common errors: Keras3/legacy mismatch; feature/stats length mismatch; Cloudflared tunnel 1033.')
    report.append('')
    report.append('Quick Start (copy-paste):')
    report.append('  cd /home/vitamind/my_project/model6')
    report.append('  source .venv/bin/activate')
    report.append('  bash scripts/run_servers.sh')
    report.append('  # open http://localhost:8080/aladin_from_image.html')
    report.append('  # API health: curl http://localhost:8000/health')
    report.append('')

    report.append('1) PROJECT OVERVIEW')
    report.append('')
    report.append(wrap(project_context.strip()))
    report.append('')

    report.append('2) ARCHITECTURE OVERVIEW')
    report.append('')
    report.append(wrap(architecture.strip()))
    report.append('')

    report.append('3) CONFIG / CLI REFERENCE (KEY PARAMETERS)')
    report.append('')
    report.append(wrap(config_ref.strip()))
    report.append('')

    report.append('4) CODE INDEX (SUMMARY OF KEY MODULES)')
    report.append('')
    report.append(wrap(code_index.strip()))
    report.append('')

    report.append('5) CHAT_SUMMARY.PDF (UI RULES AND HOTSPOTS)')
    report.append('')
    report.append(wrap(chat_summary.strip()))
    report.append('')

    report.append('6) DIRECTORY MAP (FOLDERS AND PURPOSES)')
    report.append('')
    for d in dirs:
        rel = d.relative_to(ROOT)
        count = file_count(d)
        report.append(f'- {rel}/  (files: {count})')
    report.append('')

    report.append('6.1) FOLDER DETAILS (SELECTED CONTENTS)')
    report.append('')
    report.append('data/:')
    report.append('  files: ' + ', '.join(list_files(ROOT / 'data', max_items=25)))
    report.append('  subdirs: ' + ', '.join(list_subdirs(ROOT / 'data', max_items=25)))
    report.append('')
    report.append('html/:')
    report.append('  files: ' + ', '.join(list_files(ROOT / 'html', max_items=40)))
    report.append('  subdirs: ' + ', '.join(list_subdirs(ROOT / 'html', max_items=40)))
    report.append('')
    report.append('scripts/:')
    report.append('  files: ' + ', '.join(list_files(ROOT / 'scripts', max_items=80)))
    report.append('')
    report.append('reports/:')
    report.append('  files: ' + ', '.join(list_files(ROOT / 'reports', max_items=80)))
    report.append('')
    report.append('new_models/:')
    report.append('  experiment folders: ' + ', '.join(new_models))
    report.append('')

    report.append('7) KEY FILES AND WHAT THEY DO (SELECTED)')
    report.append('')
    report.append('- build_features.py: core feature engineering (OHLCV + aux + TA + liq + news).')
    report.append('- train_keras.py: training runner (itransformer/patchtst/tsmixer; cls+price heads; losses).')
    report.append('- walk_forward_eval.py: walk-forward trading evaluation with fee/slippage/risk controls.')
    report.append('- serve_fastapi.py: FastAPI server; endpoints /forecast /forecast_multi /news /news_agg.')
    report.append('- html/aladin_from_image.html: UI (Lightweight Charts), forecast buttons, news panel.')
    report.append('- scripts/run_servers.sh: start FastAPI + UI server; selects latest features and models.')
    report.append('- scripts/run_live_site.sh: starts periodic live updates + run_servers.sh.')
    report.append('- scripts/live_update_once.sh: pulls news + sentiment, updates OHLCV, rebuilds features.')
    report.append('')

    report.append('8) FRONTEND DETAILS (html/aladin_from_image.html)')
    report.append('')
    report.append('Key behaviors extracted from JS:')
    report.append('- Uses Binance REST klines for candles (interval buttons 15m/1h/4h/1d).')
    report.append('- API_BASE is http://localhost:8000 (FastAPI).')
    report.append('- Forecast endpoints: /forecast?interval=h20 and /forecast_multi?interval=h80/h160.')
    report.append('- UI flow: PREDICT button must be pressed → horizon button → ПРОГНОЗ.')
    report.append('- Forecast alignment: shifts time to last candle and scales price to last close if provided by backend.')
    report.append('- Zigzag render-test: if line is flat or render_test=1, creates synthetic zigzag line for debugging.')
    report.append('- Sessions overlay: vertical dashed lines for China/Europe/New York at UTC hours.')
    report.append('- News panel: fetches /news?limit=30 and renders list items.')
    report.append('')

    report.append('9) BACKEND DETAILS (serve_fastapi.py)')
    report.append('')
    report.append('Endpoints (from serve_fastapi.py):')
    report.append('- GET /health')
    report.append('- POST /predict')
    report.append('- POST /predict_batch')
    report.append('- GET /forecast')
    report.append('- GET /forecast_multi')
    report.append('- GET /liquidations')
    report.append('- GET /news')
    report.append('- GET /news_agg')
    report.append('- GET /forecast.csv')
    report.append('Notes: uses custom layers RevIN/TSMixer/ITransformer/DropPath; supports Keras3 loader via USE_KERAS3=1.')
    report.append('')

    report.append('10) NEWS PIPELINE (INGEST → SENTIMENT → FEATURES)')
    report.append('')
    report.append('- scripts/news_ingest.py: pulls CryptoPanic/CryptoCompare/CoinMarketCal; merges into parquet.')
    report.append('- scripts/news_dedup.py: deduplicates news files.')
    report.append('- scripts/news_sentiment_hf.py: HuggingFace sentiment (LedgerBERT + XLM-R).')
    report.append('- data/news/*.parquet: persistent news store and sentiment cache.')
    report.append('- models/twitter-xlmr-sentiment: local cached HF model snapshot (for offline use).')
    report.append('')

    report.append('11) LIVE UPDATE PIPELINE (AUTO REFRESH)')
    report.append('')
    report.append('- scripts/live_update_once.sh:')
    report.append('  1) ingest news → dedup → sentiment (requires XLM-R + LedgerBERT)')
    report.append('  2) update OHLCV via bybit_data.py')
    report.append('  3) rebuild features with news windows and multi-horizons 20..160')
    report.append('- scripts/live_update_loop.sh: runs live_update_once.sh every 60s (lock file).')
    report.append('- scripts/run_live_site.sh: starts live_update loop + run_servers.sh.')
    report.append('')

    report.append('12) MODELS AND EXPERIMENTS')
    report.append('')
    report.append('Key production/battle models (from project context):')
    report.append('- model_battle_itransformer.keras + norm_stats_battle_itransformer.npz (h20 / 5h).')
    report.append('- model_15m_itransformer_tb_multi.keras + norm_stats_15m_itransformer_tb_multi.npz (multi h20/80/160).')
    report.append('New experiment folders (new_models/):')
    report.append('  ' + ', '.join(new_models))
    report.append('Current configured serve model (scripts/run_servers.sh):')
    report.append('- v6 8-node model: new_models/2026-01-14_news_xlmr_v6_dr_scale_8nodes/...')
    report.append('')

    report.append('13) KEY METRICS (FROM reports/ WALK-FORWARD CSVs)')
    report.append('')
    def metrics_block(name, row):
        if not row:
            return [f'{name}: (not found)']
        lines = [f'{name}:']
        for k in ['model','sharpe','max_dd','total_return','cagr','threshold','short_threshold','fee_bps','slip_bps','min_hold','cooldown','max_trades_per_day','n_trades','n_samples']:
            if k in row:
                lines.append(f'  {k}: {row[k]}')
        return lines
    report.extend(metrics_block('wf_tb.csv', wf_tb))
    report.extend(metrics_block('wf_battle.csv', wf_battle))
    report.extend(metrics_block('wf_itransformer_moderate_plus.csv', wf_moderate))
    report.append('')

    report.append('14) LOSSES AND OBJECTIVES (train_keras.py)')
    report.append('')
    report.append('Classification: BCE or Focal loss (focal-alpha, focal-gamma).')
    report.append('Regression: Huber (default), MSE, MAE, LogCosh. Optional quantile pinball loss if --quantiles used.')
    report.append('Price heads: can be single price or multi-horizon price heads (price_h20/price_h80/...).')
    report.append('Normalization/clipping: q_low/q_high for target clipping; stats saved in norm_stats_*.npz.')
    report.append('')

    report.append('15) ERRORS ENCOUNTERED (FROM LOGS) AND HOW WE MITIGATED')
    report.append('')
    report.append('FastAPI/Keras3 load errors (logs/fastapi.log):')
    report.append(wrap(fastapi_err.strip()))
    report.append('Mitigation: serve_fastapi.py includes Keras3 custom layer loader; run_servers.sh sets USE_KERAS3=1/TF_USE_LEGACY_KERAS=0.')
    report.append('')
    report.append('Live update errors (logs/live_update.log):')
    report.append(wrap(live_update_err.strip()))
    report.append('Mitigation: live_update_once.sh continues on ingest failures and reuses existing news cache when present.')
    report.append('')
    report.append('Cloudflared tunnel errors (logs/cloudflared.log):')
    report.append(wrap(cloudflared_err.strip()))
    report.append('Mitigation: requires valid origin cert, permissions for ICMP/ping_group_range, stable network.')
    report.append('')

    report.append('15.5) NEWS MODEL EVAL METRICS (v2–v6 FROM LOGS)')
    report.append('')
    report.append('Format: horizon n mae_usd rmse_usd mape_pct smape_pct direction_acc_pct [line_mae_usd line_rmse_usd] start_ts')
    for name, path in eval_logs.items():
        report.append('')
        report.append(f'- {name}: {path}')
        rows = parse_eval_rows(path)
        if rows:
            for r in rows:
                report.append(f'  {r}')
        else:
            report.append('  (no eval rows found in log; v6 has training log only, eval pending)')
    report.append('')

    report.append('15.6) NEWS MODELS: TEXT COMPARISON (SUMMARY)')
    report.append('')
    report.append('Based on logs in logs/eval_news_xlmr_v2_docker.log, eval_news_xlmr_v3_line.log,')
    report.append('eval_news_xlmr_v4_pw_*_line.log, eval_news_xlmr_v5_dr_scale_line.log:')
    report.append('- v2 (rolling-news baseline): h20 MAE ~702.55, h80 ~1512.02, h160 ~2190.51. No line metrics.')
    report.append('- v3 (rolling-news + line metrics): h20 MAE ~702.51 (line MAE ~472.34),')
    report.append('  h80 MAE ~1525.47 (line MAE ~997.85), h160 MAE ~2204.70 (line MAE ~1431.14).')
    report.append('- v4 (price-weight): h20 improves slightly (MAE ~700.17, line MAE ~471.26),')
    report.append('  but h80/h160 degrade (MAE ~1536.65 / ~2288.03; line MAE ~1002.75 / ~1471.93).')
    report.append('- v5 (dR + per-head scale): best overall balance in logs:')
    report.append('  h20 MAE ~698.47 (line MAE ~470.47), h80 MAE ~1511.02 (line MAE ~988.60),')
    report.append('  h160 MAE ~2195.02 (line MAE ~1422.78).')
    report.append('- v6 (dR + per-head scale + 8 nodes): trained (logs/train_news_xlmr_v6_dr_scale_8nodes.log),')
    report.append('  eval pending due to Keras3 loader issues in scripts/eval_price_quality.py.')
    report.append('')

    report.append('15.7) RECOMMENDED CURRENT PROD MODEL (SHORT)')
    report.append('')
    report.append('- Until v6 evaluation is fixed: use v5 (dR + per-head scale) as the default news model,')
    report.append('  since it shows best overall balance in both MAE and LineMAE across h20/h80/h160 in logs.')
    report.append('- v4 price-weight can be used only for h20 if you want tiny gains on short horizon;')
    report.append('  for h80/h160 it degrades accuracy in logs.')
    report.append('- v3 is acceptable but generally worse than v5 in both MAE and line metrics.')
    report.append('')

    report.append('16) RUN / START COMMANDS (SINGLE COMMANDS)')
    report.append('')
    report.append('Local site (FastAPI + UI):')
    report.append('  bash scripts/run_servers.sh')
    report.append('Live auto-update + site:')
    report.append('  bash scripts/run_live_site.sh')
    report.append('News ingest only:')
    report.append('  python3 scripts/news_ingest.py --out data/news/news.parquet --currency BTC --max-items 200')
    report.append('')

    report.append('17) IMPORTANT FILES TO KEEP IN SYNC')
    report.append('')
    report.append('- Features parquet and stats must match: mismatch causes runtime errors.')
    report.append('- Models + norm_stats in new_models/ should be kept with their training logs in logs/.')
    report.append('- API_BASE in html/aladin_from_image.html must point to running FastAPI.')
    report.append('')

    return '\n'.join(report)


def write_pdf(text: str, out_path: Path):
    PAGE_W, PAGE_H = 595, 842
    MARGIN_X, MARGIN_Y = 40, 40
    FONT_SIZE = 9
    LINE_HEIGHT = FONT_SIZE + 3

    lines = []
    for para in text.split('\n'):
        if para.strip() == '':
            lines.append('')
        else:
            lines.extend(textwrap.wrap(para, width=110, replace_whitespace=False))

    lines_per_page = int((PAGE_H - 2 * MARGIN_Y) / LINE_HEIGHT)
    pages = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]

    objects = []

    def add(obj):
        objects.append(obj)

    # Font object number is last
    font_obj_num = 3 + len(pages) * 2
    font_obj = f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    page_kids = []
    # Pages and contents
    for i, page_lines in enumerate(pages):
        content_text = "BT\n/F1 %d Tf\n%d %d Td\n" % (FONT_SIZE, MARGIN_X, PAGE_H - MARGIN_Y)
        for line in page_lines:
            safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            content_text += f"({safe}) Tj\nT*\n"
        content_text += "ET\n"
        content_obj_num = 4 + i * 2
        content_obj = f"{content_obj_num} 0 obj\n<< /Length {len(content_text.encode('utf-8'))} >>\nstream\n{content_text}endstream\nendobj\n"
        add(content_obj)

        page_obj_num = 3 + i * 2
        page_obj = (
            f"{page_obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_obj_num} 0 R >>\n"
            f"endobj\n"
        )
        add(page_obj)
        page_kids.append(f"{page_obj_num} 0 R")

    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(page_kids)}] /Count {len(pages)} >>\nendobj\n"
    catalog_obj = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    # Reorder objects: catalog, pages, then page/contents, then font
    ordered = [catalog_obj, pages_obj]
    ordered.extend(objects)
    ordered.append(font_obj)

    # Build xref
    offsets = []
    current = 0
    for obj in ordered:
        offsets.append(current)
        current += len(obj.encode('utf-8'))

    xref_offset = current + len('%PDF-1.4\n')
    xref = ["xref\n0 %d\n" % (len(ordered) + 1), "0000000000 65535 f \n"]
    for off in offsets:
        xref.append(f"{off:010d} 00000 n \n")

    trailer = (
        "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%EOF\n"
        % (len(ordered) + 1, xref_offset)
    )

    with out_path.open('wb') as f:
        f.write(b"%PDF-1.4\n")
        for obj in ordered:
            f.write(obj.encode('utf-8'))
        f.write(''.join(xref).encode('utf-8'))
        f.write(trailer.encode('utf-8'))


def main():
    text = build_report()
    OUT_TXT.write_text(text)
    write_pdf(text, OUT_PDF)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
