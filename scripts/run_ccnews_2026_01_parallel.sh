#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/vitamind/my_project/model6/.venv/bin/python"
PIPELINE="/home/vitamind/my_project/model6/scripts/cc_news_pipeline.py"

OUT_BASE="/mnt/data/cc-news-2026"
CKPT_BASE="/mnt/data/cc-news/checkpoints"
LOG_BASE="/mnt/data/cc-news-2026"

mkdir -p "$OUT_BASE/p1" "$OUT_BASE/p2" "$OUT_BASE/p3" "$CKPT_BASE" "$LOG_BASE"

common_args=(
  --start-month 2026-01 --end-month 2026-01
  --score-min 2 --gold-score 3 --silver-score 2
  --keywords "bitcoin,btc,crypto,cryptocurrency,blockchain,ethereum"
  --macro-keywords "etf,sec,regulation,macro,fomc,rate,interest,cpi,ppi,inflation,employment,jobs,treasury,bank,liquidity,credit,default,stress,volatility"
  --infra-keywords "binance,coinbase,kraken,bybit,okx,bitstamp,gemini,metamask,ledger,trezor,stablecoin,usdt,usdc,usde,usdd,bridge,layer2,l2,dex,amm,oracle,staking"
  --event-keywords "listing,delisting,futures,perpetual,airdrop,unlock,upgrade,hardfork,exploit,hack,outage,breach,sanction,investigation,indictment,settlement,funding_round,raise,acquisition,merger"
  --url-keywords "crypto,bitcoin,btc,ethereum,eth,blockchain,web3,stablecoin,digital-asset"
  --exclude-keywords "football,soccer,nba,nfl,mlb,nhl,tennis,cricket,goal.com,match,score,fixtures,weather,forecast,temperature,climate,crime,police,arrested,murder,shooting,accident,celebrity,entertainment,horoscope,astrology,gossip,lottery,gambling,coupon,betting,casino"
  --block-domains "goal.com,einpresswire.com,news.livedoor.com,infobae.com,mexc.com,mexc.fm,apolyton.net,kenyan-post.com"
  --flush 20
)

run_worker() {
  local name="$1"
  local seed="$2"
  local max_warc="$3"
  local out_dir="$OUT_BASE/$name"
  local ckpt="$CKPT_BASE/processed_2026_01_${name}.txt"
  local log="$LOG_BASE/run_2026_01_${name}.log"

  while true; do
    "$PYTHON_BIN" "$PIPELINE" \
      --out-dir "$out_dir" \
      --checkpoint "$ckpt" \
      --max-warc "$max_warc" \
      --warc-sample random \
      --warc-seed "$seed" \
      "${common_args[@]}" \
      >> "$log" 2>&1

    if tail -n 5 "$log" | rg -q "Done\."; then
      break
    fi
    sleep 10
  done
}

run_worker p1 101 17 &
run_worker p2 202 17 &
run_worker p3 303 16 &

wait
