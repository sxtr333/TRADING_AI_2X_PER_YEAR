#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/vitamind/my_project/model6"
VENV="$ROOT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found at $VENV"
  exit 1
fi

source "$VENV/bin/activate"

# Reset checkpoints + outputs
rm -f /mnt/data/cc-news/checkpoints/processed_2023.txt
rm -f /mnt/data/cc-news/checkpoints/processed_2024.txt
rm -f /mnt/data/cc-news-2023/*.parquet
rm -f /mnt/data/cc-news-2024/*.parquet

mkdir -p /mnt/data/cc-news-2023 /mnt/data/cc-news-2024 /mnt/data/cc-news/checkpoints

COMMON_ARGS=(
  --max-warc 4 --warc-sample random --warc-seed 42
  --flush 200 --log-every 200
  --score-min 2 --gold-score 3 --silver-score 2
  --keywords "bitcoin,btc,crypto,cryptocurrency,blockchain,ethereum"
  --macro-keywords "etf,sec,regulation,macro,fomc,rate,interest,cpi,ppi,inflation,employment,jobs,treasury,bank,liquidity,credit,default,stress,volatility"
  --infra-keywords "binance,coinbase,kraken,bybit,okx,bitstamp,gemini,metamask,ledger,trezor,stablecoin,usdt,usdc,usde,usdd,bridge,layer2,l2,dex,amm,oracle,staking"
  --event-keywords "listing,delisting,futures,perpetual,airdrop,unlock,upgrade,hardfork,exploit,hack,outage,breach,sanction,investigation,indictment,settlement,funding_round,raise,acquisition,merger"
  --url-keywords "crypto,bitcoin,btc,ethereum,eth,blockchain,web3,stablecoin,digital-asset"
  --exclude-keywords "football,soccer,nba,nfl,mlb,nhl,tennis,cricket,goal.com,match,score,fixtures,weather,forecast,temperature,climate,crime,police,arrested,murder,shooting,accident,celebrity,entertainment,horoscope,astrology,gossip,lottery,gambling,coupon,betting,casino"
  --block-domains "goal.com,einpresswire.com,news.livedoor.com,infobae.com,mexc.com,mexc.fm,apolyton.net,kenyan-post.com"
)

# 2023
python3 "$ROOT_DIR/scripts/cc_news_pipeline.py" \
  --start-month 2023-01 --end-month 2023-12 \
  --out-dir /mnt/data/cc-news-2023 \
  --checkpoint /mnt/data/cc-news/checkpoints/processed_2023.txt \
  "${COMMON_ARGS[@]}" \
  > /mnt/data/cc-news-2023/run.log 2>&1 &

# 2024
python3 "$ROOT_DIR/scripts/cc_news_pipeline.py" \
  --start-month 2024-01 --end-month 2024-12 \
  --out-dir /mnt/data/cc-news-2024 \
  --checkpoint /mnt/data/cc-news/checkpoints/processed_2024.txt \
  "${COMMON_ARGS[@]}" \
  > /mnt/data/cc-news-2024/run.log 2>&1 &

echo "Started. Logs:"
echo "  tail -f /mnt/data/cc-news-2023/run.log"
echo "  tail -f /mnt/data/cc-news-2024/run.log"
