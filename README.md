# TradeForge

TradeForge is a crypto market analytics dashboard for BTCUSDT signals, model comparison, market radar and subscription-based access.

## Stack

- FastAPI backend
- Static HTML/CSS/JavaScript frontend
- Docker Compose deployment
- TradingView Lightweight Charts

## Run

```bash
cp .env.example .env
./scripts/docker_site_up.sh
```

Private datasets, trained weights, user databases, logs and production secrets are not included.
