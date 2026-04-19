# Kalshi Automated Trading System

A quantitative trading system for Kalshi prediction markets.

## Quick Start (do these in order)

### Step 1: Install dependencies
```bash
cd kalshi-trader
pip install -r requirements.txt
```

### Step 2: Create your .env file
```bash
cp .env.example .env
```

Edit `.env` — for now, leave the API key fields empty. The public endpoints
work without auth, so you can test immediately.

### Step 3: Run the smoke test
```bash
python -m scripts.smoke_test
```

This verifies: config loads → database creates → public API works → orderbook
parses → Kelly sizer computes. If auth is not configured, it skips that step
gracefully and tells you what to do.

### Step 4: Scan all markets
```bash
python -m scripts.scan_markets
```

This pulls every open market on Kalshi and shows you:
- Category breakdown (sports, crypto, economics, etc.)
- Top 25 markets by volume
- Spread and depth analysis for top 10

### Step 5: Watch a live market
```bash
# Auto-pick highest volume market:
python -m scripts.watch_orderbook

# Watch a specific series (e.g., BTC 15-minute):
python -m scripts.watch_orderbook KXBTC15M

# Watch an exact market:
python -m scripts.watch_orderbook --ticker KXBTC15M-26APR09-T100000
```

This is your first "trading screen" — watch how prices move, how spreads
tighten and widen, and where volume clusters.

### Step 6: Set up API keys (when ready to paper trade)
1. Go to https://demo.kalshi.co and create a demo account
2. Go to Account Settings → API Keys → Create New API Key
3. Save the private key PEM file to `keys/kalshi-private.pem`
4. Update `.env`:
   ```
   KALSHI_API_KEY_ID=your-key-id-here
   KALSHI_PRIVATE_KEY_PATH=./keys/kalshi-private.pem
   KALSHI_ENV=demo
   ```
5. Re-run smoke test to verify auth works

## Project Structure

```
kalshi-trader/
├── config/
│   ├── settings.py        ← Central config (loads .env)
│   └── constants.py       ← Rate limits, categories, watched series
├── core/
│   ├── auth.py            ← RSA-PSS request signing
│   ├── rest_client.py     ← Authenticated REST API wrapper
│   └── rate_limiter.py    ← Token bucket rate limiter
├── data/
│   ├── db.py              ← SQLAlchemy models + connection
│   └── market_scanner.py  ← Market discovery and cataloging
├── strategies/
│   └── kelly.py           ← Fractional Kelly position sizing
├── execution/             ← (Phase 2: order management)
├── monitoring/            ← (Phase 3: dashboards, alerts)
├── scripts/
│   ├── smoke_test.py      ← Run first — verifies everything
│   ├── scan_markets.py    ← Find where the money is
│   └── watch_orderbook.py ← Live market viewer
├── .env.example           ← Template for secrets
├── .gitignore
├── requirements.txt
└── README.md
```

## Risk Warning

Trading prediction markets involves substantial risk of loss. Start with
demo money. Paper trade for 2+ weeks before using real funds. Never trade
money you can't afford to lose.
