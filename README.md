# When the Council Is Wrong
## A study of how multi-LLM councils fail at prediction-market decisions

This project runs a **3-stage LLM council** over weather markets on
[Kalshi](https://kalshi.com) and records every decision so we can study *when
the council gets it wrong, and why*. It is a research instrument, not a money
machine — **all trading is paper trading** (no real or live demo orders are
placed).

The bet behind the project: a panel of diverse LLMs that critique each other
should make better-calibrated probability estimates than any single model. The
research question is the inverse — **where does that break down?** When does
peer review push the council toward a *worse* answer, manufacture false
consensus, or fail to veto a bad trade?

---

## How it works

For each candidate weather market, three cheap LLMs and one stronger "chairman"
run a council adapted from [Karpathy's llm-council](https://github.com/karpathy/llm-council):

1. **Stage 1 — Independent Analysis.** Each council model gets the *same*
   context packet (ensemble + official forecast + live market prices) and
   answers independently: `{probability, side, confidence, reasoning}`.
2. **Stage 2 — Peer Review.** Each model sees the whole panel's Stage-1 answers,
   **anonymized as "Model A / B / C"** (it can't tell which is its own or who
   wrote what), and may revise: `{updated_probability, agreements, disagreements, reasoning}`.
3. **Stage 3 — Chairman Synthesis.** A stronger model reads every Stage-1 answer
   and every Stage-2 review and issues the final call:
   `{final_probability, confidence, should_trade, side, dissent_summary, reasoning, risk_factors}`.

Every stage, every model's probability, and every reasoning chain is persisted
to the **`council_decisions`** table — the raw material for studying failures.

Council models: `gemini-2.5-flash-lite`, `deepseek-v3.2`, `gpt-4o-mini`.
Chairman: `claude-sonnet-4-20250514`. All via an OpenAI-compatible LLM Gateway.
A full council run costs roughly **$0.012–0.016**.

### Weather data

The market in question is **KXHIGHNY** — the NWS-settled daily high temperature
at NYC Central Park. We feed the council three forecast sources:

- **GFS** ensemble (NOAA GEFS, ~31 members) via Open-Meteo
- **ICON** ensemble (DWD, ~40 members) via Open-Meteo
- **NWS** official forecast high — the *literal settlement source* for the market

These markets are a clean testbed: the settlement source is public and
deterministic, books quote tight spreads, and the outcome resolves in ~24h.

### The combined decision

The weather model and the council must **both agree** before a paper trade is
recorded:

- weather model edge ≥ **15¢**, **and**
- council `should_trade == true`, **and**
- council confidence > **0.6**

If either disagrees, no trade — but the council decision is logged regardless.
The interesting research cases are exactly the disagreements.

### Monitoring

A FastAPI dashboard (`dashboard/app.py`) reads the same database and shows
positions, P&L, and decision history. On Railway it runs as the `web` process.

---

## Layout

```
kalshi-trader/
├── agents/
│   ├── council.py          ← the 3-stage WeatherCouncil engine
│   └── base_agent.py       ← LLM Gateway client + JSON extraction + cost logging
├── strategies/
│   └── weather.py          ← GFS/ICON/NWS pipeline + get_weather_context()
├── execution/
│   ├── risk_engine.py      ← sizes council-approved (paper) positions
│   ├── position_manager.py ← exit logic (stop-loss disabled for weather)
│   └── order_executor.py   ← order plumbing (paper flow does not call it)
├── core/                   ← Kalshi REST client, RSA-PSS auth, rate limiter
├── data/
│   ├── db.py               ← SQLAlchemy models incl. council_decisions
│   └── market_scanner.py   ← category inference used by the risk engine
├── monitoring/telegram_bot.py
├── dashboard/app.py        ← FastAPI monitoring dashboard
├── analysis/analyze_trades.py  ← retrospective P&L / calibration analysis
├── scripts/
│   ├── active_trader.py    ← main loop: weather scan → council → paper trade
│   ├── council_test.py     ← run the council on one live market, print everything
│   └── wipe_db.py          ← clean-slate DB wipe
├── config/settings.py
├── Procfile · railway.toml · runtime.txt
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in LLM_GATEWAY_API_KEY, DATABASE_URL, Kalshi keys
```

Run the council on one real KXHIGHNY market and print every stage (no trade,
no DB writes):

```bash
python -m scripts.council_test
```

Run the continuous paper-trading engine (dry-run by default):

```bash
python -m scripts.active_trader
```

Launch the dashboard:

```bash
uvicorn dashboard.app:app --reload
```

---

## Note

This is a research project about decision quality, not investment advice. Every
trade here is on paper. The point is the data in `council_decisions`, not P&L.
