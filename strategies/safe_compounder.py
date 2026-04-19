"""
Strategy — Safe Compounder (rule-based, no AI)

Thesis
------
Extreme-priced binary markets (YES <= 0.15 or YES >= 0.85) converge toward
their resolved outcome as expiry approaches. Buying the extreme side with
a time-decay tailwind captures that drift at near-zero fees.

Pipeline
--------
    broad market scan (filtered by series tickers we care about)
         │
         ▼
    filter: YES bid in [0.00, 0.15]  OR  [0.85, 1.00]
         │
         ▼
    time_decay_bonus by hours-to-close:
         <6h  → 0.04
         <24h → 0.02
         <48h → 0.01
         else 0.00
         │
         ▼
    edge = time_decay_bonus  (filter edge >= 0.03)
         │
         ▼
    orderbook depth sanity check
         │
         ▼
    half-Kelly sizing, capped at 10% of portfolio

This strategy makes NO LLM calls — pure math.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import structlog

from core.rest_client import KalshiClient
from data.market_scanner import mkt_yes_bid, mkt_yes_ask, mkt_no_bid, infer_category

logger = structlog.get_logger()


# --- Thresholds (blueprint v2) ---
LOW_YES_MAX = Decimal("0.15")
HIGH_YES_MIN = Decimal("0.85")
MIN_EDGE = Decimal("0.03")
MIN_ORDERBOOK_DEPTH = Decimal("10")  # at least 10 contracts resting on the side we're buying
MAX_POSITION_PCT = Decimal("0.10")   # blueprint: 10% cap
KELLY_FRACTION = Decimal("0.50")     # half-Kelly


def _time_decay_bonus(hours_to_close: Optional[float]) -> Decimal:
    """Return the probability adjustment we grant for time decay."""
    if hours_to_close is None or hours_to_close < 0:
        return Decimal("0")
    if hours_to_close < 6:
        return Decimal("0.04")
    if hours_to_close < 24:
        return Decimal("0.02")
    if hours_to_close < 48:
        return Decimal("0.01")
    return Decimal("0")


def _parse_close_time(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _hours_to_close(market: dict) -> Optional[float]:
    close_dt = _parse_close_time(market.get("close_time"))
    if close_dt is None:
        return None
    delta = close_dt - datetime.now(timezone.utc)
    return delta.total_seconds() / 3600.0


@dataclass
class CompounderSignal:
    ticker: str
    title: str
    category: str
    side: str                    # "yes" or "no"
    cost_per_contract: Decimal   # what we pay
    edge: Decimal                # signed positive
    model_prob: Decimal          # implied probability of winning after bonus
    market_prob: Decimal         # raw market-implied probability of winning
    time_decay_bonus: Decimal
    hours_to_close: float
    orderbook_depth: Decimal     # contracts resting on the fill side
    confidence: Decimal          # 0..1 — tied to time-decay bonus magnitude


def _orderbook_depth_on_side(client: KalshiClient, ticker: str, side: str) -> Optional[Decimal]:
    """Return resting size on the side we'd be buying (top 5 price levels)."""
    try:
        raw = client.get_orderbook(ticker)
        parsed = KalshiClient.parse_orderbook(raw)
        bids = parsed["yes_bids"] if side == "yes" else parsed["no_bids"]
        return KalshiClient.compute_depth(bids, levels=5)
    except Exception as e:
        logger.warning("compounder_orderbook_failed", ticker=ticker, error=str(e))
        return None


def find_compounder_opportunities(
    client: KalshiClient,
    markets: list[dict],
    min_edge: Decimal = MIN_EDGE,
    min_depth: Decimal = MIN_ORDERBOOK_DEPTH,
) -> list[CompounderSignal]:
    """
    Score the given `markets` for safe-compounder entries.

    Caller is responsible for fetching markets — typically a filtered
    list targeting series we care about (weather + MLB + crypto etc.)
    rather than the full prod catalog.
    """
    signals: list[CompounderSignal] = []

    for m in markets:
        ticker = m.get("ticker", "")
        status = m.get("status", "")
        if status not in ("active", "open"):
            continue

        yes_bid = Decimal(str(mkt_yes_bid(m)))
        yes_ask = Decimal(str(mkt_yes_ask(m)))
        no_bid = Decimal(str(mkt_no_bid(m)))

        # Skip empty books
        if yes_bid <= 0 and yes_ask <= 0:
            continue

        hours = _hours_to_close(m)
        bonus = _time_decay_bonus(hours)
        if bonus <= 0:
            continue

        # Decide side based on which extreme we're in
        if yes_bid > 0 and yes_bid <= LOW_YES_MAX:
            # Low YES → buy NO. NO ask = 1 - yes_bid (what we pay to take NO).
            side = "no"
            cost = Decimal("1") - yes_bid
            market_prob = Decimal("1") - yes_bid   # market-implied P(NO wins)
            model_prob = market_prob + bonus
        elif yes_ask > 0 and yes_ask >= HIGH_YES_MIN:
            # High YES → buy YES at yes_ask
            side = "yes"
            cost = yes_ask
            market_prob = yes_ask               # market-implied P(YES wins)
            model_prob = market_prob + bonus
        else:
            continue

        # Guard: sane cost
        if cost <= Decimal("0.01") or cost >= Decimal("0.99"):
            continue

        # Edge on binary bet at this cost:
        #   edge = model_prob - cost   (same units as "how much the market mis-prices it")
        edge = model_prob - cost
        if edge < min_edge:
            continue

        # Orderbook depth check — we only trade where there's real liquidity
        depth = _orderbook_depth_on_side(client, ticker, side)
        if depth is None or depth < min_depth:
            continue

        # Confidence tracks bonus magnitude (0.04 → 0.80, 0.02 → 0.60, 0.01 → 0.40)
        if bonus >= Decimal("0.04"):
            confidence = Decimal("0.80")
        elif bonus >= Decimal("0.02"):
            confidence = Decimal("0.60")
        else:
            confidence = Decimal("0.40")

        signals.append(CompounderSignal(
            ticker=ticker,
            title=m.get("title", ""),
            category=infer_category(ticker),
            side=side,
            cost_per_contract=cost.quantize(Decimal("0.0001")),
            edge=edge.quantize(Decimal("0.0001")),
            model_prob=model_prob.quantize(Decimal("0.0001")),
            market_prob=market_prob.quantize(Decimal("0.0001")),
            time_decay_bonus=bonus,
            hours_to_close=round(hours, 2),
            orderbook_depth=depth,
            confidence=confidence,
        ))

    signals.sort(key=lambda s: s.edge, reverse=True)
    logger.info("compounder_scan_complete", scanned=len(markets), found=len(signals))
    return signals


def compute_compounder_size(
    edge: Decimal,
    cost: Decimal,
    portfolio_balance: Decimal,
    kelly_fraction: Decimal = KELLY_FRACTION,
    max_position_pct: Decimal = MAX_POSITION_PCT,
) -> int:
    """
    Half-Kelly sizing for compounder trades, capped at 10% of portfolio.

    Binary Kelly:  f* = (p * b - q) / b
        where b = payoff / cost = (1 - cost) / cost,  p = cost + edge,  q = 1 - p
    """
    edge = Decimal(str(edge))
    cost = Decimal(str(cost))

    if cost <= Decimal("0.01") or cost >= Decimal("0.99"):
        return 0
    if edge <= 0:
        return 0

    win_prob = cost + edge
    win_prob = min(Decimal("0.999"), max(Decimal("0.001"), win_prob))

    payoff = Decimal("1") - cost
    b = payoff / cost
    q = Decimal("1") - win_prob

    kelly_raw = (win_prob * b - q) / b
    kelly_raw = max(Decimal("0"), kelly_raw)

    kelly_bet = kelly_raw * kelly_fraction
    position_pct = min(kelly_bet, max_position_pct)

    dollar_amount = (portfolio_balance * position_pct).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    contracts = int(dollar_amount / cost)
    return max(0, contracts)
