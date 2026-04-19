"""
Strategy — AI Debate (Layer 3 in blueprint v2).

Pipeline
--------
    bull agent       (gemini-2.5-flash)
        │
        ▼ bull.arguments
    bear agent       (deepseek-chat-v3)  — forced to counter bull
        │
        ▼ bear.counter_arguments
    judge agent      (claude-sonnet-4)   — synthesizes final probability

Disagreement penalty
--------------------
If |bull.prob - bear.prob| > 0.30, we scale judge.confidence by 0.7.
Big disagreement between the bull and bear means the LLM universe
itself can't agree — not a regime we should trade confidently.

Cost control
------------
- Only runs on markets with price in [0.10, 0.90] (real uncertainty).
- Max 5 markets per scan cycle.
- Picks the 5 with highest orderbook depth (liquidity where the edge is
  actually capturable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import structlog

from agents.bull_agent import run_bull, BullCase
from agents.bear_agent import run_bear, BearCase
from agents.judge_agent import run_judge, JudgeVerdict
from core.rest_client import KalshiClient
from data.market_scanner import mkt_yes_bid, mkt_yes_ask, mkt_volume, infer_category

logger = structlog.get_logger()


# --- Gate parameters ---
PRICE_MIN = Decimal("0.10")
PRICE_MAX = Decimal("0.90")
MAX_DEBATES_PER_CYCLE = 5
DISAGREEMENT_THRESHOLD = 0.30
DISAGREEMENT_PENALTY = Decimal("0.70")   # multiply confidence by this when over threshold


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------

@dataclass
class DebateResult:
    ticker: str
    market_title: str
    category: str
    market_price: Decimal
    side: str                   # "yes" | "no" | "hold"
    probability: Decimal        # judge's final probability
    confidence: Decimal         # judge's confidence, after disagreement penalty
    should_trade: bool
    edge: Decimal               # judge.probability - cost_on_chosen_side
    cost_per_contract: Decimal  # what we'd pay on the chosen side
    bull_prob: float
    bear_prob: float
    judge_prob: float
    disagreement: float
    total_cost_usd: float
    bull: BullCase
    bear: BearCase
    judge: JudgeVerdict


# ----------------------------------------------------------------------
# Market dict → agent input
# ----------------------------------------------------------------------

def _market_to_agent_input(market: dict[str, Any]) -> dict[str, Any]:
    """Normalize Kalshi market fields into the names the agents expect."""
    yes_bid = mkt_yes_bid(market)
    yes_ask = mkt_yes_ask(market)
    yes_price = (yes_bid + yes_ask) / 2 if (yes_bid > 0 and yes_ask > 0) else (yes_bid or yes_ask)
    return {
        "title": market.get("title", ""),
        "ticker": market.get("ticker", ""),
        "category": infer_category(market.get("ticker", "")),
        "yes_price": round(yes_price, 2),
        "no_price": round(1.0 - yes_price, 2) if yes_price else None,
        "volume": int(mkt_volume(market)),
        "close_time": market.get("close_time", ""),
    }


# ----------------------------------------------------------------------
# Core debate
# ----------------------------------------------------------------------

def run_debate(client: KalshiClient, market_data: dict[str, Any]) -> DebateResult:
    """
    Run the full 3-agent debate on ONE market. `market_data` is the raw
    Kalshi market dict (same shape `get_markets` returns).
    """
    agent_input = _market_to_agent_input(market_data)

    # 1. Bull
    bull = run_bull(agent_input)
    # 2. Bear (sees bull)
    bear = run_bear(agent_input, bull)
    # 3. Judge (sees both)
    judge = run_judge(agent_input, bull, bear)

    # --- Disagreement penalty ---
    disagreement = abs(bull.probability - bear.probability)
    adjusted_confidence = Decimal(str(judge.confidence))
    if disagreement > DISAGREEMENT_THRESHOLD:
        adjusted_confidence = adjusted_confidence * DISAGREEMENT_PENALTY

    # --- Edge computation on the judge's chosen side ---
    yes_bid = Decimal(str(mkt_yes_bid(market_data)))
    yes_ask = Decimal(str(mkt_yes_ask(market_data)))
    market_price = (yes_bid + yes_ask) / 2 if (yes_bid > 0 and yes_ask > 0) else (yes_bid or yes_ask)

    judge_prob = Decimal(str(judge.probability))
    if judge.side == "yes":
        cost = yes_ask if yes_ask > 0 else market_price
        edge = judge_prob - cost
    elif judge.side == "no":
        cost = Decimal("1") - yes_bid if yes_bid > 0 else Decimal("1") - market_price
        edge = (Decimal("1") - judge_prob) - cost
    else:
        cost = market_price
        edge = Decimal("0")

    total_cost = bull.llm.cost_usd + bear.llm.cost_usd + judge.llm.cost_usd

    logger.info(
        "ai_debate_complete",
        ticker=market_data.get("ticker"),
        bull_prob=bull.probability,
        bear_prob=bear.probability,
        judge_prob=judge.probability,
        disagreement=round(disagreement, 3),
        confidence=float(adjusted_confidence),
        side=judge.side,
        edge=float(edge),
        cost_usd=round(total_cost, 4),
    )

    return DebateResult(
        ticker=market_data.get("ticker", ""),
        market_title=market_data.get("title", ""),
        category=infer_category(market_data.get("ticker", "")),
        market_price=Decimal(str(round(market_price, 4))),
        side=judge.side,
        probability=judge_prob,
        confidence=adjusted_confidence,
        should_trade=judge.should_trade,
        edge=edge,
        cost_per_contract=cost,
        bull_prob=bull.probability,
        bear_prob=bear.probability,
        judge_prob=judge.probability,
        disagreement=round(disagreement, 4),
        total_cost_usd=round(total_cost, 6),
        bull=bull,
        bear=bear,
        judge=judge,
    )


# ----------------------------------------------------------------------
# Cycle-level scan
# ----------------------------------------------------------------------

def _orderbook_depth(client: KalshiClient, ticker: str) -> Decimal:
    """Return total depth (YES + NO, top 5 levels) for ranking."""
    try:
        raw = client.get_orderbook(ticker)
        parsed = KalshiClient.parse_orderbook(raw)
        return (
            KalshiClient.compute_depth(parsed["yes_bids"]) +
            KalshiClient.compute_depth(parsed["no_bids"])
        )
    except Exception:
        return Decimal("0")


def _eligible(market: dict[str, Any]) -> bool:
    """Price in [0.10, 0.90] and market is actually open."""
    if market.get("status") not in ("active", "open"):
        return False
    yes_bid = Decimal(str(mkt_yes_bid(market)))
    yes_ask = Decimal(str(mkt_yes_ask(market)))
    mid = (yes_bid + yes_ask) / 2 if (yes_bid > 0 and yes_ask > 0) else (yes_bid or yes_ask)
    return PRICE_MIN <= mid <= PRICE_MAX


def scan_with_debate(
    client: KalshiClient,
    markets: list[dict[str, Any]],
    max_debates: int = MAX_DEBATES_PER_CYCLE,
) -> list[DebateResult]:
    """
    Run the debate on up to `max_debates` markets from `markets`, picked
    by orderbook depth (most liquidity first).

    Returns DebateResult objects, including ones where should_trade=False
    so the caller can log them.
    """
    eligible = [m for m in markets if _eligible(m)]
    if not eligible:
        logger.info("debate_scan_empty", scanned=len(markets))
        return []

    # Rank by orderbook depth — one API call per market, so we cap at 30
    # candidates for the depth query before picking the top N.
    ranked_candidates = eligible[:30]
    depths = [(m, _orderbook_depth(client, m["ticker"])) for m in ranked_candidates]
    depths.sort(key=lambda t: t[1], reverse=True)
    chosen = [m for m, _d in depths[:max_debates]]

    logger.info(
        "debate_scan_start",
        eligible=len(eligible),
        chosen=len(chosen),
        max_debates=max_debates,
    )

    results: list[DebateResult] = []
    for m in chosen:
        try:
            res = run_debate(client, m)
            results.append(res)
        except Exception as e:
            logger.warning(
                "debate_failed",
                ticker=m.get("ticker"),
                error=str(e)[:200],
            )
    return results
