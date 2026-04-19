"""
Bull agent — argues WHY a Kalshi market resolves YES.

Runs first in the debate; its output feeds the bear agent, which is
forced to counter each argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.base_agent import call_llm, LLMResponse

BULL_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a bull analyst for prediction markets. Your job is to argue WHY "
    "this event WILL happen. Provide your probability estimate (0-1), a "
    "probability floor (minimum reasonable YES probability), 3 key arguments, "
    "and your confidence (0-1). "
    "Respond in JSON only: "
    "{probability, probability_floor, confidence, arguments: [str, str, str], reasoning: str}"
)


def _format_market(market: dict[str, Any]) -> str:
    return (
        f"Market: {market.get('title', '')}\n"
        f"Ticker: {market.get('ticker', '')}\n"
        f"Category: {market.get('category', 'unknown')}\n"
        f"YES price: ${market.get('yes_price', '?')}\n"
        f"NO price: ${market.get('no_price', '?')}\n"
        f"Volume (contracts): {market.get('volume', '?')}\n"
        f"Close time: {market.get('close_time', '?')}\n"
    )


@dataclass
class BullCase:
    probability: float
    probability_floor: float
    confidence: float
    arguments: list[str]
    reasoning: str
    llm: LLMResponse


def run_bull(market: dict[str, Any]) -> BullCase:
    user_prompt = (
        f"{_format_market(market)}\n"
        "Argue the BULL case (YES will resolve). "
        "Return JSON only, no prose outside the JSON."
    )
    resp = call_llm(
        model=BULL_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    p = resp.parsed
    return BullCase(
        probability=float(p.get("probability", 0.5)),
        probability_floor=float(p.get("probability_floor", 0.0)),
        confidence=float(p.get("confidence", 0.5)),
        arguments=list(p.get("arguments", []))[:3],
        reasoning=str(p.get("reasoning", "")),
        llm=resp,
    )
