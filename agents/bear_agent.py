"""
Bear agent — argues WHY a Kalshi market resolves NO.

Sees the bull's output first and must counter each argument. This
forced-counter protocol prevents shallow "agree with everything"
capitulation from the bear side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents.base_agent import call_llm, LLMResponse
from agents.bull_agent import BullCase

BEAR_MODEL = "deepseek-v3.2"

SYSTEM_PROMPT = (
    "You are a bear analyst for prediction markets. You've seen the bull's "
    "case. Your job is to argue WHY this event will NOT happen. Counter each "
    "of the bull's arguments. Provide your probability estimate, a probability "
    "ceiling (maximum reasonable YES probability), 3 counter-arguments. "
    "Respond in JSON: "
    "{probability, probability_ceiling, confidence, counter_arguments: [str, str, str], reasoning: str}"
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


def _format_bull(bull: BullCase) -> str:
    bull_payload = {
        "probability": bull.probability,
        "probability_floor": bull.probability_floor,
        "confidence": bull.confidence,
        "arguments": bull.arguments,
        "reasoning": bull.reasoning,
    }
    return json.dumps(bull_payload, indent=2)


@dataclass
class BearCase:
    probability: float
    probability_ceiling: float
    confidence: float
    counter_arguments: list[str]
    reasoning: str
    llm: LLMResponse


def run_bear(market: dict[str, Any], bull: BullCase) -> BearCase:
    user_prompt = (
        f"{_format_market(market)}\n"
        f"BULL CASE (counter each argument):\n{_format_bull(bull)}\n\n"
        "Argue the BEAR case (NO will resolve) and counter the bull point by point. "
        "Return JSON only."
    )
    resp = call_llm(
        model=BEAR_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    p = resp.parsed
    return BearCase(
        probability=float(p.get("probability", 0.5)),
        probability_ceiling=float(p.get("probability_ceiling", 1.0)),
        confidence=float(p.get("confidence", 0.5)),
        counter_arguments=list(p.get("counter_arguments", []))[:3],
        reasoning=str(p.get("reasoning", "")),
        llm=resp,
    )
