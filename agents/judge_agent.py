"""
Judge agent — synthesizes bull + bear into a final probability + trade call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents.base_agent import call_llm, LLMResponse
from agents.bull_agent import BullCase
from agents.bear_agent import BearCase

JUDGE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are the final judge. You've seen the bull case and bear case for "
    "this prediction market. Synthesize both perspectives. Your job is to "
    "output the TRUE probability of this event happening, accounting for "
    "both sides. Also assess: should we trade this? What side? How confident "
    "are you? "
    "Respond in JSON: "
    "{probability, confidence, should_trade: bool, side: 'yes'|'no'|'hold', edge_assessment: str, reasoning: str}"
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
    return json.dumps({
        "probability": bull.probability,
        "probability_floor": bull.probability_floor,
        "confidence": bull.confidence,
        "arguments": bull.arguments,
        "reasoning": bull.reasoning,
    }, indent=2)


def _format_bear(bear: BearCase) -> str:
    return json.dumps({
        "probability": bear.probability,
        "probability_ceiling": bear.probability_ceiling,
        "confidence": bear.confidence,
        "counter_arguments": bear.counter_arguments,
        "reasoning": bear.reasoning,
    }, indent=2)


@dataclass
class JudgeVerdict:
    probability: float
    confidence: float
    should_trade: bool
    side: str              # "yes" | "no" | "hold"
    edge_assessment: str
    reasoning: str
    llm: LLMResponse


def run_judge(market: dict[str, Any], bull: BullCase, bear: BearCase) -> JudgeVerdict:
    user_prompt = (
        f"{_format_market(market)}\n"
        f"BULL CASE:\n{_format_bull(bull)}\n\n"
        f"BEAR CASE:\n{_format_bear(bear)}\n\n"
        "Synthesize both into a final probability and trade recommendation. "
        "Return JSON only."
    )
    resp = call_llm(
        model=JUDGE_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    p = resp.parsed
    side = str(p.get("side", "hold")).lower()
    if side not in ("yes", "no", "hold"):
        side = "hold"
    return JudgeVerdict(
        probability=float(p.get("probability", 0.5)),
        confidence=float(p.get("confidence", 0.5)),
        should_trade=bool(p.get("should_trade", False)),
        side=side,
        edge_assessment=str(p.get("edge_assessment", "")),
        reasoning=str(p.get("reasoning", "")),
        llm=resp,
    )
