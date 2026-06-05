"""
Base LLM agent — OpenAI-compatible client pointed at LLM Gateway.

Every debate agent (bull, bear, judge, forecaster, ...) goes through this
module. Centralizes: API client construction, JSON extraction, retry
logic, and per-call logging (model, tokens, latency, cost).

JSON extraction has 3 fallbacks because real LLMs will randomly:
  1. wrap JSON in ```json fences
  2. return bare {...} inside narration
  3. return raw JSON already (best case)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import structlog
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

from config.settings import settings

logger = structlog.get_logger()


# ---- Rough input/output pricing per 1M tokens (USD). Used for cost logging.
# These are estimates — refreshed periodically. A missing entry just means
# cost logging falls back to 0, the call still succeeds.
_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_1m, output_per_1m) — LLM Gateway bare names
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "deepseek-v3.2": (0.27, 1.10),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
}


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Singleton client. Created lazily so imports don't fail without the key."""
    global _client
    if _client is None:
        if not settings.llm_gateway_api_key:
            raise RuntimeError(
                "LLM_GATEWAY_API_KEY not set in .env. Add it and restart."
            )
        _client = OpenAI(
            base_url=settings.llm_gateway_base_url,
            api_key=settings.llm_gateway_api_key,
        )
    return _client


# ----------------------------------------------------------------------
# JSON extraction — 3-strategy fallback
# ----------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """
    Try three parse strategies in order:
      1. fenced ```json ... ``` block
      2. first balanced {...} in the text
      3. raw text itself

    Raises ValueError if none parse.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    # 1. Fenced block
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2. Bare {...} regex (greedy, grabs outermost braces)
    brace_match = _BRACE_RE.search(text)
    if brace_match:
        candidate = brace_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Raw text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"could not extract JSON: {e}; snippet={text[:200]!r}")


# ----------------------------------------------------------------------
# Main LLM call
# ----------------------------------------------------------------------

@dataclass
class LLMResponse:
    parsed: dict            # extracted JSON
    raw_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    cost_usd: float


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICING.get(model)
    if not prices:
        return 0.0
    in_price, out_price = prices
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 3000,
    retries: int = 3,
) -> LLMResponse:
    """
    Call an LLM via LLM Gateway. Returns parsed JSON + usage stats.

    Retries on transient errors (rate limit, connection, 5xx) with
    exponential backoff: 1s, 2s, 4s. Permanent errors raise immediately.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None

    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as e:
            wait = 2 ** attempt
            logger.warning("llm_rate_limited", model=model, wait=wait, attempt=attempt + 1)
            last_error = e
            time.sleep(wait)
            continue
        except APIConnectionError as e:
            wait = 2 ** attempt
            logger.warning("llm_connection_error", model=model, error=str(e)[:200], wait=wait)
            last_error = e
            time.sleep(wait)
            continue
        except APIError as e:
            # Permanent 4xx errors (auth, insufficient credits, bad request) should NOT retry.
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                logger.error("llm_permanent_error", model=model, status=status, error=str(e)[:200])
                raise
            wait = 2 ** attempt
            logger.warning("llm_transient_error", model=model, error=str(e)[:200], wait=wait)
            last_error = e
            time.sleep(wait)
            continue

        latency_ms = int((time.time() - t0) * 1000)
        raw_text = resp.choices[0].message.content or ""
        usage = resp.usage
        p_tok = usage.prompt_tokens if usage else 0
        c_tok = usage.completion_tokens if usage else 0
        total_tok = usage.total_tokens if usage else (p_tok + c_tok)
        cost = _estimate_cost(model, p_tok, c_tok)

        try:
            parsed = extract_json(raw_text)
        except ValueError as e:
            logger.warning(
                "llm_json_parse_failed",
                model=model, error=str(e), attempt=attempt + 1, raw_snippet=raw_text[:300],
            )
            last_error = e
            # Parse failure is often a stochastic bad-format; retry once
            time.sleep(1)
            continue

        logger.info(
            "llm_call",
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=total_tok,
            latency_ms=latency_ms,
            cost_usd=round(cost, 6),
        )

        return LLMResponse(
            parsed=parsed,
            raw_text=raw_text,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=total_tok,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    raise RuntimeError(
        f"LLM call to {model} failed after {retries} attempts: {last_error}"
    )
