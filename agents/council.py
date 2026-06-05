"""
WeatherCouncil — a 3-stage LLM council for weather-market decisions.

Adapted from Andrej Karpathy's llm-council (github.com/karpathy/llm-council).
We extract the PATTERN, not the code: three stages, anonymized peer review,
a chairman that synthesizes. The plumbing (LLM Gateway client, JSON
extraction, retry/cost logging) is our existing agents/base_agent.py.

    STAGE 1 — Independent Analysis
        Each council model receives the SAME context packet (ensemble +
        NWS forecast + live Kalshi prices) and answers independently:
        {probability, side, confidence, reasoning}.

    STAGE 2 — Peer Review
        Each model sees the WHOLE panel's Stage-1 answers, anonymized as
        "Model A / Model B / Model C" (it can't tell which is its own or
        who wrote what), and may update its probability:
        {updated_probability, agreements, disagreements, reasoning}.

    STAGE 3 — Chairman Synthesis
        One stronger model sees every Stage-1 answer + every Stage-2 review
        and produces the final call:
        {final_probability, confidence, should_trade, side,
         dissent_summary, reasoning, risk_factors}.

Design note: the A/B/C labels are STABLE across stages and map positionally
to `council_models`, so the council_decisions table can store per-model
columns cleanly. This is a research instrument about *council failure
modes* — so we keep every model's full reasoning chain, every stage, even
when a model errors out. Nothing is silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import structlog

from agents.base_agent import call_llm

logger = structlog.get_logger()


# ----------------------------------------------------------------------
# Model configuration — cheap council, stronger chairman
# ----------------------------------------------------------------------
# Bare LLM Gateway names (NOT provider/model form). Stage 1 + 2 use the
# three cheap models; Stage 3 uses the stronger chairman.
#
# NOTE: we use gemini-2.5-flash-LITE, not plain gemini-2.5-flash. The full
# flash model is a "thinking" model: via this OpenAI-compatible gateway its
# hidden reasoning tokens consume max_tokens before the visible JSON closes,
# so it reliably returned truncated/unparseable JSON in testing and lost its
# vote every run. flash-lite is non-thinking, ~3x cheaper, and returned clean
# JSON 3/3 in testing — same Google-family perspective, none of the truncation.
COUNCIL_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "deepseek-v3.2",
    "gpt-4o-mini",
)
CHAIRMAN_MODEL: str = "claude-sonnet-4-20250514"

# Anonymized labels, positional to COUNCIL_MODELS.
_LABELS = ("Model A", "Model B", "Model C", "Model D", "Model E")

# Per-call generation knobs. Low temperature — we want calibrated estimates,
# not creative writing. max_tokens is generous so the reasoning chain isn't
# truncated mid-JSON (we log it all). Headroom matters because some models
# (e.g. gemini-2.5-flash) spend tokens on internal "thinking" before the
# JSON, and a truncated response fails to parse — costing us that model's
# whole vote. We also ask models to keep reasoning concise (see prompts).
_TEMPERATURE = 0.3
_MAX_TOKENS = 2800

# Appended to every stage's instructions: keeps reasoning from ballooning
# past the token budget (which truncates the JSON and loses the vote).
_CONCISE = (
    "\nKeep your reasoning concise — a few focused sentences, not an essay. "
    "Output the JSON object and nothing else."
)


# ----------------------------------------------------------------------
# Result structures
# ----------------------------------------------------------------------

@dataclass
class Stage1Answer:
    model: str
    label: str
    probability: Optional[float]   # P(YES settles), 0..1
    side: str                      # "yes" | "no"
    confidence: Optional[float]    # 0..1
    reasoning: str
    raw_text: str = ""
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class Stage2Review:
    model: str
    label: str
    updated_probability: Optional[float]
    agreements: str
    disagreements: str
    reasoning: str
    raw_text: str = ""
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class Stage3Synthesis:
    model: str
    final_probability: Optional[float]
    confidence: Optional[float]
    should_trade: bool
    side: str
    dissent_summary: str
    reasoning: str
    risk_factors: str
    raw_text: str = ""
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class CouncilResult:
    ticker: str
    market_title: str
    stage1_results: list[Stage1Answer]
    stage2_results: list[Stage2Review]
    stage3_result: Stage3Synthesis

    final_probability: Optional[float]
    side: str
    should_trade: bool
    confidence: Optional[float]
    total_cost: float
    all_reasoning: str             # every stage's reasoning concatenated

    council_models: list[str] = field(default_factory=list)
    chairman_model: str = ""


# ----------------------------------------------------------------------
# Small coercion helpers — LLMs return messy types
# ----------------------------------------------------------------------

def _to_float(v, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None or v == "":
            return default
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


def _clamp01(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return max(0.0, min(1.0, v))


def _norm_side(v, default: str = "yes") -> str:
    s = str(v or "").strip().lower()
    if s in ("yes", "y", "buy_yes", "long"):
        return "yes"
    if s in ("no", "n", "buy_no", "short"):
        return "no"
    return default


def _as_text(v) -> str:
    """Stage fields may come back as str, list, or dict — flatten to text."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        return "; ".join(_as_text(x) for x in v if x is not None)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_as_text(val)}" for k, val in v.items())
    return str(v)


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("true", "yes", "1", "y", "trade")


# ----------------------------------------------------------------------
# The council
# ----------------------------------------------------------------------

class WeatherCouncil:
    """
    Run a 3-stage council on a single weather market.

    Usage:
        council = WeatherCouncil()
        result = council.run_council(weather_data, market_data)
    """

    def __init__(
        self,
        council_models: tuple[str, ...] = COUNCIL_MODELS,
        chairman_model: str = CHAIRMAN_MODEL,
    ):
        self.council_models = list(council_models)
        self.chairman_model = chairman_model
        if len(self.council_models) > len(_LABELS):
            raise ValueError(f"at most {len(_LABELS)} council models supported")

    # ------------------------------------------------------------------
    # Context packet — identical input to every Stage-1 model
    # ------------------------------------------------------------------

    def _build_context(self, weather_data: dict, market_data: dict) -> str:
        """Render the weather + market data into one plain-text packet."""

        def fc_line(name: str, fc: Optional[dict]) -> str:
            if not fc:
                return f"  {name}: (unavailable)"
            members = fc.get("members") or []
            sample = ", ".join(str(m) for m in members[:12])
            more = f", … (+{len(members) - 12} more)" if len(members) > 12 else ""
            return (
                f"  {name}: mean={fc.get('mean')}°F  min={fc.get('min')}°F  "
                f"max={fc.get('max')}°F  stdev={fc.get('stdev')}°F  "
                f"n_members={fc.get('n_members')}\n"
                f"      members[°F]: {sample}{more}"
            )

        w = weather_data
        m = market_data

        nws_high = w.get("nws_high")
        nws_line = (
            f"{nws_high}°F"
            + (f" ({w.get('nws_short_forecast')})" if w.get("nws_short_forecast") else "")
            if nws_high is not None
            else "(unavailable)"
        )

        lines = [
            "=== WEATHER FORECAST (NYC Central Park / KNYC) ===",
            f"  Target date (observation day): {w.get('target_date')}",
            fc_line("GFS ensemble", w.get("gfs_forecast")),
            fc_line("ICON ensemble", w.get("icon_forecast")),
            f"  Combined ensemble: mean={w.get('ensemble_mean')}°F  "
            f"spread(stdev)={w.get('ensemble_spread')}°F  "
            f"total_members={w.get('n_members')}  models={w.get('n_models')}",
            f"  NWS OFFICIAL forecast high: {nws_line}",
            "  NOTE: NWS is the LITERAL settlement source for this market.",
            "",
            "=== KALSHI MARKET ===",
            f"  Ticker: {m.get('ticker')}",
            f"  Title: {m.get('title')}",
            f"  Threshold: {m.get('threshold_label')}",
            f"  YES ask (cost to buy YES): ${m.get('yes_price')}",
            f"  NO ask (cost to buy NO):  ${m.get('no_price')}",
            f"  Bid/ask spread: ${m.get('spread')}",
            f"  Market-implied P(YES): {m.get('market_prob')}",
            f"  Volume: {m.get('volume')}",
            f"  Close time: {m.get('close_time')}",
            f"  Hours to settlement: {m.get('hours_to_settlement')}",
            "",
            "=== QUESTION ===",
            f"  Should we buy YES or NO on \"{m.get('title')}\"?",
            "  What is the TRUE probability this market settles YES?",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 1 — independent analysis
    # ------------------------------------------------------------------

    _STAGE1_SYSTEM = (
        "You are a quantitative weather-derivatives analyst serving on a "
        "decision council. You price NWS-settled daily-high temperature "
        "markets on Kalshi. You are given ensemble forecasts (GFS, ICON), "
        "the official NWS forecast (the literal settlement source), and live "
        "market prices. Estimate the TRUE probability the market settles YES, "
        "and decide whether buying YES or NO is the better value at the quoted "
        "price. Reason explicitly about: ensemble spread/uncertainty, how much "
        "to trust NWS vs the raw ensembles, and the market's implied "
        "probability. Be calibrated and concrete. Respond with ONLY a JSON "
        "object, no prose outside it."
    )

    _STAGE1_SCHEMA = (
        'Respond with ONLY this JSON:\n'
        '{\n'
        '  "probability": <float 0..1, your P(market settles YES)>,\n'
        '  "side": "yes" | "no",   // the better BUY at current prices\n'
        '  "confidence": <float 0..1, how sure you are>,\n'
        '  "reasoning": "<your full reasoning chain>"\n'
        '}'
    )

    def _run_stage1(self, context: str) -> list[Stage1Answer]:
        answers: list[Stage1Answer] = []
        user_prompt = f"{context}\n\n{self._STAGE1_SCHEMA}{_CONCISE}"
        for i, model in enumerate(self.council_models):
            label = _LABELS[i]
            try:
                resp = call_llm(
                    model=model,
                    system_prompt=self._STAGE1_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=_TEMPERATURE,
                    max_tokens=_MAX_TOKENS,
                )
                p = resp.parsed
                answers.append(Stage1Answer(
                    model=model,
                    label=label,
                    probability=_clamp01(_to_float(p.get("probability"))),
                    side=_norm_side(p.get("side")),
                    confidence=_clamp01(_to_float(p.get("confidence"))),
                    reasoning=_as_text(p.get("reasoning")),
                    raw_text=resp.raw_text,
                    cost_usd=resp.cost_usd,
                ))
                logger.info("council_stage1", model=model, label=label,
                            prob=answers[-1].probability, side=answers[-1].side)
            except Exception as e:
                logger.warning("council_stage1_failed", model=model, error=str(e)[:200])
                answers.append(Stage1Answer(
                    model=model, label=label, probability=None, side="yes",
                    confidence=None, reasoning="", error=str(e)[:300],
                ))
        return answers

    # ------------------------------------------------------------------
    # Stage 2 — anonymized peer review
    # ------------------------------------------------------------------

    _STAGE2_SYSTEM = (
        "You are on a weather-market analyst council. You have already given "
        "your own independent analysis. Below is the WHOLE panel's analysis, "
        "ANONYMIZED (you cannot tell which one is yours or who wrote any of "
        "them). Critically review the panel's reasoning: where do you agree, "
        "where do you disagree, and does anyone raise a point that should move "
        "your probability? Update your probability if the arguments warrant it "
        "— but do not cave to consensus without a real reason. Respond with "
        "ONLY a JSON object."
    )

    _STAGE2_SCHEMA = (
        'Respond with ONLY this JSON:\n'
        '{\n'
        '  "updated_probability": <float 0..1, your revised P(YES)>,\n'
        '  "agreements": "<points you agree with>",\n'
        '  "disagreements": "<points you disagree with>",\n'
        '  "reasoning": "<why you updated or held your estimate>"\n'
        '}'
    )

    def _format_stage1_panel(self, stage1: list[Stage1Answer]) -> str:
        blocks = []
        for a in stage1:
            if a.error:
                blocks.append(f"{a.label}: (no answer — model error)")
                continue
            blocks.append(
                f"{a.label}:\n"
                f"  P(YES) = {a.probability}  side = {a.side.upper()}  "
                f"confidence = {a.confidence}\n"
                f"  Reasoning: {a.reasoning}"
            )
        return "\n\n".join(blocks)

    def _run_stage2(self, context: str, stage1: list[Stage1Answer]) -> list[Stage2Review]:
        panel = self._format_stage1_panel(stage1)
        reviews: list[Stage2Review] = []
        for i, model in enumerate(self.council_models):
            label = _LABELS[i]
            user_prompt = (
                f"{context}\n\n"
                f"=== PANEL — INDEPENDENT ANALYSES (anonymized) ===\n{panel}\n\n"
                f"{self._STAGE2_SCHEMA}{_CONCISE}"
            )
            try:
                resp = call_llm(
                    model=model,
                    system_prompt=self._STAGE2_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=_TEMPERATURE,
                    max_tokens=_MAX_TOKENS,
                )
                p = resp.parsed
                reviews.append(Stage2Review(
                    model=model,
                    label=label,
                    updated_probability=_clamp01(_to_float(p.get("updated_probability"))),
                    agreements=_as_text(p.get("agreements")),
                    disagreements=_as_text(p.get("disagreements")),
                    reasoning=_as_text(p.get("reasoning")),
                    raw_text=resp.raw_text,
                    cost_usd=resp.cost_usd,
                ))
                logger.info("council_stage2", model=model, label=label,
                            updated_prob=reviews[-1].updated_probability)
            except Exception as e:
                logger.warning("council_stage2_failed", model=model, error=str(e)[:200])
                reviews.append(Stage2Review(
                    model=model, label=label, updated_probability=None,
                    agreements="", disagreements="", reasoning="", error=str(e)[:300],
                ))
        return reviews

    # ------------------------------------------------------------------
    # Stage 3 — chairman synthesis
    # ------------------------------------------------------------------

    _STAGE3_SYSTEM = (
        "You are the CHAIRMAN of a weather-market analyst council. You see the "
        "council's independent analyses (Stage 1) and their anonymized peer "
        "reviews (Stage 2). Synthesize them into one final decision. Weigh the "
        "NWS official forecast heavily — it is the literal settlement source. "
        "Treat disagreement among analysts as a genuine risk signal: a divided "
        "council should lower confidence and the bar for trading. Only set "
        "should_trade=true when the synthesized probability gives a real edge "
        "over the market price AND the council is reasonably aligned. Respond "
        "with ONLY a JSON object."
    )

    _STAGE3_SCHEMA = (
        'Respond with ONLY this JSON:\n'
        '{\n'
        '  "final_probability": <float 0..1, synthesized P(YES)>,\n'
        '  "confidence": <float 0..1>,\n'
        '  "should_trade": true | false,\n'
        '  "side": "yes" | "no",\n'
        '  "dissent_summary": "<where/how the council disagreed>",\n'
        '  "reasoning": "<your synthesis>",\n'
        '  "risk_factors": "<key risks to this decision>"\n'
        '}'
    )

    def _format_stage2_panel(self, stage2: list[Stage2Review]) -> str:
        blocks = []
        for r in stage2:
            if r.error:
                blocks.append(f"{r.label} (review): (no review — model error)")
                continue
            blocks.append(
                f"{r.label} (updated):\n"
                f"  Updated P(YES) = {r.updated_probability}\n"
                f"  Agreements: {r.agreements}\n"
                f"  Disagreements: {r.disagreements}\n"
                f"  Reasoning: {r.reasoning}"
            )
        return "\n\n".join(blocks)

    def _run_stage3(
        self, context: str,
        stage1: list[Stage1Answer], stage2: list[Stage2Review],
    ) -> Stage3Synthesis:
        s1_panel = self._format_stage1_panel(stage1)
        s2_panel = self._format_stage2_panel(stage2)
        user_prompt = (
            f"{context}\n\n"
            f"=== STAGE 1 — INDEPENDENT ANALYSES (anonymized) ===\n{s1_panel}\n\n"
            f"=== STAGE 2 — PEER REVIEWS (anonymized) ===\n{s2_panel}\n\n"
            f"{self._STAGE3_SCHEMA}{_CONCISE}"
        )
        try:
            resp = call_llm(
                model=self.chairman_model,
                system_prompt=self._STAGE3_SYSTEM,
                user_prompt=user_prompt,
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS + 500,
            )
            p = resp.parsed
            return Stage3Synthesis(
                model=self.chairman_model,
                final_probability=_clamp01(_to_float(p.get("final_probability"))),
                confidence=_clamp01(_to_float(p.get("confidence"))),
                should_trade=_truthy(p.get("should_trade")),
                side=_norm_side(p.get("side")),
                dissent_summary=_as_text(p.get("dissent_summary")),
                reasoning=_as_text(p.get("reasoning")),
                risk_factors=_as_text(p.get("risk_factors")),
                raw_text=resp.raw_text,
                cost_usd=resp.cost_usd,
            )
        except Exception as e:
            logger.warning("council_stage3_failed", model=self.chairman_model, error=str(e)[:200])
            return Stage3Synthesis(
                model=self.chairman_model, final_probability=None, confidence=None,
                should_trade=False, side="yes", dissent_summary="", reasoning="",
                risk_factors="", error=str(e)[:300],
            )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_council(self, weather_data: dict, market_data: dict) -> CouncilResult:
        """Run all three stages and assemble the full CouncilResult."""
        context = self._build_context(weather_data, market_data)
        ticker = market_data.get("ticker", "")
        title = market_data.get("title", "")

        logger.info("council_start", ticker=ticker, models=self.council_models,
                    chairman=self.chairman_model)

        stage1 = self._run_stage1(context)
        stage2 = self._run_stage2(context, stage1)
        stage3 = self._run_stage3(context, stage1, stage2)

        total_cost = (
            sum(a.cost_usd for a in stage1)
            + sum(r.cost_usd for r in stage2)
            + stage3.cost_usd
        )

        all_reasoning = self._assemble_reasoning(stage1, stage2, stage3)

        result = CouncilResult(
            ticker=ticker,
            market_title=title,
            stage1_results=stage1,
            stage2_results=stage2,
            stage3_result=stage3,
            final_probability=stage3.final_probability,
            side=stage3.side,
            should_trade=stage3.should_trade,
            confidence=stage3.confidence,
            total_cost=total_cost,
            all_reasoning=all_reasoning,
            council_models=list(self.council_models),
            chairman_model=self.chairman_model,
        )

        logger.info(
            "council_done", ticker=ticker, final_prob=result.final_probability,
            side=result.side, should_trade=result.should_trade,
            confidence=result.confidence, total_cost=round(total_cost, 5),
        )
        return result

    @staticmethod
    def _assemble_reasoning(
        stage1: list[Stage1Answer], stage2: list[Stage2Review],
        stage3: Stage3Synthesis,
    ) -> str:
        parts = ["### STAGE 1 — INDEPENDENT ANALYSIS"]
        for a in stage1:
            parts.append(
                f"[{a.label} / {a.model}] "
                + (f"ERROR: {a.error}" if a.error
                   else f"P(YES)={a.probability} side={a.side} conf={a.confidence}\n{a.reasoning}")
            )
        parts.append("\n### STAGE 2 — PEER REVIEW")
        for r in stage2:
            parts.append(
                f"[{r.label} / {r.model}] "
                + (f"ERROR: {r.error}" if r.error
                   else f"updated P(YES)={r.updated_probability}\n"
                        f"agreements: {r.agreements}\n"
                        f"disagreements: {r.disagreements}\n{r.reasoning}")
            )
        parts.append("\n### STAGE 3 — CHAIRMAN SYNTHESIS")
        parts.append(
            f"[{stage3.model}] "
            + (f"ERROR: {stage3.error}" if stage3.error
               else f"final P(YES)={stage3.final_probability} conf={stage3.confidence} "
                    f"should_trade={stage3.should_trade} side={stage3.side}\n"
                    f"dissent: {stage3.dissent_summary}\n"
                    f"risk_factors: {stage3.risk_factors}\n{stage3.reasoning}")
        )
        return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Persistence — write one council_decisions row (research audit trail)
# ----------------------------------------------------------------------

def persist_council_decision(
    result: CouncilResult,
    market_yes_price: Decimal | float | None,
    market_no_price: Decimal | float | None,
    edge: Decimal | float | None,
    weather_nws_high: Optional[int],
) -> Optional[int]:
    """
    Write the full council run to council_decisions. Returns the new row id,
    or None on failure (logged, never raised — logging must not break trading).

    Stage-1/2 answers are stored POSITIONALLY: stage1_results[0] -> model_a,
    [1] -> model_b, [2] -> model_c, matching the A/B/C labels.
    """
    from data.db import CouncilDecision, get_session

    def s1(idx: int) -> Stage1Answer | None:
        return result.stage1_results[idx] if idx < len(result.stage1_results) else None

    def s2(idx: int) -> Stage2Review | None:
        return result.stage2_results[idx] if idx < len(result.stage2_results) else None

    def _num(v) -> Optional[Decimal]:
        return None if v is None else Decimal(str(v))

    a1, b1, c1 = s1(0), s1(1), s1(2)
    a2, b2, c2 = s2(0), s2(1), s2(2)
    s3 = result.stage3_result

    session = get_session()
    try:
        row = CouncilDecision(
            ticker=result.ticker,
            market_title=result.market_title,
            stage1_model_a_prob=_num(a1.probability) if a1 else None,
            stage1_model_a_side=a1.side if a1 else None,
            stage1_model_a_reasoning=(a1.reasoning or a1.error) if a1 else None,
            stage1_model_b_prob=_num(b1.probability) if b1 else None,
            stage1_model_b_side=b1.side if b1 else None,
            stage1_model_b_reasoning=(b1.reasoning or b1.error) if b1 else None,
            stage1_model_c_prob=_num(c1.probability) if c1 else None,
            stage1_model_c_side=c1.side if c1 else None,
            stage1_model_c_reasoning=(c1.reasoning or c1.error) if c1 else None,
            stage2_model_a_updated_prob=_num(a2.updated_probability) if a2 else None,
            stage2_model_a_reasoning=_stage2_text(a2) if a2 else None,
            stage2_model_b_updated_prob=_num(b2.updated_probability) if b2 else None,
            stage2_model_b_reasoning=_stage2_text(b2) if b2 else None,
            stage2_model_c_updated_prob=_num(c2.updated_probability) if c2 else None,
            stage2_model_c_reasoning=_stage2_text(c2) if c2 else None,
            stage3_final_prob=_num(s3.final_probability),
            stage3_confidence=_num(s3.confidence),
            stage3_should_trade=1 if s3.should_trade else 0,
            stage3_side=s3.side,
            stage3_reasoning=s3.reasoning or s3.error,
            stage3_dissent_summary=s3.dissent_summary,
            stage3_risk_factors=s3.risk_factors,
            market_yes_price=_num(market_yes_price),
            market_no_price=_num(market_no_price),
            edge=_num(edge),
            weather_nws_high=weather_nws_high,
            council_models=result.council_models,
            chairman_model=result.chairman_model,
            total_cost_usd=_num(round(result.total_cost, 6)),
            created_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()
        return row.id
    except Exception as e:
        session.rollback()
        logger.warning("council_persist_failed", ticker=result.ticker, error=str(e)[:200])
        return None
    finally:
        session.close()


def _stage2_text(r: Stage2Review) -> str:
    """Fold a Stage-2 review's structured fields into one stored reasoning blob."""
    if r.error:
        return r.error
    return (
        f"AGREEMENTS: {r.agreements}\n"
        f"DISAGREEMENTS: {r.disagreements}\n"
        f"REASONING: {r.reasoning}"
    )
