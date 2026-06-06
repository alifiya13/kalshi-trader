"""
WeatherCouncil — a 3-stage LLM council for weather-EVENT decisions.

Adapted from Andrej Karpathy's llm-council (github.com/karpathy/llm-council).
We extract the PATTERN, not the code: three stages, anonymized peer review,
a chairman that synthesizes. The plumbing (LLM Gateway client, JSON
extraction, retry/cost logging) is our existing agents/base_agent.py.

Event-level model (2026-06-06): the council runs ONCE per weather event and
sees ALL brackets (B-bands + T-tails) with their YES/NO prices together. It
works for ANY city and either daily extreme — the event carries its own
city/temp_type ("high"/"low") from data/weather_discovery.py. The council
predicts the temperature and selects which brackets to trade. The chairman
MUST select at least one trade — this is a research study measuring council
accuracy, so "skip" is not a decision we can score.

    STAGE 1 — Independent Analysis
        Each council model receives the SAME context packet (ensemble +
        NWS forecast + the FULL bracket table) and answers independently:
        {predicted_temp_f, confidence, trades: [{ticker, side, reasoning}]}.

    STAGE 2 — Peer Review
        Each model sees the WHOLE panel's Stage-1 answers, anonymized as
        "Model A / Model B / Model C" (it can't tell which is its own or
        who wrote what), and may update its prediction + trade selections:
        {updated_predicted_temp_f, updated_trades, agreements, disagreements}.

    STAGE 3 — Chairman Synthesis
        One stronger model sees every Stage-1 answer + every Stage-2 review
        and produces the final call — at least one trade, each with a
        win probability:
        {predicted_temp_f, confidence, trades: [{ticker, side, probability,
         reasoning}], dissent_summary, risk_factors, overall_reasoning}.

Design note: the A/B/C labels are STABLE across stages and map positionally
to `council_models`, so the council_decisions table can store per-model
columns cleanly. This is a research instrument about *council failure
modes* — so we keep every model's full reasoning chain, every stage, even
when a model errors out. Nothing is silently dropped.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
# spend tokens on internal "thinking" before the JSON, and a truncated
# response fails to parse — costing us that model's whole vote. We also ask
# models to keep reasoning concise (see prompts).
_TEMPERATURE = 0.3
_MAX_TOKENS = 2800

# Appended to every stage's instructions: keeps reasoning from ballooning
# past the token budget (which truncates the JSON and loses the vote).
_CONCISE = (
    "\nKeep each reasoning field concise — a few focused sentences, not an "
    "essay. Output the JSON object and nothing else."
)


# ----------------------------------------------------------------------
# Result structures
# ----------------------------------------------------------------------

@dataclass
class TradeCall:
    """One bracket trade named by a council member or the chairman."""
    ticker: str
    side: str                          # "yes" | "no"
    reasoning: str
    probability: Optional[float] = None  # P(this trade WINS) — chairman only


@dataclass
class Stage1Answer:
    model: str
    label: str
    predicted_temp_f: Optional[float]  # the model's daily high/low prediction
    confidence: Optional[float]        # 0..1
    trades: list[TradeCall] = field(default_factory=list)
    raw_text: str = ""
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class Stage2Review:
    model: str
    label: str
    updated_predicted_temp_f: Optional[float]
    updated_trades: list[TradeCall] = field(default_factory=list)
    agreements: str = ""
    disagreements: str = ""
    raw_text: str = ""
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class Stage3Synthesis:
    model: str
    predicted_temp_f: Optional[float]
    confidence: Optional[float]
    trades: list[TradeCall] = field(default_factory=list)  # ≥1, each with probability
    dissent_summary: str = ""
    risk_factors: str = ""
    overall_reasoning: str = ""
    raw_text: str = ""
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class CouncilEventResult:
    """Full record of one council run over an entire weather event."""
    event_date: str                    # ISO date of the observation day
    series_ticker: str
    stage1_results: list[Stage1Answer]
    stage2_results: list[Stage2Review]
    stage3_result: Stage3Synthesis

    # Chairman's final call, surfaced for convenience
    predicted_temp_f: Optional[float]
    confidence: Optional[float]
    trades: list[TradeCall]            # the trades to paper-trade (≥1)
    total_cost: float
    all_reasoning: str                 # every stage's reasoning concatenated

    council_models: list[str] = field(default_factory=list)
    chairman_model: str = ""


# ----------------------------------------------------------------------
# Small coercion helpers — LLMs return messy types
# ----------------------------------------------------------------------

def _to_float(v, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None or v == "":
            return default
        return float(v)
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


def _parse_trades(raw, valid_tickers: set[str], with_probability: bool = False) -> list[TradeCall]:
    """
    Coerce a model's `trades` list into TradeCall objects. Unknown tickers
    (hallucinated or mangled) are dropped with a log line; duplicates keep
    the first occurrence.
    """
    out: list[TradeCall] = []
    seen: set[str] = set()
    if not isinstance(raw, (list, tuple)):
        return out
    for t in raw:
        if not isinstance(t, dict):
            continue
        ticker = str(t.get("ticker") or "").strip().upper()
        if ticker not in valid_tickers:
            logger.warning("council_unknown_ticker", ticker=ticker)
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        out.append(TradeCall(
            ticker=ticker,
            side=_norm_side(t.get("side")),
            reasoning=_as_text(t.get("reasoning")),
            probability=_clamp01(_to_float(t.get("probability"))) if with_probability else None,
        ))
    return out


def _trades_text(trades: list[TradeCall]) -> str:
    """Render a trade list as one readable line-per-trade block."""
    if not trades:
        return "(no trades named)"
    lines = []
    for t in trades:
        prob = f" P(win)={t.probability}" if t.probability is not None else ""
        lines.append(f"- {t.ticker} {t.side.upper()}{prob} — {t.reasoning}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# The council
# ----------------------------------------------------------------------

class WeatherCouncil:
    """
    Run a 3-stage council on a whole weather EVENT (all brackets at once).

    Usage:
        council = WeatherCouncil()
        result = council.run_council(weather_data, event_data)

    `event_data` is WeatherEvent.as_council_event() from
    data/weather_discovery.py:
        {event_ticker, event_date, series_ticker, city, temp_type,
         brackets: [{ticker, threshold, type, yes_price, no_price,
         volume, ...}, ...]}
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

    @staticmethod
    def _bracket_table(event_data: dict) -> str:
        """Render every bracket of the event as one aligned table."""
        ev_date = event_data.get("event_date")
        if isinstance(ev_date, date):
            date_label = f"{ev_date:%b} {ev_date.day}, {ev_date.year}"
        else:
            date_label = str(ev_date)
        city = event_data.get("city") or "the city"
        kind = "Low" if event_data.get("temp_type") == "low" else "High"

        lines = [f"Available brackets for {city} {kind} Temperature on {date_label}:"]
        for b in event_data.get("brackets", []):
            suffix = b["ticker"].rsplit("-", 1)[-1]
            lines.append(
                f"  {suffix:<7} ({b['threshold']}F): "
                f"YES ${float(b['yes_price']):.2f} / NO ${float(b['no_price']):.2f}"
                f"   volume={b.get('volume', 0)}   ticker={b['ticker']}"
            )
        return "\n".join(lines)

    def _build_context(self, weather_data: dict, event_data: dict) -> str:
        """Render the weather + full-event market data into one text packet."""

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
        city = event_data.get("city") or w.get("city") or "the city"
        kind = "low" if event_data.get("temp_type") == "low" else "high"

        nws_temp = w.get("nws_temp")
        nws_line = (
            f"{nws_temp}°F"
            + (f" ({w.get('nws_short_forecast')})" if w.get("nws_short_forecast") else "")
            if nws_temp is not None
            else "(unavailable)"
        )

        lines = [
            f"=== WEATHER FORECAST ({city}) ===",
            f"  Target date (observation day): {w.get('target_date')}",
            f"  Variable: daily {kind.upper()} temperature",
            fc_line(f"GFS ensemble (per-member daily {kind})", w.get("gfs_forecast")),
            fc_line(f"ICON ensemble (per-member daily {kind})", w.get("icon_forecast")),
            f"  Combined ensemble: mean={w.get('ensemble_mean')}°F  "
            f"spread(stdev)={w.get('ensemble_spread')}°F  "
            f"total_members={w.get('n_members')}  models={w.get('n_models')}",
            f"  NWS OFFICIAL forecast {kind}: {nws_line}",
            "  NOTE: NWS is the LITERAL settlement source for these markets.",
            "",
            "=== KALSHI EVENT — ALL BRACKETS ===",
            self._bracket_table(event_data),
            "",
            "  The brackets are mutually exclusive: exactly ONE settles YES",
            f"  (the one containing the observed NWS daily {kind}, integer °F).",
            "  In your trades, use the full ticker exactly as given above.",
            "",
            "=== QUESTION ===",
            "  Given the weather data and these market prices, what temperature",
            f"  do you predict for the {city} {kind}? Which bracket(s) would you",
            "  trade? For each bracket you'd trade, specify: buy YES or NO,",
            "  and why.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 1 — independent analysis
    # ------------------------------------------------------------------

    _STAGE1_SYSTEM = (
        "You are a quantitative weather-derivatives analyst serving on a "
        "decision council. You price NWS-settled daily temperature (high or "
        "low) markets on Kalshi. You are given ensemble forecasts (GFS, "
        "ICON), the official NWS forecast (the literal settlement source), "
        "and the FULL set of temperature brackets for one event with live "
        "YES/NO prices. First predict the day's temperature extreme the "
        "event asks about, then pick which brackets are mispriced relative "
        "to your prediction and the forecast uncertainty. Reason explicitly "
        "about: ensemble spread, how much to trust NWS vs the raw "
        "ensembles, and what each bracket's price implies. Be calibrated "
        "and concrete. Respond with ONLY a JSON object, no prose outside it."
    )

    _STAGE1_SCHEMA = (
        'Respond with ONLY this JSON:\n'
        '{\n'
        '  "predicted_temp_f": <float, your predicted temperature in °F (the daily high or low the event asks about)>,\n'
        '  "confidence": <float 0..1, how sure you are of the prediction>,\n'
        '  "trades": [\n'
        '    {"ticker": "<full bracket ticker>", "side": "yes" | "no",\n'
        '     "reasoning": "<why this bracket at this price>"},\n'
        '    ...  // the bracket(s) you would trade, usually 1-3\n'
        '  ]\n'
        '}'
    )

    def _run_stage1(self, context: str, valid_tickers: set[str]) -> list[Stage1Answer]:
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
                    predicted_temp_f=_to_float(p.get("predicted_temp_f")),
                    confidence=_clamp01(_to_float(p.get("confidence"))),
                    trades=_parse_trades(p.get("trades"), valid_tickers),
                    raw_text=resp.raw_text,
                    cost_usd=resp.cost_usd,
                ))
                logger.info("council_stage1", model=model, label=label,
                            predicted_temp=answers[-1].predicted_temp_f,
                            n_trades=len(answers[-1].trades))
            except Exception as e:
                logger.warning("council_stage1_failed", model=model, error=str(e)[:200])
                answers.append(Stage1Answer(
                    model=model, label=label, predicted_temp_f=None,
                    confidence=None, error=str(e)[:300],
                ))
        return answers

    # ------------------------------------------------------------------
    # Stage 2 — anonymized peer review
    # ------------------------------------------------------------------

    _STAGE2_SYSTEM = (
        "You are on a weather-market analyst council. You have already given "
        "your own independent analysis. Below is the WHOLE panel's analysis, "
        "ANONYMIZED (you cannot tell which one is yours or who wrote any of "
        "them) — each member's predicted temperature and the bracket "
        "trades they selected. Critically review the panel: where do you "
        "agree, where do you disagree, and does anyone raise a point that "
        "should move your prediction or change which brackets you'd trade? "
        "Update if the arguments warrant it — but do not cave to consensus "
        "without a real reason. Respond with ONLY a JSON object."
    )

    _STAGE2_SCHEMA = (
        'Respond with ONLY this JSON:\n'
        '{\n'
        '  "updated_predicted_temp_f": <float, your revised predicted temperature °F>,\n'
        '  "updated_trades": [\n'
        '    {"ticker": "<full bracket ticker>", "side": "yes" | "no",\n'
        '     "reasoning": "<why>"},\n'
        '    ...\n'
        '  ],\n'
        '  "agreements": "<points you agree with>",\n'
        '  "disagreements": "<points you disagree with>"\n'
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
                f"  Predicted high = {a.predicted_temp_f}°F  "
                f"confidence = {a.confidence}\n"
                f"  Trades:\n{_trades_text(a.trades)}"
            )
        return "\n\n".join(blocks)

    def _run_stage2(
        self, context: str, stage1: list[Stage1Answer], valid_tickers: set[str],
    ) -> list[Stage2Review]:
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
                    updated_predicted_temp_f=_to_float(p.get("updated_predicted_temp_f")),
                    updated_trades=_parse_trades(p.get("updated_trades"), valid_tickers),
                    agreements=_as_text(p.get("agreements")),
                    disagreements=_as_text(p.get("disagreements")),
                    raw_text=resp.raw_text,
                    cost_usd=resp.cost_usd,
                ))
                logger.info("council_stage2", model=model, label=label,
                            updated_temp=reviews[-1].updated_predicted_temp_f,
                            n_trades=len(reviews[-1].updated_trades))
            except Exception as e:
                logger.warning("council_stage2_failed", model=model, error=str(e)[:200])
                reviews.append(Stage2Review(
                    model=model, label=label, updated_predicted_temp_f=None,
                    error=str(e)[:300],
                ))
        return reviews

    # ------------------------------------------------------------------
    # Stage 3 — chairman synthesis
    # ------------------------------------------------------------------

    _STAGE3_SYSTEM = (
        "You are the CHAIRMAN of a weather-market analyst council. You see "
        "the council's independent analyses (Stage 1) and their anonymized "
        "peer reviews (Stage 2). Synthesize them into one final decision: a "
        "predicted temperature and the bracket trades to place. Weigh "
        "the NWS official forecast heavily — it is the literal settlement "
        "source. Treat disagreement among analysts as a genuine risk signal "
        "and fold it into each trade's win probability. You MUST select at "
        "least one bracket to trade. This is a research study measuring "
        "council accuracy — skipping is not an option. If nothing looks "
        "mispriced, pick the trade(s) you most believe in at the quoted "
        "prices. Respond with ONLY a JSON object."
    )

    _STAGE3_SCHEMA = (
        'Respond with ONLY this JSON:\n'
        '{\n'
        '  "predicted_temp_f": <float, final predicted temperature °F (the daily high or low the event asks about)>,\n'
        '  "confidence": <float 0..1>,\n'
        '  "trades": [\n'
        '    {"ticker": "<full bracket ticker>", "side": "yes" | "no",\n'
        '     "probability": <float 0..1, P(this trade WINS — the side you chose settles correct)>,\n'
        '     "reasoning": "<why this trade>"},\n'
        '    ...  // AT LEAST ONE trade — skipping is not allowed\n'
        '  ],\n'
        '  "dissent_summary": "<where/how the council disagreed>",\n'
        '  "risk_factors": "<key risks to this decision>",\n'
        '  "overall_reasoning": "<your synthesis>"\n'
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
                f"  Updated predicted high = {r.updated_predicted_temp_f}°F\n"
                f"  Updated trades:\n{_trades_text(r.updated_trades)}\n"
                f"  Agreements: {r.agreements}\n"
                f"  Disagreements: {r.disagreements}"
            )
        return "\n\n".join(blocks)

    def _run_stage3(
        self, context: str,
        stage1: list[Stage1Answer], stage2: list[Stage2Review],
        valid_tickers: set[str],
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
                predicted_temp_f=_to_float(p.get("predicted_temp_f")),
                confidence=_clamp01(_to_float(p.get("confidence"))),
                trades=_parse_trades(p.get("trades"), valid_tickers, with_probability=True),
                dissent_summary=_as_text(p.get("dissent_summary")),
                risk_factors=_as_text(p.get("risk_factors")),
                overall_reasoning=_as_text(p.get("overall_reasoning")),
                raw_text=resp.raw_text,
                cost_usd=resp.cost_usd,
            )
        except Exception as e:
            logger.warning("council_stage3_failed", model=self.chairman_model, error=str(e)[:200])
            return Stage3Synthesis(
                model=self.chairman_model, predicted_temp_f=None, confidence=None,
                error=str(e)[:300],
            )

    # ------------------------------------------------------------------
    # Mandatory-trade backstop
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_trade(stage3: Stage3Synthesis, event_data: dict) -> Optional[TradeCall]:
        """
        The chairman must always trade. If it returned zero VALID trades
        (refused, hallucinated tickers, or errored), synthesize one
        deterministically: buy YES on the bracket containing (or nearest to)
        the best available predicted temperature. Marked as a fallback in the
        reasoning so research analysis can separate these rows.
        """
        brackets = event_data.get("brackets", [])
        if not brackets:
            return None

        predicted = stage3.predicted_temp_f

        def _dist(b: dict) -> float:
            if predicted is None:
                return 0.0
            lo, hi = b.get("floor_strike"), b.get("cap_strike")
            if b["type"] == "band":
                if float(lo) <= predicted <= float(hi):
                    return 0.0
                return min(abs(predicted - float(lo)), abs(predicted - float(hi)))
            if b["type"] == "above":
                return 0.0 if predicted > float(lo) else float(lo) - predicted
            return 0.0 if predicted < float(hi) else predicted - float(hi)  # below

        target = min(brackets, key=_dist) if predicted is not None else max(
            brackets, key=lambda b: float(b["market_prob"]))
        return TradeCall(
            ticker=target["ticker"],
            side="yes",
            probability=None,
            reasoning=(
                "FALLBACK: chairman returned no valid trade "
                f"({'error: ' + stage3.error if stage3.error else 'empty/invalid trades list'}); "
                f"auto-selected the bracket nearest predicted temperature "
                f"{predicted}°F."
            ),
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_council(self, weather_data: dict, event_data: dict) -> CouncilEventResult:
        """Run all three stages on the whole event and assemble the result."""
        context = self._build_context(weather_data, event_data)
        valid_tickers = {b["ticker"] for b in event_data.get("brackets", [])}
        event_date = event_data.get("event_date")
        event_date_iso = event_date.isoformat() if isinstance(event_date, date) else str(event_date)

        logger.info("council_start", event_date=event_date_iso,
                    n_brackets=len(valid_tickers), models=self.council_models,
                    chairman=self.chairman_model)

        stage1 = self._run_stage1(context, valid_tickers)
        stage2 = self._run_stage2(context, stage1, valid_tickers)
        stage3 = self._run_stage3(context, stage1, stage2, valid_tickers)

        # Mandatory-trade backstop: the council ALWAYS produces ≥1 trade.
        if not stage3.trades:
            fb = self._fallback_trade(stage3, event_data)
            if fb:
                logger.warning("council_fallback_trade", ticker=fb.ticker)
                stage3.trades = [fb]

        total_cost = (
            sum(a.cost_usd for a in stage1)
            + sum(r.cost_usd for r in stage2)
            + stage3.cost_usd
        )

        all_reasoning = self._assemble_reasoning(stage1, stage2, stage3)

        result = CouncilEventResult(
            event_date=event_date_iso,
            series_ticker=event_data.get("series_ticker", ""),
            stage1_results=stage1,
            stage2_results=stage2,
            stage3_result=stage3,
            predicted_temp_f=stage3.predicted_temp_f,
            confidence=stage3.confidence,
            trades=stage3.trades,
            total_cost=total_cost,
            all_reasoning=all_reasoning,
            council_models=list(self.council_models),
            chairman_model=self.chairman_model,
        )

        logger.info(
            "council_done", event_date=event_date_iso,
            predicted_temp=result.predicted_temp_f,
            n_trades=len(result.trades),
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
                   else f"predicted_temp={a.predicted_temp_f}°F conf={a.confidence}\n"
                        f"{_trades_text(a.trades)}")
            )
        parts.append("\n### STAGE 2 — PEER REVIEW")
        for r in stage2:
            parts.append(
                f"[{r.label} / {r.model}] "
                + (f"ERROR: {r.error}" if r.error
                   else f"updated_temp={r.updated_predicted_temp_f}°F\n"
                        f"{_trades_text(r.updated_trades)}\n"
                        f"agreements: {r.agreements}\n"
                        f"disagreements: {r.disagreements}")
            )
        parts.append("\n### STAGE 3 — CHAIRMAN SYNTHESIS")
        parts.append(
            f"[{stage3.model}] "
            + (f"ERROR: {stage3.error}" if stage3.error
               else f"final predicted_temp={stage3.predicted_temp_f}°F conf={stage3.confidence}\n"
                    f"{_trades_text(stage3.trades)}\n"
                    f"dissent: {stage3.dissent_summary}\n"
                    f"risk_factors: {stage3.risk_factors}\n{stage3.overall_reasoning}")
        )
        return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Persistence — one council_decisions row PER TRADE (research audit trail)
# ----------------------------------------------------------------------

def _stage1_blob(a: Stage1Answer) -> str:
    if a.error:
        return a.error
    return (
        f"PREDICTED TEMP: {a.predicted_temp_f}°F (confidence {a.confidence})\n"
        f"TRADES:\n{_trades_text(a.trades)}"
    )


def _stage2_blob(r: Stage2Review) -> str:
    if r.error:
        return r.error
    return (
        f"UPDATED PREDICTED TEMP: {r.updated_predicted_temp_f}°F\n"
        f"UPDATED TRADES:\n{_trades_text(r.updated_trades)}\n"
        f"AGREEMENTS: {r.agreements}\n"
        f"DISAGREEMENTS: {r.disagreements}"
    )


def persist_event_decision(
    result: CouncilEventResult,
    event_data: dict,
    weather_nws_high: Optional[int],
) -> list[int]:
    """
    Write one council_decisions row per chairman trade, all sharing a fresh
    council_run_id (UUID). Stage-1/2/3 reasoning is duplicated across the
    run's rows; per-trade fields (ticker, side, P(win), trade reasoning,
    prices, edge) differ per row.

    Returns the new row ids ([] on failure — logged, never raised; logging
    must not break the research loop).
    """
    from data.db import CouncilDecision, get_session

    def _num(v) -> Optional[Decimal]:
        return None if v is None else Decimal(str(v))

    def sN(items: list, idx: int):
        return items[idx] if idx < len(items) else None

    a1, b1, c1 = sN(result.stage1_results, 0), sN(result.stage1_results, 1), sN(result.stage1_results, 2)
    a2, b2, c2 = sN(result.stage2_results, 0), sN(result.stage2_results, 1), sN(result.stage2_results, 2)
    s3 = result.stage3_result

    bracket_by_ticker = {b["ticker"]: b for b in event_data.get("brackets", [])}
    run_id = str(uuid.uuid4())
    row_ids: list[int] = []

    session = get_session()
    try:
        for trade in result.trades:
            bracket = bracket_by_ticker.get(trade.ticker, {})
            yes_price = bracket.get("yes_price")
            no_price = bracket.get("no_price")

            # Edge = P(win) − entry cost on the chosen side. Descriptive only.
            edge = None
            if trade.probability is not None:
                entry = yes_price if trade.side == "yes" else no_price
                if entry is not None:
                    edge = Decimal(str(trade.probability)) - Decimal(str(entry))

            row = CouncilDecision(
                council_run_id=run_id,
                event_ticker=event_data.get("event_ticker"),
                city=event_data.get("city"),
                temp_type=event_data.get("temp_type"),
                ticker=trade.ticker,
                market_title=bracket.get("title") or bracket.get("threshold") or trade.ticker,
                predicted_temp_f=_num(result.predicted_temp_f),
                stage1_model_a_predicted_temp=_num(a1.predicted_temp_f) if a1 else None,
                stage1_model_b_predicted_temp=_num(b1.predicted_temp_f) if b1 else None,
                stage1_model_c_predicted_temp=_num(c1.predicted_temp_f) if c1 else None,
                stage2_model_a_updated_temp=_num(a2.updated_predicted_temp_f) if a2 else None,
                stage2_model_b_updated_temp=_num(b2.updated_predicted_temp_f) if b2 else None,
                stage2_model_c_updated_temp=_num(c2.updated_predicted_temp_f) if c2 else None,
                stage1_model_a_reasoning=_stage1_blob(a1) if a1 else None,
                stage1_model_b_reasoning=_stage1_blob(b1) if b1 else None,
                stage1_model_c_reasoning=_stage1_blob(c1) if c1 else None,
                stage2_model_a_reasoning=_stage2_blob(a2) if a2 else None,
                stage2_model_b_reasoning=_stage2_blob(b2) if b2 else None,
                stage2_model_c_reasoning=_stage2_blob(c2) if c2 else None,
                stage3_final_prob=_num(trade.probability),
                stage3_confidence=_num(s3.confidence),
                stage3_should_trade=1,
                stage3_side=trade.side,
                stage3_reasoning=s3.overall_reasoning or s3.error,
                stage3_dissent_summary=s3.dissent_summary,
                stage3_risk_factors=s3.risk_factors,
                trade_reasoning=trade.reasoning,
                market_yes_price=_num(yes_price),
                market_no_price=_num(no_price),
                edge=edge,
                weather_nws_high=weather_nws_high,
                council_models=result.council_models,
                chairman_model=result.chairman_model,
                total_cost_usd=_num(round(result.total_cost, 6)),
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.flush()
            row_ids.append(row.id)
        session.commit()
        return row_ids
    except Exception as e:
        session.rollback()
        logger.warning("council_persist_failed", run_id=run_id, error=str(e)[:200])
        return []
    finally:
        session.close()
