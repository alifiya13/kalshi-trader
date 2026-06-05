#!/usr/bin/env python3
"""
Council smoke test — run the 3-stage WeatherCouncil on ONE real KXHIGHNY
market and print the FULL output: every stage, every model's reasoning, the
final decision.

This DOES NOT TRADE and does NOT write to the database. It only reads live
market data + forecasts and runs the LLMs.

Run:
  python -m scripts.council_test                 # auto-pick best edge market
  python -m scripts.council_test KXHIGHNY-...    # a specific ticker
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_key_files
ensure_key_files()

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from core.rest_client import get_data_client
from strategies.weather import find_weather_edge, get_weather_context, parse_ticker_date
from agents.council import WeatherCouncil

console = Console(width=120)


def _build_market_data(sig, raw: dict | None) -> dict:
    yes_bid = sig.market_yes_bid
    yes_ask = sig.market_yes_ask
    yes_price = yes_ask if yes_ask > 0 else sig.market_prob
    no_price = (Decimal("1") - yes_bid) if yes_bid > 0 else (Decimal("1") - yes_price)
    spread = (yes_ask - yes_bid) if (yes_ask > 0 and yes_bid > 0) else None
    raw = raw or {}
    return {
        "ticker": sig.ticker,
        "title": raw.get("title") or sig.title,
        "threshold_label": sig.threshold_label,
        "yes_price": yes_price,
        "no_price": no_price,
        "spread": spread,
        "market_prob": float(sig.market_prob),
        "volume": raw.get("volume"),
        "close_time": raw.get("close_time"),
        "hours_to_settlement": None,
    }


def main() -> int:
    want_ticker = sys.argv[1] if len(sys.argv) > 1 else None

    console.print(Rule("[bold magenta]WeatherCouncil — live smoke test (NO TRADE)"))
    client = get_data_client()

    # Raw open markets, for volume/close_time + ticker lookup.
    resp = client.get_markets(series_ticker="KXHIGHNY", status="open", limit=200)
    raw_by_ticker = {m.get("ticker"): m for m in resp.get("markets", [])}

    # Score every market; pick the highest-edge one (or the requested ticker).
    signals = find_weather_edge(client, min_edge=Decimal("0"))
    if not signals:
        console.print("[red]No KXHIGHNY signals scored (no target-date markets or no ensemble). "
                      "Try again closer to a trading day.[/]")
        return 1

    if want_ticker:
        sig = next((s for s in signals if s.ticker == want_ticker), None)
        if sig is None:
            console.print(f"[red]Ticker {want_ticker} not found among scored signals.[/]")
            console.print("Available:", ", ".join(s.ticker for s in signals[:20]))
            return 1
    else:
        # Highest signed edge — the market the weather model likes most.
        sig = signals[0]

    console.print(f"[bold]Selected market:[/] {sig.ticker}")
    console.print(
        f"  {sig.title}  ({sig.threshold_label})\n"
        f"  weather model_prob={float(sig.model_prob):.3f}  market_prob={float(sig.market_prob):.3f}  "
        f"edge={float(sig.edge):.3f}  side={sig.side.upper()}  confidence={sig.confidence}\n"
        f"  NWS high={sig.nws_forecast_high}°F  ensemble_mean={sig.ensemble_mean_f}°F  "
        f"per-model={sig.per_model_probs}"
    )

    target_date = parse_ticker_date(sig.ticker)
    console.print(f"\n[dim]Building weather context for {target_date} …[/]")
    weather_ctx = get_weather_context(target_date=target_date)
    console.print(
        f"[dim]  GFS={weather_ctx['gfs_forecast'] and weather_ctx['gfs_forecast']['mean']}°F  "
        f"ICON={weather_ctx['icon_forecast'] and weather_ctx['icon_forecast']['mean']}°F  "
        f"NWS={weather_ctx['nws_high']}°F  ensemble_mean={weather_ctx['ensemble_mean']}°F  "
        f"spread={weather_ctx['ensemble_spread']}°F  members={weather_ctx['n_members']}[/]")

    market_data = _build_market_data(sig, raw_by_ticker.get(sig.ticker))

    council = WeatherCouncil()
    console.print(f"\n[dim]Council: {council.council_models}  chairman: {council.chairman_model}[/]")
    console.print(Rule("[bold cyan]Running council (7 LLM calls)…"))

    result = council.run_council(weather_ctx, market_data)

    # ---- STAGE 1 ----
    console.print(Rule("[bold]STAGE 1 — Independent Analysis"))
    for a in result.stage1_results:
        if a.error:
            console.print(Panel(f"[red]ERROR: {a.error}[/]",
                                title=f"{a.label} — {a.model}", border_style="red"))
            continue
        console.print(Panel(
            f"[bold]P(YES) = {a.probability}[/]   side = [bold]{a.side.upper()}[/]   "
            f"confidence = {a.confidence}   cost = ${a.cost_usd:.5f}\n\n"
            f"{a.reasoning}",
            title=f"{a.label} — {a.model}", border_style="cyan"))

    # ---- STAGE 2 ----
    console.print(Rule("[bold]STAGE 2 — Peer Review (anonymized)"))
    for r in result.stage2_results:
        if r.error:
            console.print(Panel(f"[red]ERROR: {r.error}[/]",
                                title=f"{r.label} — {r.model}", border_style="red"))
            continue
        console.print(Panel(
            f"[bold]Updated P(YES) = {r.updated_probability}[/]   cost = ${r.cost_usd:.5f}\n\n"
            f"[green]Agreements:[/] {r.agreements}\n\n"
            f"[yellow]Disagreements:[/] {r.disagreements}\n\n"
            f"[dim]Reasoning:[/] {r.reasoning}",
            title=f"{r.label} — {r.model}", border_style="magenta"))

    # ---- STAGE 3 ----
    console.print(Rule("[bold]STAGE 3 — Chairman Synthesis"))
    s3 = result.stage3_result
    if s3.error:
        console.print(Panel(f"[red]ERROR: {s3.error}[/]",
                            title=f"Chairman — {s3.model}", border_style="red"))
    else:
        console.print(Panel(
            f"[bold]Final P(YES) = {s3.final_probability}[/]   side = [bold]{s3.side.upper()}[/]\n"
            f"confidence = {s3.confidence}   should_trade = "
            f"[bold]{s3.should_trade}[/]   cost = ${s3.cost_usd:.5f}\n\n"
            f"[bold]Reasoning:[/]\n{s3.reasoning}\n\n"
            f"[yellow]Dissent summary:[/] {s3.dissent_summary}\n\n"
            f"[red]Risk factors:[/] {s3.risk_factors}",
            title=f"Chairman — {s3.model}", border_style="green"))

    # ---- FINAL DECISION + combined gate (printed, NOT acted on) ----
    console.print(Rule("[bold]FINAL DECISION"))
    conf = result.confidence or 0.0
    edge_ok = sig.edge >= Decimal("0.15")
    trade_ok = result.should_trade
    conf_ok = conf > 0.6
    would_trade = edge_ok and trade_ok and conf_ok
    console.print(
        f"  weather edge ≥ 15¢ : {edge_ok}  (edge={float(sig.edge):.3f})\n"
        f"  council should_trade: {trade_ok}\n"
        f"  council conf > 0.6  : {conf_ok}  (conf={conf:.2f})\n"
        f"  [bold]→ combined gate: {'WOULD TRADE' if would_trade else 'NO TRADE'}[/]  "
        f"(side={result.side.upper()}, final_prob={result.final_probability})"
    )
    console.print(f"\n  [bold]Total council cost: ${result.total_cost:.5f}[/]")
    console.print("\n[bold yellow]TEST ONLY — no trade placed, nothing written to the database.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
