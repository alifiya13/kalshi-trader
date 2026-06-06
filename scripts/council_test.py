#!/usr/bin/env python3
"""
Council event test — run the 3-stage WeatherCouncil ONCE on one
discovered weather event (ALL brackets at once) and print the FULL output: the
bracket table the council saw, every model's predicted temperature, every
stage's reasoning, the chairman's final trades.

This WRITES to the database — one council_decisions row per chairman trade
(grouped by council_run_id) and one paper position per trade — because the
research study requires every council decision to be measurable.

Run:
  python -m scripts.council_test                      # first discovered event
  python -m scripts.council_test KXHIGHCHI-26JUN07    # a specific event
  python -m scripts.council_test --no-db    # print only, write nothing
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_key_files
ensure_key_files()

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from core.rest_client import get_data_client
from data.db import Position, get_session, init_db
from data.weather_discovery import discover_weather_events
from strategies.weather import get_weather_context
from agents.council import WeatherCouncil, persist_event_decision

console = Console(width=120)

PAPER_CONTRACTS = 10  # mirror active_trader's fixed research size


def main() -> int:
    write_db = "--no-db" not in sys.argv
    want = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    console.print(Rule("[bold magenta]WeatherCouncil — event-level test"))
    client = get_data_client()
    init_db()

    # --- Discover all weather events; pick one temperature event ---
    console.print("[dim]Discovering weather events…[/]")
    events = [ev for ev in discover_weather_events(client) if ev.temp_type]
    if not events:
        console.print("[red]No temperature events discovered inside the deadline.[/]")
        return 1
    if want:
        ev = next((e for e in events if e.event_ticker == want), None)
        if ev is None:
            console.print(f"[red]{want} not found.[/] Available: "
                          + ", ".join(e.event_ticker for e in events[:30]))
            return 1
    else:
        ev = events[0]

    event = ev.as_council_event()
    console.print(f"\n[bold]Event:[/] {ev.event_ticker} — {ev.city} {ev.temp_type} "
                  f"on {ev.event_date}  (lat={ev.lat:.3f} lon={ev.lon:.3f})")
    console.print(Panel(WeatherCouncil._bracket_table(event),
                        title="Bracket table (as the council sees it)",
                        border_style="blue"))

    # --- Weather context ---
    console.print(f"[dim]Building weather context for {ev.city} {ev.event_date} …[/]")
    weather_ctx = get_weather_context(
        latitude=ev.lat, longitude=ev.lon,
        target_date=ev.event_date, temp_type=ev.temp_type, city=ev.city,
    )
    console.print(
        f"[dim]  GFS={weather_ctx['gfs_forecast'] and weather_ctx['gfs_forecast']['mean']}°F  "
        f"ICON={weather_ctx['icon_forecast'] and weather_ctx['icon_forecast']['mean']}°F  "
        f"NWS={weather_ctx['nws_temp']}°F  ensemble_mean={weather_ctx['ensemble_mean']}°F  "
        f"spread={weather_ctx['ensemble_spread']}°F  members={weather_ctx['n_members']}[/]")

    council = WeatherCouncil()
    console.print(f"\n[dim]Council: {council.council_models}  chairman: {council.chairman_model}[/]")
    console.print(Rule("[bold cyan]Running council (7 LLM calls)…"))

    result = council.run_council(weather_ctx, event)

    def trades_block(trades) -> str:
        if not trades:
            return "[dim](no trades named)[/]"
        return "\n".join(
            f"  • {t.ticker}  [bold]{t.side.upper()}[/]"
            + (f"  P(win)={t.probability}" if t.probability is not None else "")
            + f"\n    [dim]{t.reasoning}[/]"
            for t in trades
        )

    # ---- STAGE 1 ----
    console.print(Rule("[bold]STAGE 1 — Independent Analysis"))
    for a in result.stage1_results:
        if a.error:
            console.print(Panel(f"[red]ERROR: {a.error}[/]",
                                title=f"{a.label} — {a.model}", border_style="red"))
            continue
        console.print(Panel(
            f"[bold]Predicted temp = {a.predicted_temp_f}°F[/]   "
            f"confidence = {a.confidence}   cost = ${a.cost_usd:.5f}\n\n"
            f"{trades_block(a.trades)}",
            title=f"{a.label} — {a.model}", border_style="cyan"))

    # ---- STAGE 2 ----
    console.print(Rule("[bold]STAGE 2 — Peer Review (anonymized)"))
    for r in result.stage2_results:
        if r.error:
            console.print(Panel(f"[red]ERROR: {r.error}[/]",
                                title=f"{r.label} — {r.model}", border_style="red"))
            continue
        console.print(Panel(
            f"[bold]Updated predicted temp = {r.updated_predicted_temp_f}°F[/]   "
            f"cost = ${r.cost_usd:.5f}\n\n"
            f"{trades_block(r.updated_trades)}\n\n"
            f"[green]Agreements:[/] {r.agreements}\n\n"
            f"[yellow]Disagreements:[/] {r.disagreements}",
            title=f"{r.label} — {r.model}", border_style="magenta"))

    # ---- STAGE 3 ----
    console.print(Rule("[bold]STAGE 3 — Chairman Synthesis"))
    s3 = result.stage3_result
    if s3.error:
        console.print(Panel(f"[red]ERROR: {s3.error}[/]\n\n"
                            f"{trades_block(s3.trades)}",
                            title=f"Chairman — {s3.model}", border_style="red"))
    else:
        console.print(Panel(
            f"[bold]Final predicted temp = {s3.predicted_temp_f}°F[/]   "
            f"confidence = {s3.confidence}   cost = ${s3.cost_usd:.5f}\n\n"
            f"[bold]Trades:[/]\n{trades_block(s3.trades)}\n\n"
            f"[bold]Overall reasoning:[/]\n{s3.overall_reasoning}\n\n"
            f"[yellow]Dissent summary:[/] {s3.dissent_summary}\n\n"
            f"[red]Risk factors:[/] {s3.risk_factors}",
            title=f"Chairman — {s3.model}", border_style="green"))

    console.print(f"\n  [bold]Total council cost: ${result.total_cost:.5f}[/]")

    if not write_db:
        console.print("\n[bold yellow]--no-db: nothing written to the database.[/]")
        return 0

    # ---- Persist: one decision row per trade + one paper position each ----
    console.print(Rule("[bold]LOGGED TO DATABASE"))
    row_ids = persist_event_decision(result, event, weather_ctx.get("nws_temp"))
    console.print(f"  council_decisions: {len(row_ids)} row(s) {row_ids}")

    bracket_by_ticker = {b["ticker"]: b for b in ev.brackets}
    session = get_session()
    try:
        for trade in result.trades:
            bracket = bracket_by_ticker[trade.ticker]
            entry_price = bracket["yes_price"] if trade.side == "yes" else bracket["no_price"]
            if entry_price <= 0 or entry_price >= 1:
                console.print(f"  [dim]{trade.ticker}: unpriceable ({trade.side} @ ${entry_price}) — no position[/]")
                continue
            pos = Position(
                ticker=trade.ticker,
                strategy="weather_council",
                side=trade.side,
                entry_price=entry_price.quantize(Decimal("0.0001")),
                contracts=PAPER_CONTRACTS,
                entry_time=datetime.now(timezone.utc),
                status="open",
            )
            session.add(pos)
            session.commit()
            console.print(
                f"  [green]PAPER TRADE[/] {trade.ticker}  {trade.side.upper()} "
                f"x{PAPER_CONTRACTS} @ ${entry_price:.2f}  "
                f"cost=${entry_price * PAPER_CONTRACTS:.2f}  (position id={pos.id})"
            )
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
