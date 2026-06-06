#!/usr/bin/env python3
"""
=============================================================
  ACTIVE TRADER — LLM-council research engine (weather-only)
=============================================================

This is a RESEARCH project about council accuracy, not a
money-making bot. Every council decision results in a paper
trade so we can measure how well the council predicts.

Runs continuously:
  - Every 30 minutes: DISCOVER all open Kalshi weather events
    (category "Climate and Weather" — no hardcoded tickers,
    cities, or coordinates), filter to temperature events
    closing before the study deadline, and run the council
    ONCE on each event not yet decided
  - Every 30 seconds: update prices + evaluate exits
  - Prints live status table to terminal

Per event, the council sees the full bracket table + that
city's forecast (geocoded via Nominatim, ensembles via
Open-Meteo, official forecast via NWS), predicts the
temperature, and MUST name at least one trade — skipping is
not an option (research mandate).

There are NO trading gates. No edge threshold, no confidence
floor, no risk engine. The council's decision IS the decision —
that's the experiment. The only limits are cost controls:
  - $2/day cap on LLM spend (many cities now)
  - 30-minute interval between scans
  - each event is decided at most once (dedup by event_ticker)

Ctrl+C to stop gracefully.

Run:
  python -m scripts.active_trader              # dry-run (no orders)
  python -m scripts.active_trader --live        # live demo orders
  python -m scripts.active_trader --duration 60 # run for 60 seconds

=============================================================
"""

import sys
import time
import signal
import argparse
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Railway / container deploys pass PEM key material as env vars; write it to
# disk BEFORE importing anything that builds a Kalshi REST client, since
# client construction reads the PEM file eagerly.
from config.settings import ensure_key_files
ensure_key_files()

# Force line-buffered stdout/stderr so Railway's log stream flushes promptly
# instead of waiting for a full buffer (container stdout is a pipe, not a tty).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

from rich.console import Console
from rich.table import Table
from rich import box

from core.rest_client import get_data_client, get_trade_client
from data.db import CouncilDecision, Position, get_session, init_db
from data.weather_discovery import DEADLINE_UTC, discover_weather_events
from execution.position_manager import PositionManager, ExitAction
from agents.council import WeatherCouncil, persist_event_decision
from strategies.weather import get_weather_context
from scripts.settle_council import settle_council_decisions
from config.settings import settings as _settings

console = Console(width=140)

# Loop cadence
SCAN_INTERVAL = 1800  # 30 minutes between discovery+council scans
PRICE_INTERVAL = 30   # 30 seconds between price updates

# Fixed paper-trade size. This is a research project about council
# accuracy — sizing is irrelevant, so every trade is the same size to
# keep the P&L series interpretable.
PAPER_CONTRACTS = 10

# --- Council cost control (budget protection, NOT a trading gate) ---
# Council calls 7 LLMs per run (3 + 3 + 1 chairman), ~$0.018/run. With
# ~40 temperature events/day across all cities, a full sweep is ~$0.75.
COUNCIL_DAILY_COST_CAP = 5.00   # USD — stop running the council past this/day

_council_cost_day: str = ""     # UTC date str for the current cost window
_council_cost_today: float = 0.0  # USD spent on councils in _council_cost_day


def _reset_council_budget_if_new_day() -> None:
    """Rollover the daily council-cost counter at UTC midnight."""
    global _council_cost_day, _council_cost_today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _council_cost_day:
        _council_cost_day = today
        _council_cost_today = 0.0


# --- Graceful shutdown ---
_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ----------------------------------------------------------------------
# Dedup — events the council has already decided
# ----------------------------------------------------------------------

def _decided_event_tickers() -> set[str]:
    """Event tickers that already have council_decisions rows."""
    session = get_session()
    try:
        rows = (
            session.query(CouncilDecision.event_ticker)
            .filter(CouncilDecision.event_ticker.isnot(None))
            .distinct()
            .all()
        )
        return {r[0] for r in rows}
    finally:
        session.close()


# ----------------------------------------------------------------------
# Main scan — discover events, one council run per undecided event
# ----------------------------------------------------------------------

def scan_for_opportunities(
    client, pm: PositionManager, dry_run: bool,
    keep_going: Callable[[], bool] = lambda: True,
) -> dict:
    """
    One full discovery + council sweep. `keep_going` is checked between
    events so SIGTERM / --duration can stop a long multi-city sweep
    mid-way without losing the decisions already made.
    """
    global _council_cost_today

    stats = {
        "entries": [], "events_discovered": 0, "events_in_window": 0,
        "events_already_decided": 0, "council_runs": 0, "scan_cost": 0.0,
    }

    # 1. Discover ALL open weather events (category-driven, nothing hardcoded)
    try:
        events = discover_weather_events(client)
    except Exception as e:
        console.print(f"[yellow]  Discovery failed: {e}[/]")
        return stats
    stats["events_discovered"] = len(events)

    # 2./3. Temperature events only (we can only forecast temperature), with
    # brackets inside the study deadline (discovery already dropped brackets
    # past DEADLINE_UTC; events with none left were dropped there too).
    tradeable = [ev for ev in events if ev.temp_type in ("high", "low")]
    stats["events_in_window"] = len(tradeable)

    # 4. Skip events the council already decided (one decision per event).
    decided = _decided_event_tickers()
    todo = [ev for ev in tradeable if ev.event_ticker not in decided]
    stats["events_already_decided"] = len(tradeable) - len(todo)

    console.print(
        f"[dim]  Discovered {len(events)} weather event(s) | "
        f"temperature & in-window: {len(tradeable)} | "
        f"already decided: {stats['events_already_decided']} | "
        f"to run: {len(todo)} (deadline {DEADLINE_UTC:%Y-%m-%d %H:%M} UTC)[/]"
    )

    _reset_council_budget_if_new_day()
    existing = {p.ticker for p in pm.positions if p.status == "open"}

    # 5. One council run per remaining event.
    for ev in todo:
        if not keep_going():
            console.print("[yellow]  Scan interrupted — stopping sweep[/]")
            break
        if _council_cost_today >= COUNCIL_DAILY_COST_CAP:
            console.print(
                f"[yellow]  Council stopped: daily cap hit "
                f"(${_council_cost_today:.4f} ≥ ${COUNCIL_DAILY_COST_CAP:.2f})[/]"
            )
            break

        console.print(
            f"\n  [bold]{ev.event_ticker}[/] — {ev.city} {ev.temp_type} "
            f"on {ev.event_date}  ({len(ev.brackets)} brackets, "
            f"lat={ev.lat:.3f} lon={ev.lon:.3f})"
        )

        # a. Forecast for this city/day/variable
        try:
            weather_ctx = get_weather_context(
                latitude=ev.lat, longitude=ev.lon,
                target_date=ev.event_date, temp_type=ev.temp_type,
                city=ev.city,
            )
        except Exception as e:
            console.print(f"  [yellow]weather context failed: {e}[/]")
            continue
        if not weather_ctx.get("n_members"):
            console.print("  [yellow]no ensemble members for this date — skipped[/]")
            continue

        # b./c. Council sees all brackets + forecast, must name ≥1 trade
        event_data = ev.as_council_event()
        try:
            result = WeatherCouncil().run_council(weather_ctx, event_data)
        except Exception as e:
            console.print(f"  [yellow]council error: {e}[/]")
            continue
        stats["council_runs"] += 1
        stats["scan_cost"] += result.total_cost
        _council_cost_today += result.total_cost

        row_ids = persist_event_decision(result, event_data, weather_ctx.get("nws_temp"))

        conf = result.confidence or 0.0
        console.print(
            f"  [cyan]council[/] predicted_{ev.temp_type}={result.predicted_temp_f}°F "
            f"(NWS {weather_ctx.get('nws_temp')}°F, ens {weather_ctx.get('ensemble_mean')}°F) "
            f"conf={conf:.2f}  {len(result.trades)} trade(s)  "
            f"cost=${result.total_cost:.4f} "
            f"(today=${_council_cost_today:.4f}/${COUNCIL_DAILY_COST_CAP:.2f})  "
            f"rows={row_ids}"
        )

        # d. Paper-trade every trade the council named.
        bracket_by_ticker = {b["ticker"]: b for b in ev.brackets}
        for trade in result.trades:
            bracket = bracket_by_ticker[trade.ticker]  # tickers pre-validated
            console.print(
                f"  [cyan]trade[/] {trade.ticker} [{bracket['threshold']}]: "
                f"{trade.side.upper()}  P(win)={trade.probability}"
            )
            if trade.ticker in existing:
                console.print("    [dim]already holding — position not duplicated[/]")
                continue

            entry_price = bracket["yes_price"] if trade.side == "yes" else bracket["no_price"]
            if entry_price <= 0 or entry_price >= 1:
                console.print(f"    [dim]unpriceable ({trade.side} @ ${entry_price}) — no position[/]")
                continue

            edge = (
                Decimal(str(trade.probability)) - entry_price
                if trade.probability is not None else Decimal("0")
            )
            entry = _record_entry(
                pm, trade.ticker, trade.side, entry_price,
                PAPER_CONTRACTS, "weather_council", dry_run, edge=edge,
            )
            if entry:
                stats["entries"].append(entry)
                existing.add(trade.ticker)

    return stats


def _record_entry(
    pm: PositionManager, ticker: str, side: str,
    entry_price: Decimal, contracts: int, strategy: str, dry_run: bool,
    edge: Decimal = Decimal("0"),
) -> dict | None:
    session = get_session()
    try:
        pos = Position(
            ticker=ticker,
            strategy=strategy,
            side=side,
            entry_price=entry_price.quantize(Decimal("0.0001")),
            contracts=contracts,
            entry_time=datetime.now(timezone.utc),
            status="open",
        )
        session.add(pos)
        session.commit()

        cost = entry_price * contracts
        console.print(
            f"    [green]{'[DRY] ' if dry_run else ''}NEW[/] {ticker}  "
            f"{side.upper()} x{contracts} @ ${entry_price:.2f}  "
            f"cost=${cost:.2f}  edge={float(edge):+.3f}"
        )
        return {
            "ticker": ticker, "side": side, "entry_price": entry_price,
            "contracts": contracts, "strategy": strategy,
        }
    except Exception as e:
        session.rollback()
        console.print(f"[red]    Entry failed: {e}[/]")
        return None
    finally:
        session.close()


# ----------------------------------------------------------------------
# Status table
# ----------------------------------------------------------------------

def build_status_table(pm: PositionManager, cycle: int, mode: str) -> Table:
    tbl = Table(
        title=f"Active Trader [{mode}]  |  Cycle {cycle}  |  {datetime.now().strftime('%H:%M:%S')}",
        box=box.ROUNDED,
        pad_edge=False,
    )
    tbl.add_column("#", justify="right", style="dim", width=3)
    tbl.add_column("Ticker", style="cyan", no_wrap=True, max_width=35)
    tbl.add_column("Side", justify="center", width=4)
    tbl.add_column("Entry", justify="right", width=7)
    tbl.add_column("Current", justify="right", width=7)
    tbl.add_column("Qty", justify="right", width=4)
    tbl.add_column("P&L", justify="right", width=9)
    tbl.add_column("Status", justify="center", width=12)
    tbl.add_column("Strategy", style="dim", max_width=18)

    for i, pos in enumerate(pm.positions, 1):
        pnl = pos.unrealized_pnl
        pnl_color = "green" if pnl >= 0 else "red"
        status_color = {
            "open": "white",
            "closed_profit": "green",
            "closed_loss": "red",
            "settled": "blue",
        }.get(pos.status, "dim")

        tbl.add_row(
            str(i),
            pos.ticker[:35],
            pos.side.upper(),
            f"${pos.entry_price:.2f}",
            f"${pos.current_price:.2f}",
            str(pos.contracts),
            f"[{pnl_color}]${float(pnl):+.2f}[/]",
            f"[{status_color}]{pos.status}[/]",
            pos.strategy,
        )

    if not pm.positions:
        tbl.add_row("", "[dim]No positions[/]", "", "", "", "", "", "", "")

    return tbl


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Active trading engine")
    parser.add_argument("--live", action="store_true", help="Place real demo orders (default: dry-run)")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds then stop (0=forever)")
    args = parser.parse_args()

    dry_run = not args.live
    mode = "DRY-RUN" if dry_run else "LIVE DEMO"

    console.print("[bold magenta]" + "=" * 70)
    console.print(f"[bold magenta]  ACTIVE TRADER — {mode}")
    console.print("[bold magenta]" + "=" * 70)
    console.print()

    init_db()

    data_client = get_data_client()
    trade_client = get_trade_client() if not dry_run else data_client

    pm = PositionManager(data_client, dry_run=dry_run)
    pm.load_open_positions()
    console.print(f"[dim]Loaded {len(pm.positions)} existing position(s)[/]")
    console.print(f"[dim]Research mode: every council decision is paper-traded. "
                  f"Study deadline: {DEADLINE_UTC:%Y-%m-%d %H:%M} UTC[/]")
    console.print()

    global _running
    start_time = time.time()
    cycle = 0
    last_scan = 0.0
    last_price = 0.0

    def keep_going() -> bool:
        if not _running:
            return False
        if args.duration > 0 and (time.time() - start_time) >= args.duration:
            return False
        return True

    try:
        while _running:
            now = time.time()
            cycle += 1

            if args.duration > 0 and (now - start_time) >= args.duration:
                console.print(f"\n[yellow]Duration limit ({args.duration}s) reached. Stopping.[/]")
                break

            # --- Discovery + council sweep ---
            if now - last_scan >= SCAN_INTERVAL or last_scan == 0:
                console.print(f"\n[bold cyan]--- Scan #{cycle} @ {datetime.now().strftime('%H:%M:%S')} ---[/]")
                try:
                    s = scan_for_opportunities(data_client, pm, dry_run, keep_going)
                    console.print(
                        f"\n  [bold]Scan summary:[/] discovered={s['events_discovered']} "
                        f"in_window={s['events_in_window']} "
                        f"already_decided={s['events_already_decided']} "
                        f"council_runs={s['council_runs']} "
                        f"paper_trades={len(s['entries'])} "
                        f"scan_cost=${s['scan_cost']:.4f}"
                    )
                except Exception as e:
                    console.print(f"  [red]Scan error: {e}[/]")
                last_scan = now

                # Settlement check — reconcile council decisions + positions
                # against Kalshi. Public API only, no LLM cost. Once per scan.
                try:
                    settle_stats = settle_council_decisions(data_client, verbose=False)
                    if settle_stats["decisions_settled"] or settle_stats["positions_settled"]:
                        console.print(
                            f"  [blue]Settled {settle_stats['decisions_settled']} decision(s), "
                            f"{settle_stats['positions_settled']} position(s)[/]"
                        )
                except Exception as e:
                    console.print(f"  [yellow]Settlement check failed: {e}[/]")

                pm.load_open_positions()

            # --- Price updates + exit evaluation ---
            if now - last_price >= PRICE_INTERVAL or last_price == 0:
                pm.update_prices()

                decisions = pm.evaluate_exits()
                for pos, action in decisions:
                    if action in (ExitAction.SELL_PROFIT, ExitAction.SELL_STOP_LOSS):
                        pm.execute_exit(pos, action)
                        label = "PROFIT" if action == ExitAction.SELL_PROFIT else "STOP-LOSS"
                        pnl = (pos.current_price - pos.entry_price) * pos.contracts
                        console.print(
                            f"  [{'green' if action == ExitAction.SELL_PROFIT else 'red'}]"
                            f"{'[DRY] ' if dry_run else ''}{label}[/] {pos.ticker}  "
                            f"${pos.entry_price:.2f}→${pos.current_price:.2f}  "
                            f"pnl=${float(pnl):+.2f}"
                        )

                last_price = now

            console.print(build_status_table(pm, cycle, mode))

            # Sleep, checking _running frequently for responsiveness
            sleep_until = now + PRICE_INTERVAL
            while _running and time.time() < sleep_until:
                if args.duration > 0 and (time.time() - start_time) >= args.duration:
                    break
                time.sleep(1)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/]")
        raise

    console.print("\n[bold yellow]Active trader stopped.[/]")

    # Final summary
    session = get_session()
    try:
        from sqlalchemy import func
        total = session.query(Position).count()
        open_pos = session.query(Position).filter(Position.status == "open").count()
        closed = session.query(Position).filter(
            Position.status.in_(["closed_profit", "closed_loss", "settled"])
        ).count()
        total_pnl = session.query(func.sum(Position.realized_pnl)).filter(
            Position.status.in_(["closed_profit", "closed_loss", "settled"])
        ).scalar() or 0
        decisions = session.query(CouncilDecision).count()
        runs = session.query(CouncilDecision.council_run_id).distinct().count()
        cost = session.query(func.sum(CouncilDecision.total_cost_usd)).scalar()

        console.print(f"\n[bold]Session Summary:[/]")
        console.print(f"  Council runs (events decided): {runs}")
        console.print(f"  Decision rows (trades named):  {decisions}")
        console.print(f"  Total positions: {total}")
        console.print(f"  Still open:      {open_pos}")
        console.print(f"  Closed:          {closed}")
        console.print(f"  Realized P&L:    ${float(total_pnl):+.2f}")
        if cost is not None:
            # total_cost_usd is per-run, duplicated across that run's rows —
            # sum of per-run costs needs the distinct run ids.
            run_costs = (
                session.query(CouncilDecision.council_run_id,
                              func.max(CouncilDecision.total_cost_usd))
                .group_by(CouncilDecision.council_run_id).all()
            )
            llm_total = sum(float(c or 0) for _, c in run_costs)
            console.print(f"  Total LLM cost:  ${llm_total:.4f}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
