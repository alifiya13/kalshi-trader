#!/usr/bin/env python3
"""
=============================================================
  ACTIVE TRADER — LLM-council research engine (weather-only)
=============================================================

This is a RESEARCH project about council accuracy, not a
money-making bot. Every council decision results in a paper
trade so we can measure how well the council predicts.

Runs continuously:
  - Every 30 minutes: scan the weather EVENT (tomorrow's NYC
    high), run the council on each bracket, paper-trade every
    bracket the council says to trade
  - Every 30 seconds: update prices + evaluate exits
  - Prints live status table to terminal

The market model is EVENT-level: KXHIGHNY-26JUN06 is one event
with 6-8 mutually exclusive temperature brackets under it. The
council sees the full bracket picture and decides per bracket.

There are NO trading gates. No edge threshold, no confidence
floor, no risk engine. The council's decision IS the decision —
that's the experiment. The only limits are cost controls:
  - $1/day cap on LLM spend
  - 30-minute interval between council scans

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
from typing import Optional

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
from data.db import Position, get_session, init_db
from execution.position_manager import PositionManager, ExitAction
from agents.council import WeatherCouncil, CouncilResult, persist_council_decision
from strategies.weather import get_weather_context, get_weather_event_with_brackets
from scripts.settle_council import settle_council_decisions
from monitoring.telegram_bot import (
    alert_startup,
    alert_new_position,
    alert_exit,
    alert_scan_summary,
    alert_error,
)
from config.settings import settings as _settings

console = Console(width=140)

# Loop cadence
SCAN_INTERVAL = 1800  # 30 minutes between council scans (cost control)
PRICE_INTERVAL = 30   # 30 seconds between price updates

# Fixed paper-trade size. This is a research project about council
# accuracy — sizing is irrelevant, so every trade is the same size to
# keep the P&L series interpretable.
PAPER_CONTRACTS = 10

# --- Council cost control (budget protection, NOT a trading gate) ---
# Council calls 7 LLMs per run (3 + 3 + 1 chairman), ~$0.016/run.
COUNCIL_DAILY_COST_CAP = 1.00   # USD — stop running the council past this/day

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
# Market-data packet for the council
# ----------------------------------------------------------------------

def _build_market_data(bracket: dict) -> dict:
    """
    Assemble the live-market half of the council's context packet from
    one bracket of the weather event.
    """
    close_time = bracket.get("close_time")
    hours_to_settlement = None
    if close_time:
        try:
            ct = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
            hours_to_settlement = round((ct - datetime.now(timezone.utc)).total_seconds() / 3600, 1)
        except (ValueError, AttributeError):
            pass

    yes_ask = bracket["yes_price"]
    yes_bid = bracket["yes_bid"]
    spread = (yes_ask - yes_bid) if (yes_ask > 0 and yes_bid > 0) else None

    return {
        "ticker": bracket["ticker"],
        "title": bracket["title"] or bracket["threshold"],
        "threshold_label": bracket["threshold"],
        "yes_price": bracket["yes_price"],
        "no_price": bracket["no_price"],
        "spread": spread,
        "market_prob": float(bracket["market_prob"]),
        "volume": bracket["volume"],
        "close_time": close_time,
        "hours_to_settlement": hours_to_settlement,
    }


def _council_edge(result: CouncilResult, bracket: dict) -> Optional[Decimal]:
    """
    Edge implied by the council's own probability vs the entry cost on
    the side it picked. Purely descriptive — logged for research, never
    used to gate the trade.
    """
    if result.final_probability is None:
        return None
    p = Decimal(str(result.final_probability))
    if result.side == "yes":
        return p - bracket["yes_price"]
    return (Decimal("1") - p) - bracket["no_price"]


# ----------------------------------------------------------------------
# Main scan — council runs on every bracket; its decision IS the decision
# ----------------------------------------------------------------------

def scan_for_opportunities(client, pm: PositionManager, dry_run: bool) -> dict:
    global _council_cost_today

    empty = {"entries": [], "num_markets": 0, "num_signals": 0, "council_runs": 0}

    # 1. Event-level fetch: tomorrow's NYC high with ALL its brackets
    event = get_weather_event_with_brackets(client)
    if not event:
        console.print("[dim]  No active weather event found[/]")
        return empty

    brackets = event["brackets"]
    console.print(
        f"[dim]  Event: {event['series_ticker']} {event['event_date']} — "
        f"{len(brackets)} bracket(s): "
        + ", ".join(b["threshold"] for b in brackets) + "[/]"
    )

    # 2. Weather context (ensembles + NWS), fetched once for the event
    try:
        weather_ctx = get_weather_context(target_date=event["event_date"])
    except Exception as e:
        console.print(f"[yellow]  Weather context failed: {e}[/]")
        return empty

    council = WeatherCouncil()
    _reset_council_budget_if_new_day()

    existing = {p.ticker for p in pm.positions if p.status == "open"}
    entries: list[dict] = []
    council_runs = 0

    # 3. Run the council on every bracket of the event. No edge gate, no
    #    confidence gate, no risk engine — every decision is logged and
    #    every should_trade=true becomes a paper trade. Only the daily
    #    LLM budget can stop the loop.
    for bracket in brackets:
        ticker = bracket["ticker"]
        if ticker in existing:
            console.print(f"  [dim]{ticker}: already holding — skipped[/]")
            continue
        if _council_cost_today >= COUNCIL_DAILY_COST_CAP:
            console.print(
                f"[yellow]  Council stopped: daily cap hit "
                f"(${_council_cost_today:.4f} ≥ ${COUNCIL_DAILY_COST_CAP:.2f})[/]"
            )
            break

        market_data = _build_market_data(bracket)

        try:
            result = council.run_council(weather_ctx, market_data)
        except Exception as e:
            console.print(f"[yellow]  Council error on {ticker}: {e}[/]")
            continue
        council_runs += 1
        _council_cost_today += result.total_cost

        edge = _council_edge(result, bracket)

        # Log EVERY council decision, trade or not (research audit trail).
        persist_council_decision(
            result,
            market_yes_price=bracket["yes_price"],
            market_no_price=bracket["no_price"],
            edge=edge,
            weather_nws_high=weather_ctx.get("nws_high"),
        )

        conf = result.confidence or 0.0
        console.print(
            f"  [cyan]council[/] {ticker} [{bracket['threshold']}]: "
            f"final_prob={result.final_probability} should_trade={result.should_trade} "
            f"side={result.side} conf={conf:.2f}  cost=${result.total_cost:.4f} "
            f"(today=${_council_cost_today:.4f}/${COUNCIL_DAILY_COST_CAP:.2f})"
        )

        if not result.should_trade:
            continue

        # --- Council said trade → paper trade, no second-guessing ---
        entry_price = bracket["yes_price"] if result.side == "yes" else bracket["no_price"]
        if entry_price <= 0 or entry_price >= 1:
            console.print(f"  [dim]{ticker}: unpriceable ({result.side} @ ${entry_price}) — skipped[/]")
            continue

        entry = _record_entry(
            pm, ticker, result.side, entry_price,
            PAPER_CONTRACTS, "weather_council", dry_run,
            edge=edge or Decimal("0"),
        )
        if entry:
            entries.append(entry)
            existing.add(ticker)

    return {
        "entries": entries,
        "num_markets": len(brackets),
        "num_signals": council_runs,
        "council_runs": council_runs,
    }


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
            f"  [green]{'[DRY] ' if dry_run else ''}NEW[/] {ticker}  "
            f"{side.upper()} x{contracts} @ ${entry_price:.2f}  "
            f"cost=${cost:.2f}  strategy={strategy}"
        )
        alert_new_position(
            ticker=ticker, side=side, contracts=contracts,
            entry_price=entry_price, cost=cost, strategy=strategy, edge=edge,
        )
        return {
            "ticker": ticker, "side": side, "entry_price": entry_price,
            "contracts": contracts, "strategy": strategy,
        }
    except Exception as e:
        session.rollback()
        console.print(f"[red]  Entry failed: {e}[/]")
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

    balance = Decimal("100.00")
    if not dry_run:
        try:
            resp = trade_client.get_balance()
            balance = resp.get("balance", Decimal("100"))
            if not isinstance(balance, Decimal):
                balance = Decimal(str(balance))
        except Exception:
            pass
    console.print(f"[dim]Balance: ${balance:,.2f}  |  Research mode: every council decision is paper-traded[/]")
    console.print()

    alert_startup(env=_settings.kalshi_env + (" (dry-run)" if dry_run else " (live)"), balance=balance)

    global _running
    start_time = time.time()
    cycle = 0
    last_scan = 0.0
    last_price = 0.0

    try:
        while _running:
            now = time.time()
            cycle += 1

            if args.duration > 0 and (now - start_time) >= args.duration:
                console.print(f"\n[yellow]Duration limit ({args.duration}s) reached. Stopping.[/]")
                break

            # --- Scan: run the council on the weather event's brackets ---
            if now - last_scan >= SCAN_INTERVAL or last_scan == 0:
                console.print(f"\n[bold cyan]--- Scan #{cycle} @ {datetime.now().strftime('%H:%M:%S')} ---[/]")
                scan_stats = {"entries": [], "num_markets": 0, "num_signals": 0}
                try:
                    scan_stats = scan_for_opportunities(data_client, pm, dry_run)
                    new_entries = scan_stats["entries"]
                    if new_entries:
                        console.print(f"  [green]Entered {len(new_entries)} new position(s)[/]")
                    else:
                        console.print(f"  [dim]No new positions this scan[/]")
                except Exception as e:
                    console.print(f"  [red]Scan error: {e}[/]")
                    alert_error(f"Scan error: {e}")
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
                alert_scan_summary(
                    num_markets=scan_stats["num_markets"],
                    num_signals=scan_stats["num_signals"],
                    num_trades=len(scan_stats["entries"]),
                    balance=balance,
                )

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
                        alert_exit(
                            ticker=pos.ticker, side=pos.side,
                            exit_price=pos.current_price, pnl=pnl, reason=label,
                        )

                last_price = now

            console.print(build_status_table(pm, cycle, mode))

            # Sleep, checking _running frequently for responsiveness
            sleep_until = now + PRICE_INTERVAL
            while _running and time.time() < sleep_until:
                time.sleep(1)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/]")
        alert_error(f"Fatal error: {e}")
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

        console.print(f"\n[bold]Session Summary:[/]")
        console.print(f"  Total positions: {total}")
        console.print(f"  Still open:      {open_pos}")
        console.print(f"  Closed:          {closed}")
        console.print(f"  Realized P&L:    ${float(total_pnl):+.2f}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
