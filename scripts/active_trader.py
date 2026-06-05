#!/usr/bin/env python3
"""
=============================================================
  ACTIVE TRADER — Continuous trading engine (weather-only)
=============================================================

Runs continuously:
  - Every 5 minutes: scan for new weather opportunities
  - Every 30 seconds: update prices + evaluate exits
  - Prints live status table to terminal

Scanning is TARGETED: we hit the KXHIGHNY weather series directly
via `get_markets(series_ticker=...)` instead of paginating through
the entire prod catalog (which is dominated by zero-volume
parlay permutations).

Every entry candidate is gated by the single RiskEngine
(execution/risk_engine.py). No strategy does its own sizing.

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
from execution.risk_engine import RiskEngine, TradeSignal
from monitoring.telegram_bot import (
    alert_startup,
    alert_new_position,
    alert_exit,
    alert_scan_summary,
    alert_error,
)
from config.settings import settings as _settings

console = Console(width=140)

# --- Scanning targets ---
# Weather-only. We pull this series directly and feed it into the weather
# strategy. Every candidate still passes the single RiskEngine gate.
TARGETED_SERIES: tuple[str, ...] = (
    "KXHIGHNY",       # NYC daily high — weather_v1 strategy
)

# Loop cadence
SCAN_INTERVAL = 300   # 5 minutes between opportunity scans
PRICE_INTERVAL = 30   # 30 seconds between price updates


# --- Graceful shutdown ---
_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ----------------------------------------------------------------------
# Targeted market fetch
# ----------------------------------------------------------------------

def _fetch_series_markets(client, series_ticker: str) -> list[dict]:
    """Pull every open market for a specific series in one call."""
    try:
        resp = client.get_markets(series_ticker=series_ticker, status="open", limit=200)
        markets = resp.get("markets", [])
        # Paginate if there's a cursor (rare for a single series, but some have
        # enough strikes to exceed 200).
        cursor = resp.get("cursor")
        while cursor:
            resp = client.get_markets(
                series_ticker=series_ticker, status="open", limit=200, cursor=cursor,
            )
            markets.extend(resp.get("markets", []))
            cursor = resp.get("cursor")
        return markets
    except Exception as e:
        console.print(f"[yellow]  Fetch {series_ticker} failed: {e}[/]")
        return []


# ----------------------------------------------------------------------
# Weather scanner — returns a list of TradeSignal + extra context
# ----------------------------------------------------------------------

def _weather_signals(client, weather_markets: list[dict]) -> list[dict]:
    """
    Run the weather strategy against an already-fetched KXHIGHNY market
    list. We pass the client to reuse the existing weather pipeline
    (ensemble + NWS), but the market filter is already targeted.
    """
    try:
        from strategies.weather import find_weather_edge
        signals = find_weather_edge(client, min_edge=Decimal("0.03"))
    except Exception as e:
        console.print(f"[yellow]  Weather scan error: {e}[/]")
        return []

    _CONF_MAP = {"high": Decimal("0.80"), "medium": Decimal("0.60"), "low": Decimal("0.30")}

    out: list[dict] = []
    for s in signals:
        if not s.tradeable:
            continue
        entry_price = s.market_yes_ask if s.side == "yes" else (Decimal("1") - s.market_yes_bid)
        if entry_price <= 0 or entry_price >= 1:
            continue
        confidence = _CONF_MAP.get(s.confidence, Decimal("0.30"))
        # Desired contracts: weather previously used its own Kelly. Now we hand
        # the engine a generous target; the engine will clamp to the table cap.
        desired = 20
        out.append({
            "signal": TradeSignal(
                ticker=s.ticker,
                side=s.side,
                strategy="weather_v1",
                edge=s.edge,
                confidence=confidence,
                cost_per_contract=entry_price,
                desired_contracts=desired,
                category="weather",
            ),
            "extra": {
                "label": s.threshold_label,
                "model_prob": float(s.model_prob),
                "market_prob": float(s.market_prob),
            },
        })
    return out


# ----------------------------------------------------------------------
# Main scan — feeds every signal through RiskEngine
# ----------------------------------------------------------------------

def scan_for_opportunities(client, pm: PositionManager, dry_run: bool, balance: Decimal) -> dict:
    risk = RiskEngine(balance)

    # 1. Targeted fetch — one API call for the weather series
    weather_markets = _fetch_series_markets(client, "KXHIGHNY")
    num_markets = len(weather_markets)
    console.print(f"[dim]  Targeted: weather={len(weather_markets)}[/]")

    # 2. Build weather signals
    weather_sigs = _weather_signals(client, weather_markets)
    num_signals = len(weather_sigs)
    console.print(f"[dim]  Signals: weather={len(weather_sigs)}[/]")

    # 3. Gate each signal via RiskEngine
    existing = {p.ticker for p in pm.positions if p.status == "open"}
    entries: list[dict] = []
    for bundle in weather_sigs:
        sig: TradeSignal = bundle["signal"]
        if sig.ticker in existing:
            continue

        decision = risk.check_can_trade(sig)
        if not decision.allowed:
            console.print(
                f"  [dim]skip {sig.ticker} ({sig.strategy}): {decision.reason}[/]"
            )
            continue

        entry = _record_entry(
            pm, sig.ticker, sig.side, sig.cost_per_contract,
            decision.approved_contracts, sig.strategy, dry_run,
            edge=sig.edge,
        )
        if entry:
            entries.append(entry)
            existing.add(sig.ticker)

    return {
        "entries": entries,
        "num_markets": num_markets,
        "num_signals": num_signals,
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
    console.print(f"[dim]Balance: ${balance:,.2f}  |  Risk-gated via RiskEngine (blueprint v2)[/]")
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

            # --- Scan for new opportunities ---
            if now - last_scan >= SCAN_INTERVAL or last_scan == 0:
                console.print(f"\n[bold cyan]--- Scan #{cycle} @ {datetime.now().strftime('%H:%M:%S')} ---[/]")
                scan_stats = {"entries": [], "num_markets": 0, "num_signals": 0}
                try:
                    scan_stats = scan_for_opportunities(data_client, pm, dry_run, balance)
                    new_entries = scan_stats["entries"]
                    if new_entries:
                        console.print(f"  [green]Entered {len(new_entries)} new position(s)[/]")
                    else:
                        console.print(f"  [dim]No new opportunities found[/]")
                except Exception as e:
                    console.print(f"  [red]Scan error: {e}[/]")
                    alert_error(f"Scan error: {e}")
                last_scan = now
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
