#!/usr/bin/env python3
"""
=============================================================
  ACTIVE TRADER — Continuous trading engine
=============================================================

Runs continuously:
  - Every 5 minutes: scan for new opportunities
  - Every 30 seconds: update prices + evaluate exits
  - Prints live status table to terminal

Scanning is TARGETED: we hit specific series directly via
`get_markets(series_ticker=...)` instead of paginating through
the entire prod catalog (which is dominated by zero-volume
parlay permutations).

Every entry candidate — from every strategy — is gated by the
single RiskEngine (execution/risk_engine.py). No strategy does
its own sizing.

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
from data.db import Position, DebateLog, get_session, init_db
from data.market_scanner import infer_category
from execution.position_manager import PositionManager, ExitAction
from execution.risk_engine import RiskEngine, TradeSignal
from strategies.safe_compounder import (
    find_compounder_opportunities,
    compute_compounder_size,
)
from strategies.ai_debate import scan_with_debate, DebateResult
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
# Series we actively pull and feed into strategies. Adding a series here
# gives it to BOTH the weather/MLB/strategy-specific scorers (where
# applicable) AND the Safe Compounder broad scan.
TARGETED_SERIES: tuple[str, ...] = (
    "KXHIGHNY",       # NYC daily high — weather_v1 strategy
    "KXMLBTOTAL",     # MLB game totals (over/under)
    "KXMLBSPREAD",    # MLB game spreads (run line)
)

# Optional: prefixes to sweep into the compounder when the series filter
# above misses things. Kalshi's series_ticker mapping isn't always 1:1
# with ticker prefix, so we keep a fallback list.
COMPOUNDER_PREFIX_FALLBACK: tuple[str, ...] = (
    "KXMLBTOTAL", "KXMLBSPREAD",
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
        # Paginate if there's a cursor (shouldn't happen often for a single series,
        # but some have enough strikes to exceed 200 — e.g., MLB on a full slate day).
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


def _fetch_targeted_markets(client) -> dict[str, list[dict]]:
    """Return {series_ticker: [markets]} for every series in TARGETED_SERIES."""
    out: dict[str, list[dict]] = {}
    for s in TARGETED_SERIES:
        markets = _fetch_series_markets(client, s)
        out[s] = markets
        console.print(f"[dim]  {s}: {len(markets)} market(s)[/]")
    return out


# ----------------------------------------------------------------------
# Strategy scanners — each returns a list of TradeSignal + extra context
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


def _log_debate_result(res: DebateResult) -> None:
    """Persist every debate to DB for later calibration (regardless of trade)."""
    session = get_session()
    try:
        session.add(DebateLog(
            ticker=res.ticker,
            market_title=res.market_title,
            bull_prob=Decimal(str(res.bull_prob)),
            bear_prob=Decimal(str(res.bear_prob)),
            judge_prob=Decimal(str(res.judge_prob)),
            disagreement=Decimal(str(res.disagreement)),
            market_price=res.market_price,
            edge=res.edge,
            side=res.side,
            confidence=res.confidence,
            should_trade=1 if res.should_trade else 0,
            total_cost=Decimal(str(res.total_cost_usd)),
            created_at=datetime.now(timezone.utc),
        ))
        session.commit()
    except Exception as e:
        session.rollback()
        console.print(f"[yellow]  Failed to log debate for {res.ticker}: {e}[/]")
    finally:
        session.close()


def _debate_signals(client, mlb_markets: list[dict]) -> list[dict]:
    """
    Run AI debate on top 5 MLB markets by orderbook depth. Every debate
    result (tradeable or not) is written to debate_logs. Only tradeable
    results with a concrete side return as TradeSignals.
    """
    try:
        results = scan_with_debate(client, mlb_markets)
    except Exception as e:
        console.print(f"[yellow]  AI debate error: {e}[/]")
        return []

    out: list[dict] = []
    for res in results:
        _log_debate_result(res)
        if not res.should_trade or res.side == "hold":
            continue
        if res.cost_per_contract <= 0 or res.cost_per_contract >= 1:
            continue
        desired = 30  # engine clamps via threshold-table max_pct
        out.append({
            "signal": TradeSignal(
                ticker=res.ticker,
                side=res.side,
                strategy="ai_debate",
                edge=res.edge,
                confidence=res.confidence,
                cost_per_contract=res.cost_per_contract,
                desired_contracts=desired,
                category=res.category,
            ),
            "extra": {
                "judge_prob": res.judge_prob,
                "disagreement": res.disagreement,
                "total_cost": res.total_cost_usd,
            },
        })
    return out


def _compounder_signals(client, all_markets: list[dict]) -> list[dict]:
    """Run the safe compounder against the combined targeted market set."""
    try:
        comp_signals = find_compounder_opportunities(client, all_markets)
    except Exception as e:
        console.print(f"[yellow]  Compounder scan error: {e}[/]")
        return []

    out: list[dict] = []
    for cs in comp_signals:
        desired = 50  # compounder wants size; engine will clamp
        out.append({
            "signal": TradeSignal(
                ticker=cs.ticker,
                side=cs.side,
                strategy="safe_compounder",
                edge=cs.edge,
                confidence=cs.confidence,
                cost_per_contract=cs.cost_per_contract,
                desired_contracts=desired,
                category=cs.category,
            ),
            "extra": {
                "time_decay": float(cs.time_decay_bonus),
                "hours_to_close": cs.hours_to_close,
                "depth": float(cs.orderbook_depth),
            },
        })
    return out


# ----------------------------------------------------------------------
# Main scan — feeds every signal through RiskEngine
# ----------------------------------------------------------------------

def scan_for_opportunities(client, pm: PositionManager, dry_run: bool, balance: Decimal) -> dict:
    risk = RiskEngine(balance)

    # 1. Targeted fetch — one API call per series
    series_markets = _fetch_targeted_markets(client)
    weather_markets = series_markets.get("KXHIGHNY", [])
    mlb_total = series_markets.get("KXMLBTOTAL", [])
    mlb_spread = series_markets.get("KXMLBSPREAD", [])

    num_markets = len(weather_markets) + len(mlb_total) + len(mlb_spread)

    console.print(
        f"[dim]  Targeted: weather={len(weather_markets)} "
        f"mlb_total={len(mlb_total)} mlb_spread={len(mlb_spread)}[/]"
    )

    # 2. Build signals from each strategy
    weather_sigs = _weather_signals(client, weather_markets)
    all_series_markets = weather_markets + mlb_total + mlb_spread
    compounder_sigs = _compounder_signals(client, all_series_markets)
    # AI debate runs on MLB markets (no domain data → LLM-driven edge)
    mlb_markets = mlb_total + mlb_spread
    debate_sigs = _debate_signals(client, mlb_markets)

    num_signals = len(weather_sigs) + len(compounder_sigs) + len(debate_sigs)

    console.print(
        f"[dim]  Signals: weather={len(weather_sigs)}  "
        f"compounder={len(compounder_sigs)}  debate={len(debate_sigs)}[/]"
    )

    # 3. Gate each signal via RiskEngine
    existing = {p.ticker for p in pm.positions if p.status == "open"}
    entries: list[dict] = []
    # Priority order: weather (hardest domain signal), debate (AI-synthesized),
    # compounder (mechanical baseline). Each still passes the same gate.
    for bundle in weather_sigs + debate_sigs + compounder_sigs:
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
