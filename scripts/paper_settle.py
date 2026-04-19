#!/usr/bin/env python3
"""
=============================================================
  PAPER SETTLER — score resolved paper trades
=============================================================

Reads every unsettled row in `paper_trades`, checks Kalshi
for the market's `result` field, and fills in result / pnl /
settled_at. Then prints a strategy-level performance summary.

Run:  python -m scripts.paper_settle

Intended cadence: once per day after KXHIGHNY settlement
(late evening ET, after NWS midnight cutoff).
=============================================================
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from core.rest_client import get_data_client
from data.db import PaperTrade, get_session, init_db


console = Console(width=140)


def _fetch_market_result(client, ticker: str) -> tuple[str, str]:
    """
    Ask Kalshi for a single market's terminal state. Returns
    (status, result) — result is "yes"/"no"/"" depending on settlement.

    Settled markets expose `result` in the /markets/{ticker} response.
    Active markets return status="active" and result="" (or missing).
    """
    try:
        resp = client.get_market(ticker)
    except Exception as e:
        return ("error", str(e))

    m = resp.get("market", resp) if isinstance(resp, dict) else {}
    status = m.get("status", "")
    result = (m.get("result") or "").lower()
    return (status, result)


def _compute_pnl(trade: PaperTrade, market_result: str) -> Decimal:
    """
    Binary contract P&L:
      - Win  → profit = (1 - entry_price) * contracts
      - Loss → loss   = -entry_price * contracts
    `market_result` is the YES/NO outcome of the MARKET. We win if
    our `side` matches.
    """
    entry = Decimal(str(trade.entry_price))
    contracts = Decimal(str(trade.contracts))

    won = (trade.side.lower() == market_result.lower())
    if won:
        return (Decimal("1") - entry) * contracts
    else:
        return -entry * contracts


def main():
    console.print("[bold magenta]" + "=" * 70)
    console.print("[bold magenta]  PAPER SETTLER")
    console.print("[bold magenta]" + "=" * 70)
    console.print()

    init_db()
    client = get_data_client()  # public endpoint, no auth needed

    session = get_session()
    try:
        open_trades = session.query(PaperTrade).filter(
            PaperTrade.settled_at.is_(None)
        ).all()

        console.print(f"[dim]Open paper trades:[/] {len(open_trades)}")

        if open_trades:
            newly_settled = 0
            for trade in open_trades:
                status, result = _fetch_market_result(client, trade.ticker)

                if status == "error":
                    console.print(f"  [red]✗[/] {trade.ticker}: {result}")
                    continue

                if status not in ("settled", "finalized") or not result:
                    console.print(
                        f"  [dim]⏳[/] {trade.ticker}: "
                        f"status={status} result={result or '—'} "
                        f"(still open)"
                    )
                    continue

                pnl = _compute_pnl(trade, result)
                trade.result = result
                trade.pnl = pnl.quantize(Decimal("0.0001"))
                trade.settled_at = datetime.now(timezone.utc)
                session.add(trade)
                newly_settled += 1

                won = trade.side.lower() == result.lower()
                tag = "[green]WIN[/]" if won else "[red]LOSS[/]"
                console.print(
                    f"  {tag} {trade.ticker}  side={trade.side.upper()}  "
                    f"result={result.upper()}  pnl=${float(pnl):+.2f}"
                )

            session.commit()
            console.print()
            console.print(f"[bold]Newly settled: {newly_settled}[/]")
            console.print()

        # --- Strategy-level summary across ALL settled trades ---
        all_settled = session.query(PaperTrade).filter(
            PaperTrade.settled_at.isnot(None)
        ).all()

        if not all_settled:
            console.print("[yellow]No settled paper trades yet.[/]")
            return

        # Group by strategy
        by_strat: dict[str, list[PaperTrade]] = {}
        for t in all_settled:
            by_strat.setdefault(t.strategy, []).append(t)

        tbl = Table(
            title="Paper trading performance (settled)",
            box=box.SIMPLE_HEAVY,
        )
        tbl.add_column("Strategy", style="cyan")
        tbl.add_column("Trades", justify="right")
        tbl.add_column("Wins", justify="right", style="green")
        tbl.add_column("Losses", justify="right", style="red")
        tbl.add_column("Win %", justify="right")
        tbl.add_column("Total Cost", justify="right")
        tbl.add_column("Total P&L", justify="right", style="bold")
        tbl.add_column("ROI", justify="right", style="bold")

        for strat, trades in sorted(by_strat.items()):
            wins = sum(1 for t in trades if (t.pnl or 0) > 0)
            losses = len(trades) - wins
            total_cost = sum((Decimal(str(t.cost)) for t in trades), Decimal("0"))
            total_pnl = sum((Decimal(str(t.pnl or 0)) for t in trades), Decimal("0"))
            win_rate = wins / len(trades) if trades else 0
            roi = (total_pnl / total_cost * 100) if total_cost > 0 else Decimal("0")

            pnl_color = "green" if total_pnl >= 0 else "red"
            tbl.add_row(
                strat,
                str(len(trades)),
                str(wins),
                str(losses),
                f"{win_rate*100:.1f}%",
                f"${float(total_cost):,.2f}",
                f"[{pnl_color}]${float(total_pnl):+,.2f}[/]",
                f"[{pnl_color}]{float(roi):+.1f}%[/]",
            )

        console.print(tbl)
        console.print()

        # Detail table for the most recent settled trades
        recent = sorted(
            all_settled,
            key=lambda t: t.settled_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:20]

        det = Table(title="Most recent settled trades (up to 20)", box=box.SIMPLE)
        det.add_column("Ticker", style="cyan", no_wrap=True)
        det.add_column("Strat", no_wrap=True)
        det.add_column("Side", justify="center")
        det.add_column("Entry", justify="right")
        det.add_column("Qty", justify="right")
        det.add_column("Result", justify="center")
        det.add_column("P&L", justify="right", style="bold")
        det.add_column("Conf", justify="center")

        for t in recent:
            pnl = Decimal(str(t.pnl or 0))
            pnl_color = "green" if pnl >= 0 else "red"
            det.add_row(
                t.ticker[:40],
                t.strategy,
                t.side.upper(),
                f"${float(t.entry_price):.2f}",
                str(t.contracts),
                (t.result or "").upper(),
                f"[{pnl_color}]${float(pnl):+.2f}[/]",
                t.signal_confidence or "—",
            )
        console.print(det)

    finally:
        session.close()


if __name__ == "__main__":
    main()
