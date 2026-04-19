#!/usr/bin/env python3
"""
=============================================================
  PAPER TRADER — Strategy 1 (weather_v1)
=============================================================

Runs the weather scanner once, applies the full signal gate
(tradeability + Kelly), and writes qualifying trades to the
`paper_trades` table. Does NOT place real orders.

Run:  python -m scripts.paper_trader

Intended cadence: once per day when tomorrow's KXHIGHNY
markets open. Re-running the same day will produce duplicate
rows — dedupe is out of scope for v1 (strategy validation,
not accounting).
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

from core.rest_client import get_trade_client
from data.db import PaperTrade, get_session, init_db
from strategies.kelly import compute_kelly
from strategies.weather import find_weather_edge, WeatherSignal


console = Console(width=140)

STRATEGY_NAME = "weather_v1"
EDGE_ALERT = Decimal("0.05")
KELLY_MULTIPLIER = Decimal("0.25")
MAX_POSITION_PCT = Decimal("0.02")


def _fmt_pct(d: Decimal) -> str:
    return f"{float(d) * 100:5.1f}%"


def _fmt_edge(d: Decimal) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{float(d) * 100:5.1f}¢"


def _get_balance(client) -> Decimal:
    try:
        resp = client.get_balance()
        bal = resp.get("balance", Decimal("0"))
        return bal if isinstance(bal, Decimal) else Decimal(str(bal))
    except Exception as e:
        console.print(f"[yellow]⚠ balance fetch failed ({e}) — using $100 fallback[/]")
        return Decimal("100.00")


def _entry_price(signal: WeatherSignal) -> Decimal:
    """
    Pick the execution price for the side we want.

    For a BUY YES trade, we cross the book at the YES ask.
    For a BUY NO trade, we cross at the NO ask = (1 - YES bid).
    This mirrors how a real marketable order would fill.
    """
    if signal.side == "yes":
        return signal.market_yes_ask if signal.market_yes_ask > 0 else signal.market_prob
    else:
        return (Decimal("1") - signal.market_yes_bid) if signal.market_yes_bid > 0 else (Decimal("1") - signal.market_prob)


def _render_signal_overview(signals: list[WeatherSignal]) -> Table:
    tbl = Table(
        title="Weather signals — all candidates",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
        collapse_padding=True,
    )
    tbl.add_column("Ticker", style="cyan", no_wrap=True)
    tbl.add_column("Range", style="bold", no_wrap=True)
    tbl.add_column("GFS", justify="right", style="dim", no_wrap=True)
    tbl.add_column("ICON", justify="right", style="dim", no_wrap=True)
    tbl.add_column("ECMWF", justify="right", style="dim", no_wrap=True)
    tbl.add_column("Avg", justify="right", style="magenta", no_wrap=True)
    tbl.add_column("Market", justify="right", no_wrap=True)
    tbl.add_column("Edge", justify="right", style="bold", no_wrap=True)
    tbl.add_column("Side", justify="center", no_wrap=True)
    tbl.add_column("Conf", justify="center", no_wrap=True)
    tbl.add_column("Trade?", justify="center", no_wrap=True)

    for s in signals:
        def _pm(key):
            v = s.per_model_probs.get(key)
            return f"{v*100:4.1f}%" if v is not None else "  —"

        conf_color = {"high": "green", "medium": "yellow", "low": "red"}[s.confidence]
        tradeable_disp = "[bold green]YES[/]" if s.tradeable else "[red]no[/]"
        edge_disp = f"[bold]{_fmt_edge(s.edge)}[/]"

        tbl.add_row(
            s.ticker,
            s.threshold_label,
            _pm("gfs_seamless"),
            _pm("icon_seamless"),
            _pm("ecmwf_ifs04"),
            _fmt_pct(s.model_prob),
            _fmt_pct(s.market_prob),
            edge_disp,
            s.side.upper(),
            f"[{conf_color}]{s.confidence}[/]",
            tradeable_disp,
        )
    return tbl


def _render_paper_trades(logged: list[dict]) -> Table:
    tbl = Table(
        title=f"Paper trades logged to DB ({STRATEGY_NAME})",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
        collapse_padding=True,
    )
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("Ticker", style="cyan", no_wrap=True)
    tbl.add_column("Side", justify="center", no_wrap=True)
    tbl.add_column("Entry", justify="right", no_wrap=True)
    tbl.add_column("Qty", justify="right", style="bold")
    tbl.add_column("Cost", justify="right", no_wrap=True)
    tbl.add_column("Potential P", justify="right", style="green", no_wrap=True)
    tbl.add_column("Edge", justify="right", no_wrap=True)
    tbl.add_column("Model→Mkt", justify="right", style="dim", no_wrap=True)
    tbl.add_column("NWS", justify="right", no_wrap=True)
    tbl.add_column("Conf", justify="center", no_wrap=True)

    for i, row in enumerate(logged, 1):
        tbl.add_row(
            str(i),
            row["ticker"],
            row["side"].upper(),
            f"${float(row['entry_price']):.2f}",
            str(row["contracts"]),
            f"${float(row['cost']):.2f}",
            f"+${float(row['potential_profit']):.2f}",
            _fmt_edge(row["signal_edge"]),
            f"{float(row['model_prob'])*100:.1f}% → {float(row['market_prob'])*100:.1f}%",
            f"{row['nws_high']}°F" if row["nws_high"] else "—",
            row["signal_confidence"],
        )
    return tbl


def main():
    console.print("[bold magenta]" + "=" * 70)
    console.print("[bold magenta]  PAPER TRADER — weather_v1")
    console.print("[bold magenta]" + "=" * 70)
    console.print()

    init_db()
    client = get_trade_client()
    balance = _get_balance(client)
    console.print(f"[dim]Strategy:[/] {STRATEGY_NAME}")
    console.print(f"[dim]Balance: [/]${balance:,.2f}")
    console.print(f"[dim]Kelly:   [/]{KELLY_MULTIPLIER}× / max position {MAX_POSITION_PCT}")
    console.print()

    console.print("[dim]Fetching markets + ensembles + NWS...[/]")
    signals = find_weather_edge(client, min_edge=EDGE_ALERT)

    if not signals:
        console.print("[yellow]No signals produced. Either no markets for tomorrow, "
                      "no ensemble data, or target date out of forecast horizon.[/]")
        return

    # Show diagnostic banner: consensus of all data sources
    sample = signals[0]
    console.print(
        f"[green]✓[/] {len(signals)} market(s) scored against "
        f"{sample.n_models} ensemble(s), {sample.n_members} members total"
    )
    if sample.ensemble_mean_f is not None:
        console.print(f"[dim]Ensemble mean high:[/] {sample.ensemble_mean_f:.1f}°F")
    if sample.nws_forecast_high is not None:
        console.print(f"[dim]NWS official high: [/]{sample.nws_forecast_high}°F")
    else:
        console.print("[yellow]NWS forecast unavailable — no veto applied[/]")

    # If there was a day-level veto, surface it loudly.
    vetoed_day = [s for s in signals if s.veto_reason and "disagree" in s.veto_reason]
    if vetoed_day:
        console.print(f"[bold red]★ DAY-LEVEL VETO:[/] {vetoed_day[0].veto_reason}")
    console.print()

    console.print(_render_signal_overview(signals))
    console.print()

    # --- Apply gate: signal.tradeable AND Kelly says go ---
    logged_rows: list[dict] = []
    session = get_session()
    try:
        for s in signals:
            if not s.tradeable:
                continue

            # Kelly sizer — pass the execution price as the reference.
            entry = _entry_price(s)
            # We pass market_prob to Kelly because Kelly decides side
            # internally based on model vs market. The execution price
            # (entry) is what we actually ACTUALLY log.
            kelly = compute_kelly(
                model_prob=s.model_prob,
                market_yes_price=s.market_prob,
                portfolio_balance=balance,
                kelly_multiplier=KELLY_MULTIPLIER,
                min_edge=EDGE_ALERT,
                max_position_pct=MAX_POSITION_PCT,
            )
            if not kelly.should_trade:
                continue

            contracts = kelly.contracts
            cost = entry * contracts
            potential_profit = (Decimal("1") - entry) * contracts

            row = PaperTrade(
                ticker=s.ticker,
                strategy=STRATEGY_NAME,
                side=s.side,
                entry_price=entry.quantize(Decimal("0.0001")),
                contracts=contracts,
                cost=cost.quantize(Decimal("0.0001")),
                potential_profit=potential_profit.quantize(Decimal("0.0001")),
                signal_edge=s.edge.quantize(Decimal("0.0001")),
                signal_confidence=s.confidence,
                model_prob=s.model_prob,
                market_prob=s.market_prob.quantize(Decimal("0.0001")),
                nws_high=s.nws_forecast_high,
                placed_at=datetime.now(timezone.utc),
            )
            session.add(row)
            logged_rows.append({
                "ticker": s.ticker,
                "side": s.side,
                "entry_price": entry,
                "contracts": contracts,
                "cost": cost,
                "potential_profit": potential_profit,
                "signal_edge": s.edge,
                "signal_confidence": s.confidence,
                "model_prob": s.model_prob,
                "market_prob": s.market_prob,
                "nws_high": s.nws_forecast_high,
            })
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    if not logged_rows:
        console.print("[yellow]No trades qualified. "
                      "Either every signal was vetoed (confidence=low) "
                      "or Kelly rejected every size.[/]")
        # Show WHY the top signals were skipped
        if signals:
            console.print("\n[dim]Top rejections:[/]")
            for s in signals[:5]:
                reason = s.veto_reason or (
                    "edge below 5¢" if s.edge < EDGE_ALERT else
                    "confidence low" if s.confidence == "low" else
                    "kelly gate"
                )
                console.print(
                    f"  [dim]{s.ticker}  {s.threshold_label}  "
                    f"edge={_fmt_edge(s.edge)}  conf={s.confidence}  "
                    f"→ [/]{reason}"
                )
        return

    console.print(_render_paper_trades(logged_rows))
    console.print()

    total_cost = sum(r["cost"] for r in logged_rows)
    total_max_profit = sum(r["potential_profit"] for r in logged_rows)
    console.print(f"[bold]Logged {len(logged_rows)} paper trade(s) to DB[/]")
    console.print(f"  Total at risk:     ${float(total_cost):,.2f}")
    console.print(f"  Max potential P:   ${float(total_max_profit):,.2f}")
    console.print(f"  % of balance used: {float(total_cost/balance)*100:.2f}%")
    console.print()
    console.print("[dim]Next step: after markets settle, run "
                  "python -m scripts.paper_settle to score results.[/]")


if __name__ == "__main__":
    main()
