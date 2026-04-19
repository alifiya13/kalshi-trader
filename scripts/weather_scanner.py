#!/usr/bin/env python3
"""
=============================================================
  WEATHER SCANNER — Strategy 1 signal runner
=============================================================

Scores tomorrow's KXHIGHNY (NYC daily high) markets against
a live Open-Meteo ensemble forecast, then runs Kelly sizing
against the current DEMO balance. Read-only — does NOT place
any orders.

Run:  python -m scripts.weather_scanner
=============================================================
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from core.rest_client import get_trade_client
from strategies.weather import find_weather_edge, WeatherSignal, NYC_LAT, NYC_LON
from strategies.kelly import compute_kelly


# Force a wide console so the multi-column tables render without wrap
# regardless of the user's actual terminal width.
console = Console(width=140)

EDGE_ALERT = Decimal("0.05")  # highlight signals with edge > 5 cents


def _fmt_pct(d: Decimal) -> str:
    return f"{float(d) * 100:5.1f}%"


def _fmt_edge(d: Decimal) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{float(d) * 100:5.1f}¢"


def _get_demo_balance(client) -> Decimal:
    """
    Pull the live DEMO cash balance. Falls back to $100 if the balance call
    fails for any reason (keys not set, network, etc.) — the scanner is
    informational and should still produce a sizing estimate.
    """
    try:
        resp = client.get_balance()
        bal = resp.get("balance", Decimal("0"))
        if isinstance(bal, Decimal):
            return bal
        return Decimal(str(bal))
    except Exception as e:
        console.print(f"[yellow]⚠ Could not fetch balance ({e}) — using $100 fallback[/]")
        return Decimal("100.00")


def _signal_table(signals: list[WeatherSignal]) -> Table:
    tbl = Table(
        title="KXHIGHNY — Tomorrow's NYC High vs Ensemble Forecast",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
        collapse_padding=True,
    )
    tbl.add_column("Ticker", style="cyan", no_wrap=True)
    tbl.add_column("Range", style="bold", no_wrap=True)
    tbl.add_column("Model", justify="right", style="magenta", no_wrap=True)
    tbl.add_column("Market", justify="right", no_wrap=True)
    tbl.add_column("Bid/Ask", justify="right", style="dim", no_wrap=True)
    tbl.add_column("Edge", justify="right", style="bold", no_wrap=True)
    tbl.add_column("Side", justify="center", no_wrap=True)
    tbl.add_column("Conf", justify="center", style="dim", no_wrap=True)

    for s in signals:
        edge_str = _fmt_edge(s.edge)
        if s.edge >= EDGE_ALERT:
            edge_disp = f"[bold green]{edge_str}[/]"
            side_disp = f"[bold green]{s.side.upper()}[/]"
        elif s.edge <= Decimal("-0.05"):
            edge_disp = f"[red]{edge_str}[/]"
            side_disp = f"[red]{s.side.upper()}[/]"
        else:
            edge_disp = edge_str
            side_disp = s.side.upper()

        tbl.add_row(
            s.ticker,
            s.threshold_label,
            _fmt_pct(s.model_prob),
            _fmt_pct(s.market_prob),
            f"${float(s.market_yes_bid):.2f}/${float(s.market_yes_ask):.2f}",
            edge_disp,
            side_disp,
            s.confidence,
        )
    return tbl


def _sizing_table(signals: list[WeatherSignal], balance: Decimal) -> Table:
    tbl = Table(
        title=f"Kelly Sizing @ balance ${balance:,.2f}  (0.25× Kelly, min edge 5¢)",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
        collapse_padding=True,
    )
    tbl.add_column("Ticker", style="cyan", no_wrap=True)
    tbl.add_column("Side", justify="center", no_wrap=True)
    tbl.add_column("Cost", justify="right", no_wrap=True)
    tbl.add_column("Edge", justify="right", style="bold", no_wrap=True)
    tbl.add_column("Kraw", justify="right", style="dim", no_wrap=True)
    tbl.add_column("Size%", justify="right", no_wrap=True)
    tbl.add_column("Qty", justify="right", style="bold", no_wrap=True)
    tbl.add_column("Max P/L", justify="right", no_wrap=True)
    tbl.add_column("EV", justify="right", style="green", no_wrap=True)
    tbl.add_column("Decision", style="dim")

    any_trade = False
    for s in signals:
        # Kelly sizer takes model prob and current market YES price, and
        # internally figures out which side of the book to buy. We pass
        # the market midpoint as the "yes price" — that's the fair
        # reference; real execution would use ask/bid depending on side.
        result = compute_kelly(
            model_prob=s.model_prob,
            market_yes_price=s.market_prob,
            portfolio_balance=balance,
            kelly_multiplier=Decimal("0.25"),
            min_edge=EDGE_ALERT,
            max_position_pct=Decimal("0.02"),
        )
        if result.should_trade:
            any_trade = True

        size_pct = f"{float(result.position_size_pct) * 100:4.2f}%"
        kelly_raw = f"{float(result.kelly_fraction):4.3f}"
        cost = f"${float(result.cost_per_contract):.2f}"
        edge = _fmt_edge(result.edge)
        max_pl = f"+${float(result.max_profit):.2f}/-${float(result.max_loss):.2f}"
        ev = f"${float(result.expected_value):+.2f}"
        decision = result.reason

        if result.should_trade:
            side_disp = f"[bold green]{result.side.upper()}[/]"
            decision = f"[green]{decision}[/]"
        else:
            side_disp = f"[dim]{result.side.upper()}[/]"

        tbl.add_row(
            s.ticker,
            side_disp,
            cost,
            edge,
            kelly_raw,
            size_pct,
            str(result.contracts),
            max_pl,
            ev,
            decision,
        )

    if not any_trade:
        tbl.caption = "[yellow]No trade qualifies under 5¢ min-edge gate.[/]"
    return tbl


def main():
    console.print("[bold magenta]" + "=" * 70)
    console.print("[bold magenta]  STRATEGY 1 — WEATHER SCANNER (KXHIGHNY)")
    console.print("[bold magenta]" + "=" * 70)
    console.print()

    client = get_trade_client()
    balance = _get_demo_balance(client)
    console.print(f"[dim]Using balance: [/][bold]${balance:,.2f}[/]")
    console.print(
        f"[dim]Location: NYC (Central Park)  {NYC_LAT}, {NYC_LON}[/]"
    )
    console.print(f"[dim]Ensemble: Open-Meteo gfs_seamless (NOAA GEFS, ~30 members)[/]")
    console.print()

    console.print("[dim]Fetching markets + ensemble forecast...[/]")
    signals = find_weather_edge(client)

    if not signals:
        console.print(
            "[yellow]No active KXHIGHNY markets found for tomorrow, or no "
            "ensemble data for the target date.[/]"
        )
        console.print(
            "[dim]If today is before markets open for tomorrow, this is expected. "
            "Try again in a few hours.[/]"
        )
        return

    console.print(
        f"[green]✓[/] Scored {len(signals)} market(s) against "
        f"{signals[0].n_members} ensemble members"
    )
    console.print()

    # --- Signal table ---
    console.print(_signal_table(signals))
    console.print()

    # --- Highlight alert signals ---
    alerts = [s for s in signals if s.edge >= EDGE_ALERT]
    if alerts:
        console.print(
            f"[bold green]★ {len(alerts)} signal(s) with edge ≥ "
            f"{_fmt_edge(EDGE_ALERT)}:[/]"
        )
        for s in alerts:
            console.print(
                f"  [green]{s.ticker}[/]  {s.threshold_label}  "
                f"model={_fmt_pct(s.model_prob)}  mkt={_fmt_pct(s.market_prob)}  "
                f"edge=[bold]{_fmt_edge(s.edge)}[/]  → buy [bold]{s.side.upper()}[/]"
            )
        console.print()
    else:
        console.print("[dim]No signals cleared the 5¢ edge alert threshold.[/]")
        console.print()

    # --- Kelly sizing table ---
    console.print(_sizing_table(signals, balance))
    console.print()

    # --- Summary ---
    max_edge = max(signals, key=lambda s: s.edge)
    console.print("[bold]Summary:[/]")
    console.print(f"  Markets scored:   {len(signals)}")
    console.print(f"  Ensemble members: {signals[0].n_members}")
    console.print(
        f"  Best edge:        {_fmt_edge(max_edge.edge)} on "
        f"[cyan]{max_edge.ticker}[/] ({max_edge.threshold_label}, "
        f"buy {max_edge.side.upper()})"
    )
    console.print(f"  Signals ≥ 5¢:     {len(alerts)}")
    console.print()
    console.print(
        "[dim]This is a dry run — NO orders were placed. "
        "Inspect the edges and decide manually before trading.[/]"
    )


if __name__ == "__main__":
    main()
