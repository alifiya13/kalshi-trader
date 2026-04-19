#!/usr/bin/env python3
"""
=============================================================
  LIVE MARKET WATCHER
=============================================================

Polls a market's orderbook every N seconds and displays
a live trading screen in your terminal.

Usage:
  python -m scripts.watch_orderbook                      # auto-picks highest volume market
  python -m scripts.watch_orderbook KXBTC15M             # watch a specific series
  python -m scripts.watch_orderbook --ticker KXBTC15M-26APR09-T100000  # exact market

This is how you learn to READ the market before you trade it.
=============================================================
"""

import sys
import time
import argparse
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from core.rest_client import KalshiClient


console = Console()


def build_display(ticker: str, market: dict, parsed_ob: dict, history: list[dict]) -> Table:
    """Build the terminal display."""

    # Header
    title = market.get("title", ticker)
    status = market.get("status", "?")
    volume = market.get("volume", 0)

    header = Text()
    header.append(f"  {title}\n", style="bold white")
    header.append(f"  Ticker: {ticker}  |  Status: {status}  |  Volume: {volume}", style="dim")

    # Orderbook table
    ob_table = Table(title="Orderbook", show_header=True, header_style="bold")
    ob_table.add_column("YES Bids", justify="right", style="green", min_width=20)
    ob_table.add_column("Price", justify="center", style="bold", min_width=8)
    ob_table.add_column("NO Bids", justify="left", style="red", min_width=20)

    yes_bids = parsed_ob["yes_bids"]
    no_bids = parsed_ob["no_bids"]

    # Build price levels (show top 8 from each side)
    all_prices = set()
    for p, _ in yes_bids[-8:]:
        all_prices.add(p)
    for p, _ in no_bids[-8:]:
        implied_yes = Decimal("1.00") - p
        all_prices.add(implied_yes)

    yes_dict = {p: q for p, q in yes_bids}
    no_dict = {}
    for p, q in no_bids:
        implied_yes = Decimal("1.00") - p
        no_dict[implied_yes] = q

    for price in sorted(all_prices, reverse=True):
        yes_qty = yes_dict.get(price, Decimal("0"))
        no_qty = no_dict.get(price, Decimal("0"))

        yes_bar = "█" * min(int(float(yes_qty) / 5), 20) if yes_qty > 0 else ""
        no_bar = "█" * min(int(float(no_qty) / 5), 20) if no_qty > 0 else ""

        yes_str = f"{yes_qty:>6} {yes_bar}" if yes_qty > 0 else ""
        no_str = f"{no_bar} {no_qty:<6}" if no_qty > 0 else ""

        ob_table.add_row(yes_str, f"${price:.2f}", no_str)

    # Summary
    spread = parsed_ob["spread"]
    mid = (parsed_ob["best_yes_bid"] + parsed_ob["best_yes_ask"]) / 2

    summary = Table.grid(padding=(0, 2))
    summary.add_row(
        f"[green]Best YES bid: ${parsed_ob['best_yes_bid']}[/]",
        f"[red]Best YES ask: ${parsed_ob['best_yes_ask']}[/]",
        f"[yellow]Spread: ${spread:.3f}[/]",
        f"Mid: ${mid:.3f}",
    )

    # Price history (last 10 ticks)
    hist_table = Table(title="Recent Ticks", show_header=True, header_style="dim")
    hist_table.add_column("Time", style="dim")
    hist_table.add_column("Mid", justify="right")
    hist_table.add_column("Spread", justify="right")
    hist_table.add_column("Δ", justify="right")

    for i, h in enumerate(history[-10:]):
        delta = ""
        if i > 0:
            prev_mid = history[-10:][i-1]["mid"]
            d = h["mid"] - prev_mid
            if d > 0:
                delta = f"[green]+{d:.3f}[/]"
            elif d < 0:
                delta = f"[red]{d:.3f}[/]"
        hist_table.add_row(
            h["time"],
            f"${h['mid']:.3f}",
            f"${h['spread']:.3f}",
            delta,
        )

    # Compose
    master = Table.grid()
    master.add_row(Panel(header, border_style="blue"))
    master.add_row(summary)
    master.add_row(ob_table)
    master.add_row(hist_table)
    master.add_row(Text(f"\n  Refreshing every 3s  |  Ctrl+C to stop  |  {datetime.now().strftime('%H:%M:%S')}", style="dim"))

    return master


def find_best_market(client: KalshiClient, series: str | None = None) -> dict | None:
    """Find the highest-volume open market, optionally within a series."""
    if series:
        resp = client.get_markets(series_ticker=series, status="open", limit=50)
    else:
        resp = client.get_markets(status="open", limit=50)

    markets = resp.get("markets", [])
    if not markets:
        return None

    # Sort by volume descending
    markets.sort(key=lambda m: m.get("volume", 0), reverse=True)
    return markets[0]


def main():
    parser = argparse.ArgumentParser(description="Watch a Kalshi market's orderbook in real-time")
    parser.add_argument("series", nargs="?", help="Series ticker (e.g., KXBTC15M)")
    parser.add_argument("--ticker", help="Exact market ticker")
    parser.add_argument("--interval", type=int, default=3, help="Poll interval in seconds")
    args = parser.parse_args()

    client = KalshiClient()

    # Find the market to watch
    if args.ticker:
        market = client.get_market(args.ticker).get("market", {})
        ticker = args.ticker
    else:
        console.print("[dim]Finding highest-volume market...[/]")
        market = find_best_market(client, args.series)
        if not market:
            console.print("[red]No open markets found. Try a different series.[/]")
            return
        ticker = market["ticker"]

    console.print(f"[bold]Watching: {ticker}[/]")
    console.print(f"[dim]{market.get('title', '')}[/]\n")

    history: list[dict] = []

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                raw_ob = client.get_orderbook(ticker)
                parsed = KalshiClient.parse_orderbook(raw_ob)

                mid = (parsed["best_yes_bid"] + parsed["best_yes_ask"]) / 2
                history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "mid": mid,
                    "spread": parsed["spread"],
                })

                display = build_display(ticker, market, parsed, history)
                live.update(display)

            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
