#!/usr/bin/env python3
"""
=============================================================
  MARKET SCANNER — Find where the money moves
=============================================================

Scans all open Kalshi markets and ranks them by:
  - Volume (where the liquidity is)
  - Spread (where execution is cheapest)
  - Opportunity score: volume * (1 / spread)
  - Category breakdown

Saves results to data/market_scan_results.csv.

Run:  python -m scripts.scan_markets
=============================================================
"""

import csv
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich import print as rprint

from core.rest_client import KalshiClient
from data.market_scanner import (
    MarketScanner,
    infer_category,
    mkt_volume,
    mkt_volume_24h,
    mkt_liquidity,
    mkt_yes_bid,
    mkt_yes_ask,
    mkt_total_resting_size,
)
from data.db import init_db


console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "market_scan_results.csv"


def fetch_orderbook_data(
    client: KalshiClient, markets: list, count: int = 20
) -> list:
    """
    Fetch orderbooks for the top N markets by resting-book liquidity and
    compute spread, depth, and opportunity score for each.

    Returns a list of dicts with enriched market data.
    """
    # Sort by liquidity_dollars first (best signal for "is there a live book?"),
    # with volume_fp as tiebreaker.
    sorted_markets = sorted(
        markets,
        key=lambda m: (mkt_liquidity(m), mkt_volume(m)),
        reverse=True,
    )
    top_markets = sorted_markets[:count]

    results = []
    for i, m in enumerate(top_markets):
        ticker = m.get("ticker", "")
        volume = mkt_volume(m)
        liquidity = mkt_liquidity(m)

        console.print(f"  [{i + 1}/{len(top_markets)}] Fetching orderbook for {ticker}...", end="\r")

        try:
            raw_ob = client.get_orderbook(ticker)
            parsed = KalshiClient.parse_orderbook(raw_ob)

            best_yes_bid = parsed["best_yes_bid"]
            best_yes_ask = parsed["best_yes_ask"]
            spread = parsed["spread"]
            yes_depth = KalshiClient.compute_depth(parsed["yes_bids"])
            no_depth = KalshiClient.compute_depth(parsed["no_bids"])
            total_depth = yes_depth + no_depth

            # Opportunity score: depth * (1 / spread)
            # Depth is a better live-book signal than lifetime volume on
            # day-of markets (which have low volume but real resting size).
            if spread > 0 and total_depth > 0:
                opportunity = total_depth * (Decimal("1") / spread)
            else:
                opportunity = Decimal("0")

            results.append({
                "ticker": ticker,
                "title": m.get("title", ""),
                "category": infer_category(ticker),
                "volume": volume,
                "liquidity": liquidity,
                "yes_bid": best_yes_bid,
                "yes_ask": best_yes_ask,
                "spread": spread,
                "depth": total_depth,
                "opportunity_score": opportunity,
                "yes_bids_raw": parsed["yes_bids"],
                "no_bids_raw": parsed["no_bids"],
            })
        except Exception as e:
            results.append({
                "ticker": ticker,
                "title": m.get("title", ""),
                "category": infer_category(ticker),
                "volume": volume,
                "liquidity": liquidity,
                "yes_bid": Decimal("0"),
                "yes_ask": Decimal("0"),
                "spread": Decimal("0"),
                "depth": Decimal("0"),
                "opportunity_score": Decimal("0"),
                "yes_bids_raw": [],
                "no_bids_raw": [],
                "error": str(e),
            })

    # Clear the progress line
    console.print(" " * 80, end="\r")
    return results


def save_csv(results: list, path: Path):
    """Save enriched results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    scanned_at = datetime.now(timezone.utc).isoformat()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker", "title", "category", "volume", "liquidity",
            "yes_bid", "yes_ask", "spread", "depth",
            "opportunity_score", "scanned_at",
        ])
        for r in results:
            writer.writerow([
                r["ticker"],
                r["title"],
                r["category"],
                f"{r['volume']:.0f}",
                f"{r.get('liquidity', 0):.2f}",
                f"{r['yes_bid']:.4f}",
                f"{r['yes_ask']:.4f}",
                f"{r['spread']:.4f}",
                f"{r['depth']:.0f}",
                f"{r['opportunity_score']:.2f}",
                scanned_at,
            ])


def print_top_recommendations(results: list, top_n: int = 5):
    """Print the top N markets ranked by opportunity score with reasoning."""
    ranked = sorted(results, key=lambda r: r["opportunity_score"], reverse=True)
    # Filter out markets with zero opportunity (no orderbook data)
    ranked = [r for r in ranked if r["opportunity_score"] > 0]

    if not ranked:
        console.print("[yellow]No markets with active orderbooks found.[/]")
        return

    top = ranked[:top_n]

    console.print()
    console.print("[bold magenta]" + "=" * 70)
    console.print("[bold magenta]  TOP RECOMMENDATIONS")
    console.print("[bold magenta]" + "=" * 70)
    console.print()
    console.print(
        "[dim]Ranked by opportunity score = depth * (1 / spread).[/]"
    )
    console.print(
        "[dim]Deep resting book + tight spread = best execution environment.[/]"
    )
    console.print()

    for i, r in enumerate(top, 1):
        spread = r["spread"]
        volume = r["volume"]
        liquidity = r.get("liquidity", 0)
        score = r["opportunity_score"]
        depth = r["depth"]

        # Build reasoning
        reasons = []
        if spread <= Decimal("0.05"):
            reasons.append("tight spread (<= $0.05)")
        elif spread <= Decimal("0.10"):
            reasons.append("moderate spread")
        else:
            reasons.append("wide spread -- execution risk")

        if depth >= 500:
            reasons.append("very deep book")
        elif depth >= 100:
            reasons.append("deep book")
        elif depth > 0:
            reasons.append(f"shallow book (depth={depth:.0f})")

        if liquidity >= 10000:
            reasons.append("strong resting liquidity")
        elif liquidity >= 1000:
            reasons.append("decent resting liquidity")

        if volume >= 1000:
            reasons.append("high lifetime volume")
        elif volume < 50:
            reasons.append("day-of market (low lifetime vol)")

        reasoning = ", ".join(reasons)

        title = (r["title"] or "")[:55]
        console.print(
            f"  [bold cyan]#{i}[/] [bold]{r['ticker']}[/]"
        )
        console.print(f"      {title}")
        console.print(
            f"      Score: [green]{score:,.0f}[/]  |  "
            f"Liq: ${liquidity:,.0f}  |  "
            f"Vol: {volume:,.0f}  |  "
            f"Spread: ${spread:.3f}  |  "
            f"Depth: {depth:.0f}"
        )
        console.print(f"      [dim]{reasoning}[/]")
        console.print()


def main():
    console.print("[bold magenta]Kalshi Market Scanner[/]\n")

    # Init
    init_db()
    client = KalshiClient()
    scanner = MarketScanner(client)

    # Scan open markets. Prod has hundreds of thousands of open markets,
    # ~90% of which are zero-volume KXMVE* parlay/cross-category permutations.
    # Skip those at fetch time and cap pages so the scan finishes in ~60s.
    console.print("[dim]Scanning open markets (excluding KXMVE* parlays)...[/]\n")
    all_markets = scanner.scan_all_open(
        max_pages=500,
        exclude_prefixes=("KXMVE",),
    )

    if not all_markets:
        console.print("[red]No open markets found.[/]")
        return

    # --- Category breakdown (inferred from ticker prefix) ---
    groups = scanner.group_by_category(all_markets)

    cat_table = Table(title="Markets by Category (inferred from ticker)")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Σ Liquidity ($)", justify="right", style="green")
    cat_table.add_column("Σ Volume", justify="right")
    cat_table.add_column("w/ Live Book", justify="right", style="magenta")

    def _cat_key(c):
        return -sum(mkt_liquidity(m) for m in groups[c])

    for cat in sorted(groups.keys(), key=_cat_key):
        cat_markets = groups[cat]
        total_liq = sum(mkt_liquidity(m) for m in cat_markets)
        total_vol = sum(mkt_volume(m) for m in cat_markets)
        with_book = sum(1 for m in cat_markets if mkt_total_resting_size(m) > 0)
        cat_table.add_row(
            cat,
            str(len(cat_markets)),
            f"{total_liq:,.0f}",
            f"{total_vol:,.0f}",
            str(with_book),
        )

    console.print(cat_table)
    console.print()

    # --- Top 25 by liquidity (more meaningful than volume for day-of markets) ---
    sorted_by_liq = sorted(
        all_markets,
        key=lambda m: (mkt_liquidity(m), mkt_volume(m)),
        reverse=True,
    )

    vol_table = Table(title="Top 25 Markets by Liquidity")
    vol_table.add_column("Ticker", style="cyan", max_width=32)
    vol_table.add_column("Title", max_width=40)
    vol_table.add_column("Cat", max_width=9)
    vol_table.add_column("Liquidity", justify="right", style="green")
    vol_table.add_column("Vol", justify="right")
    vol_table.add_column("YES Bid", justify="right")
    vol_table.add_column("YES Ask", justify="right", style="red")

    for m in sorted_by_liq[:25]:
        yb = mkt_yes_bid(m)
        ya = mkt_yes_ask(m)
        vol_table.add_row(
            (m.get("ticker") or "?")[:32],
            (m.get("title") or "?")[:40],
            infer_category(m.get("ticker", "")),
            f"${mkt_liquidity(m):,.0f}",
            f"{mkt_volume(m):,.0f}",
            f"${yb:.2f}" if yb else "--",
            f"${ya:.2f}" if ya else "--",
        )

    console.print(vol_table)
    console.print()

    # --- Orderbook analysis on top 20 by liquidity ---
    console.print("[bold]Fetching orderbooks for top 20 markets by liquidity...[/]\n")
    enriched = fetch_orderbook_data(client, all_markets, count=20)

    # Sort enriched by opportunity score for the table
    enriched_sorted = sorted(enriched, key=lambda r: r["opportunity_score"], reverse=True)

    depth_table = Table(title="Spread, Depth & Opportunity Analysis (Top 20)")
    depth_table.add_column("#", justify="right", style="dim")
    depth_table.add_column("Ticker", style="cyan", max_width=32)
    depth_table.add_column("YES Bid", justify="right", style="green")
    depth_table.add_column("YES Ask", justify="right", style="red")
    depth_table.add_column("Spread", justify="right", style="yellow")
    depth_table.add_column("Depth", justify="right")
    depth_table.add_column("Liq ($)", justify="right", style="green")
    depth_table.add_column("Vol", justify="right")
    depth_table.add_column("Opp Score", justify="right", style="bold magenta")

    for i, r in enumerate(enriched_sorted, 1):
        if r.get("error"):
            depth_table.add_row(
                str(i), r["ticker"][:32], "--", "--", "--", "--",
                f"{r.get('liquidity', 0):,.0f}",
                f"{r.get('volume', 0):,.0f}",
                "err",
            )
        else:
            depth_table.add_row(
                str(i),
                r["ticker"][:32],
                f"${r['yes_bid']:.2f}",
                f"${r['yes_ask']:.2f}",
                f"${r['spread']:.3f}",
                f"{r['depth']:.0f}",
                f"{r.get('liquidity', 0):,.0f}",
                f"{r.get('volume', 0):,.0f}",
                f"{r['opportunity_score']:,.0f}",
            )

    console.print(depth_table)
    console.print()

    # --- Save CSV ---
    save_csv(enriched_sorted, CSV_PATH)
    console.print(f"[bold]Results saved to:[/] {CSV_PATH}")
    console.print()

    # --- Top recommendations ---
    print_top_recommendations(enriched)

    # --- Summary ---
    live_book = scanner.filter_has_liquidity(all_markets)
    tradeable_count = len(scanner.filter_tradeable(all_markets, min_volume=50))
    console.print("[bold]Summary:[/]")
    console.print(f"  Total open markets kept:       {len(all_markets):,}")
    console.print(f"  Markets with live orderbook:   {len(live_book):,}")
    console.print(f"  Tradeable (vol>=50 or liq>0):  {tradeable_count:,}")
    console.print(f"  Orderbooks fetched:            {len(enriched):,}")
    console.print(f"  CSV saved:                     {CSV_PATH}")
    console.print()

    console.print("[bold]Next steps:[/]")
    console.print("  1. Pick a market from the recommendations above")
    console.print("  2. Run: python -m scripts.watch_orderbook --ticker <TICKER>")
    console.print("  3. Watch it for 30 min to understand the rhythm")
    console.print()


if __name__ == "__main__":
    main()
