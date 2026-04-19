#!/usr/bin/env python3
"""
Single-market dry test for the AI Debate system.

Picks ONE active MLB market (price in [0.10, 0.90], highest orderbook depth),
runs bull → bear → judge, then prints each stage plus the final edge.
Does NOT place any trade.

Run:  python -m scripts.debate_dry_test
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich import box

from core.rest_client import get_data_client, KalshiClient
from data.market_scanner import mkt_yes_bid, mkt_yes_ask, mkt_volume
from strategies.ai_debate import run_debate, PRICE_MIN, PRICE_MAX

console = Console(width=120)


def _eligible(m: dict) -> bool:
    if m.get("status") not in ("active", "open"):
        return False
    yb = Decimal(str(mkt_yes_bid(m)))
    ya = Decimal(str(mkt_yes_ask(m)))
    mid = (yb + ya) / 2 if (yb > 0 and ya > 0) else (yb or ya)
    return PRICE_MIN <= mid <= PRICE_MAX


def _pick_market(client: KalshiClient) -> dict | None:
    """Pick the eligible MLB market with the highest orderbook depth."""
    candidates: list[dict] = []
    for series in ("KXMLBTOTAL", "KXMLBSPREAD"):
        resp = client.get_markets(series_ticker=series, status="open", limit=200)
        candidates.extend(resp.get("markets", []))

    eligible = [m for m in candidates if _eligible(m)]
    console.print(f"[dim]Eligible MLB markets (price in [0.10, 0.90]): {len(eligible)}[/]")
    if not eligible:
        return None

    # Score by orderbook depth — one API call per market; cap at 20 candidates.
    scored: list[tuple[dict, Decimal]] = []
    for m in eligible[:20]:
        try:
            ob = client.get_orderbook(m["ticker"])
            parsed = KalshiClient.parse_orderbook(ob)
            depth = KalshiClient.compute_depth(parsed["yes_bids"]) + KalshiClient.compute_depth(parsed["no_bids"])
            scored.append((m, depth))
        except Exception:
            continue

    scored.sort(key=lambda t: t[1], reverse=True)
    if not scored:
        return None

    best, depth = scored[0]
    console.print(f"[dim]Top depth: {depth} contracts on {best['ticker']}[/]")
    return best


def main():
    console.print("[bold magenta]" + "=" * 70)
    console.print("[bold magenta]  AI DEBATE — SINGLE-MARKET DRY TEST (no trade)")
    console.print("[bold magenta]" + "=" * 70)
    console.print()

    client = get_data_client()
    market = _pick_market(client)
    if not market:
        console.print("[red]No eligible MLB market found.[/]")
        sys.exit(1)

    # Show the raw market being debated
    yb = mkt_yes_bid(market)
    ya = mkt_yes_ask(market)
    console.print(Panel.fit(
        f"[bold cyan]{market.get('title', '')}[/]\n"
        f"Ticker: {market.get('ticker')}\n"
        f"YES bid/ask: ${yb:.2f} / ${ya:.2f}\n"
        f"Volume: {int(mkt_volume(market))}\n"
        f"Close time: {market.get('close_time', '?')}",
        title="Market Under Debate", box=box.ROUNDED,
    ))
    console.print()

    # Run the debate — agents write logs via structlog
    console.print("[bold]Running debate... (bull → bear → judge)[/]\n")
    result = run_debate(client, market)

    # --- Bull ---
    console.print(Panel(
        f"[bold]probability[/]: {result.bull.probability}\n"
        f"[bold]probability_floor[/]: {result.bull.probability_floor}\n"
        f"[bold]confidence[/]: {result.bull.confidence}\n"
        f"[bold]arguments[/]:\n"
        + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(result.bull.arguments))
        + f"\n[bold]reasoning[/]: {result.bull.reasoning}\n"
        + f"[dim]model={result.bull.llm.model}  "
        + f"tokens={result.bull.llm.total_tokens}  "
        + f"latency={result.bull.llm.latency_ms}ms  "
        + f"cost=${result.bull.llm.cost_usd:.4f}[/]",
        title="[green]BULL[/]", box=box.ROUNDED,
    ))
    console.print()

    # --- Bear ---
    console.print(Panel(
        f"[bold]probability[/]: {result.bear.probability}\n"
        f"[bold]probability_ceiling[/]: {result.bear.probability_ceiling}\n"
        f"[bold]confidence[/]: {result.bear.confidence}\n"
        f"[bold]counter_arguments[/]:\n"
        + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(result.bear.counter_arguments))
        + f"\n[bold]reasoning[/]: {result.bear.reasoning}\n"
        + f"[dim]model={result.bear.llm.model}  "
        + f"tokens={result.bear.llm.total_tokens}  "
        + f"latency={result.bear.llm.latency_ms}ms  "
        + f"cost=${result.bear.llm.cost_usd:.4f}[/]",
        title="[red]BEAR[/]", box=box.ROUNDED,
    ))
    console.print()

    # --- Judge ---
    console.print(Panel(
        f"[bold]probability[/]: {result.judge.probability}\n"
        f"[bold]confidence[/] (raw): {result.judge.confidence}\n"
        f"[bold]should_trade[/]: {result.judge.should_trade}\n"
        f"[bold]side[/]: {result.judge.side}\n"
        f"[bold]edge_assessment[/]: {result.judge.edge_assessment}\n"
        f"[bold]reasoning[/]: {result.judge.reasoning}\n"
        + f"[dim]model={result.judge.llm.model}  "
        + f"tokens={result.judge.llm.total_tokens}  "
        + f"latency={result.judge.llm.latency_ms}ms  "
        + f"cost=${result.judge.llm.cost_usd:.4f}[/]",
        title="[yellow]JUDGE[/]", box=box.ROUNDED,
    ))
    console.print()

    # --- Final edge calc ---
    penalty_note = ""
    if result.disagreement > 0.30:
        penalty_note = " (applied 0.7x disagreement penalty — gap > 0.30)"

    console.print(Panel(
        f"Market price (mid):     [bold]${float(result.market_price):.4f}[/]\n"
        f"Judge probability:      [bold]{result.judge_prob:.4f}[/]\n"
        f"Chosen side:            [bold]{result.side.upper()}[/]\n"
        f"Cost / contract:        [bold]${float(result.cost_per_contract):.4f}[/]\n"
        f"Edge:                   [bold]{float(result.edge):+.4f}[/]\n"
        f"Bull/Bear/Judge probs:  {result.bull_prob:.3f} / {result.bear_prob:.3f} / {result.judge_prob:.3f}\n"
        f"Disagreement:           {result.disagreement:.3f}\n"
        f"Adjusted confidence:    [bold]{float(result.confidence):.3f}[/]{penalty_note}\n"
        f"Total debate cost:      [bold]${result.total_cost_usd:.4f}[/]\n"
        f"[dim]Dry-run — no trade placed.[/]",
        title="[bold]FINAL VERDICT[/]", box=box.DOUBLE,
    ))


if __name__ == "__main__":
    main()
