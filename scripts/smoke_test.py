#!/usr/bin/env python3
"""
=============================================================
  STEP 1: RUN THIS FIRST
=============================================================

This script verifies your entire setup in order:
  1. Config loads from .env
  2. Database initializes
  3. Public API works (no auth needed)
  4. Auth works (signed requests)
  5. Market scanner finds live markets
  6. Orderbook parsing works
  7. Kelly sizer computes a position

Run:  python -m scripts.smoke_test

If ANY step fails, fix it before moving on.
=============================================================
"""

import sys
from decimal import Decimal
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()


def step(name: str):
    """Decorator to wrap each step with pass/fail output."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            console.print(f"\n[bold cyan]{'='*60}[/]")
            console.print(f"[bold cyan]  STEP: {name}[/]")
            console.print(f"[bold cyan]{'='*60}[/]")
            try:
                result = func(*args, **kwargs)
                console.print(f"[bold green]  ✓ PASSED[/]\n")
                return result
            except Exception as e:
                console.print(f"[bold red]  ✗ FAILED: {e}[/]\n")
                raise
        return wrapper
    return decorator


@step("Load config from .env")
def test_config():
    from config.settings import settings
    rprint(f"  Active env:        [yellow]{settings.kalshi_env}[/]")
    rprint(f"  Public data URL:   [green]{settings.prod_base_url}[/]  (always prod)")
    rprint(f"  Trading URL:       [cyan]{settings.active_base_url}[/]  (follows KALSHI_ENV)")
    rprint(f"  Demo key ID:       {settings.kalshi_demo_api_key_id or '[red](missing)[/]'}")
    rprint(f"  Prod key ID:       {settings.kalshi_prod_api_key_id or '[yellow](not set — prod auth unavailable)[/]'}")
    rprint(f"  DB:                {settings.database_url}")
    rprint(f"  Kelly frac:        {settings.kelly_fraction}")
    assert settings.kalshi_env in ("demo", "prod"), "Invalid env"
    return settings


@step("Initialize database")
def test_db():
    from data.db import init_db, get_session
    init_db()
    session = get_session()
    session.close()
    rprint("  Tables created successfully")


@step("Public API — fetch markets from production (no auth)")
def test_public_api():
    from core.rest_client import get_trade_client
    # Trade client also serves public calls (routed to prod) and authed calls (routed to active env)
    client = get_trade_client()

    # Fetch first page of open markets — should hit prod
    resp = client.get_markets(status="open", limit=5)
    markets = resp.get("markets", [])
    rprint(f"  Found {len(markets)} markets (showing first 5)")

    table = Table(title="Sample Markets")
    table.add_column("Ticker", style="cyan")
    table.add_column("Title", max_width=40)
    table.add_column("Category")
    table.add_column("Volume")

    for m in markets[:5]:
        table.add_row(
            m.get("ticker", "?"),
            m.get("title", "?")[:40],
            m.get("category", "?"),
            str(m.get("volume", 0)),
        )
    console.print(table)

    assert len(markets) > 0, "No open markets found"
    return client, markets


@step("Orderbook parsing")
def test_orderbook(client, markets):
    from core.rest_client import KalshiClient

    # Pick first market with some activity
    ticker = markets[0]["ticker"]
    rprint(f"  Fetching orderbook for: [cyan]{ticker}[/]")

    raw_ob = client.get_orderbook(ticker)
    parsed = KalshiClient.parse_orderbook(raw_ob)

    rprint(f"  Best YES bid:  [green]${parsed['best_yes_bid']}[/]")
    rprint(f"  Best YES ask:  [red]${parsed['best_yes_ask']}[/]")
    rprint(f"  Spread:        [yellow]${parsed['spread']}[/]")
    rprint(f"  YES levels:    {len(parsed['yes_bids'])}")
    rprint(f"  NO levels:     {len(parsed['no_bids'])}")

    return parsed


@step("Authenticated API — check balance")
def test_auth(client):
    try:
        balance_resp = client.get_balance()
        balance = balance_resp.get("balance", Decimal("0"))
        portfolio_value = balance_resp.get("portfolio_value", Decimal("0"))
        rprint(f"  Cash balance:    [bold green]${balance:,.2f}[/]")
        rprint(f"  Portfolio value: [bold green]${portfolio_value:,.2f}[/]")
        rprint(f"  Full response:   {balance_resp}")
        rprint(f"  [green]Auth is working![/]")
        return balance_resp
    except Exception as e:
        if "kalshi_api_key_id" in str(e) or "Private key" in str(e) or "not found" in str(e):
            rprint(f"  [yellow]⚠ Auth skipped — no API key configured yet[/]")
            rprint(f"  [yellow]  This is OK for now. Set up keys at kalshi.com/account/profile[/]")
            rprint(f"  [yellow]  Then update .env with KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH[/]")
            return None
        raise


@step("Market scanner — full scan")
def test_scanner(client):
    from data.market_scanner import MarketScanner

    scanner = MarketScanner(client)
    # Just scan one page to keep it fast
    resp = client.get_markets(status="open", limit=50)
    markets = resp.get("markets", [])

    tradeable = scanner.filter_tradeable(markets, min_volume=10)
    groups = scanner.group_by_category(markets)

    rprint(f"  Total fetched:  {len(markets)}")
    rprint(f"  Tradeable:      {len(tradeable)}")
    rprint(f"  Categories:")
    for cat, cat_markets in sorted(groups.items(), key=lambda x: -len(x[1])):
        rprint(f"    {cat:15s}: {len(cat_markets)} markets")

    return tradeable


@step("Kelly position sizer")
def test_kelly():
    from strategies.kelly import compute_kelly

    # Scenario: Your model says 70% prob, market says 55 cents YES
    result = compute_kelly(
        model_prob=Decimal("0.70"),
        market_yes_price=Decimal("0.55"),
        portfolio_balance=Decimal("1000.00"),
        kelly_multiplier=Decimal("0.25"),
        min_edge=Decimal("0.05"),
    )

    rprint(f"  Scenario: model=70%, market=55¢, portfolio=$1000")
    rprint(f"  Should trade:   [{'green' if result.should_trade else 'red'}]{result.should_trade}[/]")
    rprint(f"  Side:           {result.side}")
    rprint(f"  Edge:           ${result.edge}")
    rprint(f"  Kelly raw:      {result.kelly_fraction:.4f}")
    rprint(f"  Contracts:      {result.contracts}")
    rprint(f"  Cost/contract:  ${result.cost_per_contract}")
    rprint(f"  Max profit:     ${result.max_profit}")
    rprint(f"  Max loss:       ${result.max_loss}")
    rprint(f"  Expected value: ${result.expected_value}")
    rprint(f"  Reason:         {result.reason}")

    assert result.should_trade, "Should have found edge here"
    assert result.side == "yes", "Should buy YES when model > market"
    assert result.contracts > 0, "Should have at least 1 contract"


def main():
    console.print("[bold magenta]" + "=" * 60)
    console.print("[bold magenta]  KALSHI TRADING SYSTEM — SMOKE TEST")
    console.print("[bold magenta]" + "=" * 60)

    # Step 1: Config
    settings = test_config()

    # Step 2: Database
    test_db()

    # Step 3: Public API
    client, markets = test_public_api()

    # Step 4: Orderbook
    test_orderbook(client, markets)

    # Step 5: Auth (may skip if no keys yet)
    test_auth(client)

    # Step 6: Scanner
    test_scanner(client)

    # Step 7: Kelly
    test_kelly()

    # Summary
    console.print("\n[bold green]" + "=" * 60)
    console.print("[bold green]  ALL CHECKS PASSED — YOUR SETUP IS READY")
    console.print("[bold green]" + "=" * 60)
    console.print()
    console.print("[bold]Next steps:[/]")
    console.print("  1. If auth was skipped → set up API keys in .env")
    console.print("  2. Run: python -m scripts.scan_markets")
    console.print("  3. Run: python -m scripts.watch_orderbook <TICKER>")
    console.print()


if __name__ == "__main__":
    main()
