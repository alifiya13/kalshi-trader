#!/usr/bin/env python3
"""
=============================================================
  DEMO ACCOUNT SETUP WIZARD
=============================================================

Walks you through setting up your Kalshi demo account:
  1. Ensures .env file exists
  2. Checks API key is configured
  3. Checks private key PEM file exists
  4. Makes an authenticated call to verify credentials
  5. Shows your demo account balance

Run:  python -m scripts.setup_demo
=============================================================
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

DEMO_SIGNUP_URL = "https://demo.kalshi.co"
API_KEYS_URL = "https://demo.kalshi.co/account/profile"


def print_header():
    console.print()
    console.print(Panel(
        "[bold magenta]Kalshi Demo Account Setup Wizard[/]\n\n"
        "This script verifies your demo credentials step by step.\n"
        "If anything is missing, it tells you exactly what to do.",
        border_style="magenta",
    ))
    console.print()


def step_env_file() -> bool:
    """Step 1: Ensure .env file exists."""
    console.print("[bold cyan]Step 1/5:[/] Checking .env file...")

    if ENV_FILE.exists():
        console.print("  [green].env file found.[/]")
        return True

    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        console.print("  [yellow].env was missing -- copied from .env.example.[/]")
        console.print("  [yellow]You will need to edit it with your credentials.[/]")
        return True

    console.print("  [red].env file not found and no .env.example to copy from.[/]")
    console.print("  [bold]Fix:[/] Create a .env file in the project root with at minimum:")
    console.print("    KALSHI_API_KEY_ID=your-key-id")
    console.print("    KALSHI_PRIVATE_KEY_PATH=./keys/kalshi-private.pem")
    console.print("    KALSHI_ENV=demo")
    return False


def step_api_key() -> bool:
    """Step 2: Check KALSHI_API_KEY_ID is set."""
    console.print("\n[bold cyan]Step 2/5:[/] Checking API key ID...")

    # Read .env directly to check the raw value before pydantic defaults kick in
    env_text = ENV_FILE.read_text()
    key_id = ""
    for line in env_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("KALSHI_API_KEY_ID="):
            key_id = stripped.split("=", 1)[1].strip()
            break

    if not key_id or key_id == "your-api-key-id-here":
        console.print("  [red]KALSHI_API_KEY_ID is not set (or still has the placeholder).[/]")
        console.print()
        console.print(Panel(
            "[bold]How to get your API key:[/]\n\n"
            f"1. Go to [link={DEMO_SIGNUP_URL}]{DEMO_SIGNUP_URL}[/link]\n"
            "   - Create a free demo account if you don't have one\n"
            "   - Demo uses fake money -- no risk\n\n"
            f"2. Go to [link={API_KEYS_URL}]{API_KEYS_URL}[/link]\n"
            "   - Click 'Create New API Key'\n"
            "   - Copy the API Key ID\n\n"
            "3. Edit .env and set:\n"
            "   KALSHI_API_KEY_ID=<paste your key ID here>",
            title="Action Required",
            border_style="yellow",
        ))
        return False

    console.print(f"  [green]API key ID found:[/] {key_id[:8]}...{key_id[-4:]}" if len(key_id) > 12 else f"  [green]API key ID found:[/] {key_id}")
    return True


def step_private_key() -> bool:
    """Step 3: Check private key PEM file exists."""
    console.print("\n[bold cyan]Step 3/5:[/] Checking private key PEM file...")

    # Read the configured path from .env
    env_text = ENV_FILE.read_text()
    key_path_str = "./keys/kalshi-private.pem"  # default
    for line in env_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("KALSHI_PRIVATE_KEY_PATH="):
            key_path_str = stripped.split("=", 1)[1].strip()
            break

    key_path = Path(key_path_str)
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path

    if not key_path.exists():
        console.print(f"  [red]Private key not found at: {key_path}[/]")
        console.print()
        console.print(Panel(
            "[bold]How to set up your private key:[/]\n\n"
            f"1. Go to [link={API_KEYS_URL}]{API_KEYS_URL}[/link]\n"
            "   - When you create an API key, it downloads a .pem file\n\n"
            f"2. Create the keys directory:\n"
            f"   mkdir -p {PROJECT_ROOT / 'keys'}\n\n"
            f"3. Move/copy the .pem file:\n"
            f"   cp ~/Downloads/*.pem {key_path}\n\n"
            "4. Make sure .env has the correct path:\n"
            f"   KALSHI_PRIVATE_KEY_PATH={key_path_str}",
            title="Action Required",
            border_style="yellow",
        ))
        return False

    # Basic sanity check: does it look like a PEM file?
    content = key_path.read_text()
    if "PRIVATE KEY" not in content:
        console.print(f"  [red]File exists but doesn't look like a PEM private key: {key_path}[/]")
        console.print("  [bold]Fix:[/] Make sure you downloaded the correct file from Kalshi.")
        return False

    console.print(f"  [green]Private key found at:[/] {key_path}")
    return True


def step_load_settings():
    """Step 4: Load settings to confirm pydantic parses everything."""
    console.print("\n[bold cyan]Step 4/5:[/] Loading configuration...")

    try:
        from config.settings import settings
        console.print(f"  [green]Environment:[/] {settings.kalshi_env}")
        console.print(f"  [green]Base URL:[/]    {settings.base_url}")

        if settings.kalshi_env != "demo":
            console.print("  [yellow]Warning: KALSHI_ENV is not 'demo'. For initial setup, use demo.[/]")
            console.print("  [bold]Fix:[/] Set KALSHI_ENV=demo in .env")
            return False

        return True
    except Exception as e:
        console.print(f"  [red]Failed to load settings: {e}[/]")
        return False


def step_verify_auth() -> bool:
    """Step 5: Make an authenticated API call to verify credentials."""
    console.print("\n[bold cyan]Step 5/5:[/] Verifying authentication with Kalshi demo API...")

    try:
        from core.rest_client import KalshiClient
        client = KalshiClient()
        balance_resp = client.get_balance()

        # client.get_balance() already converts cents → Decimal dollars
        balance_dollars = balance_resp.get("balance", 0)

        console.print()
        console.print(Panel(
            f"[bold green]Authentication successful![/]\n\n"
            f"  Balance: [bold]${balance_dollars:,.2f}[/]\n"
            f"  Response: {balance_resp}",
            title="Demo Account Verified",
            border_style="green",
        ))
        return True

    except FileNotFoundError as e:
        console.print(f"  [red]Key file error: {e}[/]")
        return False
    except Exception as e:
        error_msg = str(e)
        console.print(f"  [red]Authentication failed: {error_msg}[/]")
        console.print()

        if "401" in error_msg or "403" in error_msg:
            console.print(Panel(
                "[bold]Your API key or private key is invalid.[/]\n\n"
                "Common causes:\n"
                "  - API key ID doesn't match the private key\n"
                "  - Key was created on prod but KALSHI_ENV=demo (or vice versa)\n"
                "  - Key has been revoked\n\n"
                "Fix:\n"
                f"  1. Go to {API_KEYS_URL}\n"
                "  2. Delete existing keys and create a fresh one\n"
                "  3. Update .env with the new KALSHI_API_KEY_ID\n"
                "  4. Save the new .pem file to keys/kalshi-private.pem\n"
                "  5. Re-run this script",
                title="Auth Error",
                border_style="red",
            ))
        elif "Connection" in error_msg or "Timeout" in error_msg:
            console.print("  [yellow]Network error -- check your internet connection.[/]")
        else:
            console.print("  [yellow]Unexpected error. Check that your .env is correct and try again.[/]")

        return False


def main():
    print_header()

    # Step 1: .env file
    if not step_env_file():
        return

    # Step 2: API key
    if not step_api_key():
        console.print("\n[bold yellow]Setup incomplete.[/] Fix the issue above and re-run:")
        console.print("  python -m scripts.setup_demo")
        return

    # Step 3: Private key
    if not step_private_key():
        console.print("\n[bold yellow]Setup incomplete.[/] Fix the issue above and re-run:")
        console.print("  python -m scripts.setup_demo")
        return

    # Step 4: Load config
    if not step_load_settings():
        console.print("\n[bold yellow]Setup incomplete.[/] Fix the issue above and re-run:")
        console.print("  python -m scripts.setup_demo")
        return

    # Step 5: Verify auth
    if not step_verify_auth():
        console.print("\n[bold yellow]Setup incomplete.[/] Fix the issue above and re-run:")
        console.print("  python -m scripts.setup_demo")
        return

    # All passed
    console.print()
    console.print(Panel(
        "[bold green]Your demo account is fully set up![/]\n\n"
        "Next steps:\n"
        "  1. python -m scripts.smoke_test        -- full system check\n"
        "  2. python -m scripts.scan_markets       -- find trading opportunities\n"
        "  3. python -m scripts.watch_orderbook    -- watch a live market",
        title="Setup Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
