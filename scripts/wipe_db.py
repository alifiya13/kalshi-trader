#!/usr/bin/env python3
"""
Clean-slate DB wipe — DELETE every row from every table (keeps schema).

Pivot to weather-only + llm-council needs a fresh database. This empties
all tables in foreign-key-safe order (children before parents) so the
DELETEs don't trip referential-integrity constraints, then prints the
resulting row count for each table (every one should read 0).

Tables are NOT dropped — only emptied. Schema/migrations are untouched.

Run:
  python -m scripts.wipe_db            # prompts for confirmation
  python -m scripts.wipe_db --yes       # skip the prompt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from data.db import engine
from config.settings import settings

# FK-safe delete order: children (rows holding FKs) before parents.
#   fills   -> orders
#   orders  -> markets, signals
#   signals -> markets
# The rest have no FK dependencies.
DELETE_ORDER = [
    "fills",
    "orders",
    "signals",
    "markets",
    "positions",
    "paper_trades",
    "debate_logs",
    "council_decisions",
    "portfolio_snapshots",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe all data from every table")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    # Show where we're pointed (redact the password) before doing anything.
    url = settings.database_url
    safe = url
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        safe = f"{scheme}://{user}:***@{host}"
    print(f"Target database: {safe}")
    print(f"Tables to empty (in order): {', '.join(DELETE_ORDER)}")

    if not args.yes:
        resp = input("\nDELETE ALL ROWS from every table above? Type 'wipe' to confirm: ")
        if resp.strip().lower() != "wipe":
            print("Aborted.")
            return 1

    with engine.begin() as conn:
        for table in DELETE_ORDER:
            conn.execute(text(f"DELETE FROM {table}"))
            print(f"  emptied {table}")

    # Verify: every table should now report 0 rows.
    print("\nRow counts after wipe:")
    all_zero = True
    with engine.connect() as conn:
        for table in DELETE_ORDER:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            flag = "" if count == 0 else "  <-- NOT EMPTY"
            if count != 0:
                all_zero = False
            print(f"  {table:<22} {count}{flag}")

    print("\n" + ("All tables empty — clean slate confirmed." if all_zero
                  else "WARNING: some tables still have rows."))
    return 0 if all_zero else 2


if __name__ == "__main__":
    raise SystemExit(main())
