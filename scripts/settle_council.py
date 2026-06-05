#!/usr/bin/env python3
"""
=============================================================
  SETTLE COUNCIL — reconcile decisions + positions vs Kalshi
=============================================================

For the 1-week paper run we need to know, after the fact, whether the
council was right. This script:

  1. Finds every council_decisions row with market_result IS NULL.
  2. Fetches each market from Kalshi; if it's finalized with a result,
     records market_result and was_correct (did the council's stage-3 side
     match the settled outcome?).
  3. Settles any still-open rows in the positions table the same way
     (realized_pnl, exit_price, status='settled').
  4. Prints a summary of what was newly settled.

It's read-mostly + a few public API calls — NO LLM cost — so active_trader
runs it once per scan cycle.

Run standalone:
  python -m scripts.settle_council
=============================================================
"""

import sys
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.db import CouncilDecision, Position, get_session


def _market_settlement(client, ticker: str) -> Optional[str]:
    """
    Return "yes"/"no" if the market is finalized & resolved, else None.

    Mirrors how position_manager reads settlement: the /markets/{ticker}
    response wraps the market under a "market" key, with `status` in
    (finalized, settled) and a `result` of "yes"/"no" once resolved.
    """
    try:
        resp = client.get_market(ticker)
    except Exception:
        return None
    m = resp.get("market", resp) if isinstance(resp, dict) else {}
    status = (m.get("status") or "").lower()
    result = (m.get("result") or "").lower()
    if status in ("finalized", "settled") and result in ("yes", "no"):
        return result
    return None


def settle_council_decisions(client, verbose: bool = True) -> dict:
    """
    Settle pending council decisions and open positions against Kalshi.

    Returns a summary dict:
        {decisions_settled, positions_settled, checked, newly: [...]}
    """
    session = get_session()
    cache: dict[str, Optional[str]] = {}  # ticker -> "yes"/"no"/None (one API call each)

    def settlement(ticker: str) -> Optional[str]:
        if ticker not in cache:
            cache[ticker] = _market_settlement(client, ticker)
        return cache[ticker]

    decisions_settled = 0
    positions_settled = 0
    newly: list[dict] = []

    try:
        # --- 1. Council decisions awaiting a result ---
        pending = (
            session.query(CouncilDecision)
            .filter(CouncilDecision.market_result.is_(None))
            .all()
        )
        for d in pending:
            res = settlement(d.ticker)
            if res is None:
                continue
            council_side = (d.stage3_side or "").lower()
            d.market_result = res
            d.was_correct = 1 if council_side == res else 0
            decisions_settled += 1
            newly.append({
                "ticker": d.ticker,
                "council_side": council_side,
                "result": res,
                "was_correct": bool(d.was_correct),
                "should_trade": bool(d.stage3_should_trade),
                "confidence": float(d.stage3_confidence) if d.stage3_confidence is not None else None,
            })

        # --- 2. Open positions whose market has resolved ---
        open_positions = session.query(Position).filter(Position.status == "open").all()
        for p in open_positions:
            res = settlement(p.ticker)
            if res is None:
                continue
            won = (p.side or "").lower() == res
            entry = Decimal(str(p.entry_price or 0))
            p.realized_pnl = (Decimal("1") - entry) * p.contracts if won else -entry * p.contracts
            p.exit_price = Decimal("1") if won else Decimal("0")
            p.exit_time = datetime.now(timezone.utc)
            p.exit_reason = "settled"
            p.market_result = res
            p.status = "settled"
            positions_settled += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    summary = {
        "decisions_settled": decisions_settled,
        "positions_settled": positions_settled,
        "checked": len(cache),
        "newly": newly,
    }

    if verbose:
        print(
            f"Settlement: checked {len(cache)} market(s) · "
            f"settled {decisions_settled} decision(s), {positions_settled} position(s)"
        )
        for n in newly:
            mark = "✓ correct" if n["was_correct"] else "✗ wrong"
            print(
                f"  {n['ticker']:<26} council={n['council_side'].upper():<3} "
                f"result={n['result'].upper():<3} {mark}"
                f"  (decision={'TRADE' if n['should_trade'] else 'SKIP'})"
            )
        if not newly:
            print("  Nothing newly settled.")

    return summary


def main() -> int:
    from config.settings import ensure_key_files
    ensure_key_files()
    from core.rest_client import get_data_client
    client = get_data_client()
    settle_council_decisions(client, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
