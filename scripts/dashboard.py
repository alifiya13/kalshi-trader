#!/usr/bin/env python3
"""
=============================================================
  TRADING DASHBOARD — Streamlit
=============================================================

Simple dashboard showing:
  - Current balance
  - Open positions with live P&L
  - Closed trades history
  - Total realized P&L
  - Win/loss ratio

Run:  streamlit run scripts/dashboard.py
=============================================================
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from data.db import Position, PaperTrade, get_session, init_db
from core.rest_client import get_trade_client


def get_balance() -> Decimal:
    try:
        client = get_trade_client()
        resp = client.get_balance()
        bal = resp.get("balance", Decimal("0"))
        return bal if isinstance(bal, Decimal) else Decimal(str(bal))
    except Exception:
        return Decimal("0")


def main():
    st.set_page_config(page_title="Kalshi Trader Dashboard", layout="wide")
    st.title("Kalshi Trading Dashboard")

    init_db()
    session = get_session()

    try:
        # --- Balance ---
        balance = get_balance()
        col1, col2, col3, col4 = st.columns(4)

        # --- Open positions ---
        open_positions = session.query(Position).filter(
            Position.status == "open"
        ).all()

        # --- Closed positions ---
        closed_positions = session.query(Position).filter(
            Position.status.in_(["closed_profit", "closed_loss", "settled"])
        ).all()

        total_realized = sum(
            float(p.realized_pnl or 0) for p in closed_positions
        )
        total_unrealized = sum(
            float(p.unrealized_pnl or 0) for p in open_positions
        )
        wins = sum(1 for p in closed_positions if (p.realized_pnl or 0) > 0)
        losses = len(closed_positions) - wins
        win_rate = (wins / len(closed_positions) * 100) if closed_positions else 0

        col1.metric("Balance", f"${balance:,.2f}")
        col2.metric("Realized P&L", f"${total_realized:+,.2f}")
        col3.metric("Unrealized P&L", f"${total_unrealized:+,.2f}")
        col4.metric("Win Rate", f"{win_rate:.0f}% ({wins}W/{losses}L)")

        # --- Open Positions Table ---
        st.subheader(f"Open Positions ({len(open_positions)})")
        if open_positions:
            rows = []
            for p in open_positions:
                rows.append({
                    "Ticker": p.ticker,
                    "Side": p.side.upper() if p.side else "",
                    "Entry": f"${float(p.entry_price or 0):.2f}",
                    "Current": f"${float(p.current_price or 0):.2f}",
                    "Contracts": p.contracts,
                    "P&L": f"${float(p.unrealized_pnl or 0):+.2f}",
                    "Strategy": p.strategy or "",
                    "Entered": str(p.entry_time or "")[:19],
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No open positions.")

        # --- Closed Trades History ---
        st.subheader(f"Closed Trades ({len(closed_positions)})")
        if closed_positions:
            rows = []
            for p in sorted(closed_positions, key=lambda x: x.exit_time or datetime.min, reverse=True):
                rows.append({
                    "Ticker": p.ticker,
                    "Side": p.side.upper() if p.side else "",
                    "Entry": f"${float(p.entry_price or 0):.2f}",
                    "Exit": f"${float(p.exit_price or 0):.2f}",
                    "Contracts": p.contracts,
                    "P&L": f"${float(p.realized_pnl or 0):+.2f}",
                    "Reason": p.exit_reason or "",
                    "Strategy": p.strategy or "",
                    "Closed": str(p.exit_time or "")[:19],
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No closed trades yet.")

        # --- Paper Trades Performance ---
        st.subheader("Paper Trading Performance")
        settled_papers = session.query(PaperTrade).filter(
            PaperTrade.settled_at.isnot(None)
        ).all()
        if settled_papers:
            paper_pnl = sum(float(t.pnl or 0) for t in settled_papers)
            paper_wins = sum(1 for t in settled_papers if (t.pnl or 0) > 0)
            paper_losses = len(settled_papers) - paper_wins

            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Paper Trades", len(settled_papers))
            pcol2.metric("Paper P&L", f"${paper_pnl:+,.2f}")
            pcol3.metric("Paper Win Rate", f"{paper_wins}/{paper_wins + paper_losses}")

            rows = []
            for t in sorted(settled_papers, key=lambda x: x.settled_at or datetime.min, reverse=True):
                rows.append({
                    "Ticker": t.ticker,
                    "Side": t.side.upper() if t.side else "",
                    "Entry": f"${float(t.entry_price or 0):.2f}",
                    "Result": (t.result or "").upper(),
                    "P&L": f"${float(t.pnl or 0):+.2f}",
                    "Strategy": t.strategy or "",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No settled paper trades yet.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
