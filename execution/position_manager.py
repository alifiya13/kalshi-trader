"""
Position Manager — the brain of the active trading system.

Tracks all open positions, updates prices, evaluates exit conditions,
and decides when to hold or sell. Works in both live (demo orders)
and dry-run (logging only) modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

import structlog

from core.rest_client import KalshiClient
from data.db import Position, get_session, init_db

logger = structlog.get_logger()


class ExitAction(Enum):
    HOLD = "hold"
    SELL_PROFIT = "sell_profit"
    SELL_STOP_LOSS = "sell_stop_loss"
    HOLD_TO_SETTLE = "hold_to_settle"


@dataclass
class ActivePosition:
    """In-memory view of a tracked position."""
    db_id: int
    ticker: str
    side: str
    entry_price: Decimal
    contracts: int
    entry_time: datetime
    strategy: str
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    status: str = "open"
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    market_close_time: Optional[datetime] = None

    @property
    def cost(self) -> Decimal:
        return self.entry_price * self.contracts


# Exit rule thresholds
PROFIT_TARGET = Decimal("0.15")   # 15 cent gain per contract
STOP_LOSS = Decimal("0.10")       # 10 cent loss per contract
CLOSE_MINUTES = 60                # hold-to-settle if market closes within this
NEAR_CERTAIN_YES = Decimal("0.97")
NEAR_CERTAIN_NO = Decimal("0.03")


class PositionManager:
    """
    Loads open positions from DB, updates live prices, evaluates exit
    rules, and executes exits (or logs them in dry-run mode).
    """

    def __init__(self, client: KalshiClient, dry_run: bool = True):
        self.client = client
        self.dry_run = dry_run
        self.positions: list[ActivePosition] = []

    def load_open_positions(self) -> list[ActivePosition]:
        """Load all open positions from the database."""
        session = get_session()
        try:
            rows = session.query(Position).filter(Position.status == "open").all()
            self.positions = []
            for r in rows:
                self.positions.append(ActivePosition(
                    db_id=r.id,
                    ticker=r.ticker,
                    side=r.side,
                    entry_price=Decimal(str(r.entry_price)),
                    contracts=r.contracts,
                    entry_time=r.entry_time or datetime.now(timezone.utc),
                    strategy=r.strategy or "unknown",
                    current_price=Decimal(str(r.current_price or 0)),
                    unrealized_pnl=Decimal(str(r.unrealized_pnl or 0)),
                    status=r.status or "open",
                    market_close_time=r.market_close_time,
                ))
        finally:
            session.close()
        return self.positions

    def update_prices(self) -> list[ActivePosition]:
        """Fetch current market prices for all open positions."""
        for pos in self.positions:
            if pos.status != "open":
                continue
            try:
                resp = self.client.get_market(pos.ticker)
                m = resp.get("market", resp) if isinstance(resp, dict) else {}
                status = m.get("status", "")

                # Check if market has settled
                if status in ("settled", "finalized"):
                    result = (m.get("result") or "").lower()
                    if result:
                        won = pos.side.lower() == result.lower()
                        if won:
                            pos.current_price = Decimal("1.00")
                            pos.unrealized_pnl = (Decimal("1") - pos.entry_price) * pos.contracts
                        else:
                            pos.current_price = Decimal("0.00")
                            pos.unrealized_pnl = -pos.entry_price * pos.contracts
                        self._settle_position(pos, result)
                        continue

                # Live market — get current YES price
                yes_bid = Decimal(str(m.get("yes_bid_dollars") or "0"))
                yes_ask = Decimal(str(m.get("yes_ask_dollars") or "0"))

                if pos.side == "yes":
                    # Our position value is what we could sell at (the bid)
                    pos.current_price = yes_bid if yes_bid > 0 else (yes_bid + yes_ask) / 2
                else:
                    # NO position value = 1 - yes_ask (what we'd get selling NO)
                    no_bid = Decimal("1") - yes_ask if yes_ask > 0 else Decimal("0")
                    pos.current_price = no_bid if no_bid > 0 else Decimal("0.50")

                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.contracts

                # Update close_time if available
                close_time_str = m.get("close_time")
                if close_time_str:
                    try:
                        pos.market_close_time = datetime.fromisoformat(
                            close_time_str.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                # Persist updated price to DB
                self._update_db_price(pos)

            except Exception as e:
                logger.warning("price_update_failed", ticker=pos.ticker, error=str(e))

        return self.positions

    def evaluate_exits(self) -> list[tuple[ActivePosition, ExitAction]]:
        """
        For each open position, decide: HOLD, SELL_PROFIT, SELL_STOP_LOSS,
        or HOLD_TO_SETTLE.
        """
        decisions: list[tuple[ActivePosition, ExitAction]] = []
        now = datetime.now(timezone.utc)

        for pos in self.positions:
            if pos.status != "open":
                continue

            price_change = pos.current_price - pos.entry_price

            # Rule 1: Near certain win — hold to settlement
            if pos.side == "yes" and pos.current_price >= NEAR_CERTAIN_YES:
                decisions.append((pos, ExitAction.HOLD_TO_SETTLE))
                continue
            if pos.side == "no" and pos.current_price >= (Decimal("1") - NEAR_CERTAIN_NO):
                decisions.append((pos, ExitAction.HOLD_TO_SETTLE))
                continue

            # Rule 2: Market closing soon AND profitable — hold to settle
            if pos.market_close_time:
                minutes_left = (pos.market_close_time - now).total_seconds() / 60
                if minutes_left < CLOSE_MINUTES and price_change > 0:
                    decisions.append((pos, ExitAction.HOLD_TO_SETTLE))
                    continue

            # Rule 3: Profit target hit
            if price_change >= PROFIT_TARGET:
                decisions.append((pos, ExitAction.SELL_PROFIT))
                continue

            # Rule 4: Stop loss hit
            if price_change <= -STOP_LOSS:
                decisions.append((pos, ExitAction.SELL_STOP_LOSS))
                continue

            # Default: hold
            decisions.append((pos, ExitAction.HOLD))

        return decisions

    def execute_exit(self, pos: ActivePosition, action: ExitAction) -> bool:
        """
        Execute an exit. In dry_run mode, just logs. Otherwise places
        a sell order via the order executor.
        """
        if action in (ExitAction.HOLD, ExitAction.HOLD_TO_SETTLE):
            return False  # no action needed

        exit_reason = "profit_target" if action == ExitAction.SELL_PROFIT else "stop_loss"
        exit_price = pos.current_price
        realized_pnl = (exit_price - pos.entry_price) * pos.contracts

        if self.dry_run:
            logger.info(
                "dry_run_exit",
                ticker=pos.ticker,
                side=pos.side,
                reason=exit_reason,
                entry=float(pos.entry_price),
                exit=float(exit_price),
                pnl=float(realized_pnl),
            )
        else:
            # Import here to avoid circular imports
            from execution.order_executor import OrderExecutor
            executor = OrderExecutor(self.client, dry_run=False)
            price_cents = int(float(exit_price) * 100)
            executor.place_sell(pos.ticker, pos.side, price_cents, pos.contracts)

        # Update DB
        self._close_position(pos, exit_price, exit_reason, realized_pnl)
        return True

    def open_position_count(self) -> int:
        return sum(1 for p in self.positions if p.status == "open")

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _update_db_price(self, pos: ActivePosition):
        session = get_session()
        try:
            row = session.get(Position, pos.db_id)
            if row:
                row.current_price = pos.current_price
                row.unrealized_pnl = pos.unrealized_pnl
                if pos.market_close_time:
                    row.market_close_time = pos.market_close_time
                session.commit()
        finally:
            session.close()

    def _close_position(
        self, pos: ActivePosition, exit_price: Decimal,
        exit_reason: str, realized_pnl: Decimal,
    ):
        session = get_session()
        try:
            row = session.get(Position, pos.db_id)
            if row:
                row.exit_price = exit_price
                row.exit_time = datetime.now(timezone.utc)
                row.exit_reason = exit_reason
                row.realized_pnl = realized_pnl
                row.status = "closed_profit" if realized_pnl >= 0 else "closed_loss"
                session.commit()

            pos.status = row.status if row else "closed_loss"
            pos.exit_price = exit_price
            pos.exit_reason = exit_reason
        finally:
            session.close()

    def _settle_position(self, pos: ActivePosition, market_result: str):
        session = get_session()
        try:
            row = session.get(Position, pos.db_id)
            if row:
                won = pos.side.lower() == market_result.lower()
                if won:
                    row.realized_pnl = (Decimal("1") - pos.entry_price) * pos.contracts
                else:
                    row.realized_pnl = -pos.entry_price * pos.contracts
                row.exit_price = Decimal("1") if won else Decimal("0")
                row.exit_time = datetime.now(timezone.utc)
                row.exit_reason = "settled"
                row.market_result = market_result
                row.status = "settled"
                session.commit()

            pos.status = "settled"
            pos.exit_reason = "settled"
        finally:
            session.close()
