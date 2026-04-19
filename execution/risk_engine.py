"""
Risk Engine — the single source of truth for "can we take this trade?"

Blueprint v2 rules (ONE table, no conflicts across strategies):

    Confidence   Required edge   Max position
    ≥ 0.80       3¢              5% of portfolio
    ≥ 0.60       5¢              3% of portfolio
    ≥ 0.40       8¢              2% of portfolio
    < 0.40       DO NOT TRADE    0%

Plus portfolio-level guards:
    - Max 15 simultaneous open positions
    - Max 30% of portfolio exposure in any single category
    - 15% cash reserve minimum (never spend below it)
    - Daily loss > 5% → halt all new trades for 24h

Every strategy (weather, safe_compounder, ai_debate, ...) routes its
trade candidates through `RiskEngine.check_can_trade(...)`. If this
function returns (False, reason) the trade is skipped. No strategy
implements its own gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import structlog

from data.db import Position, get_session
from data.market_scanner import infer_category

logger = structlog.get_logger()


# --- Edge threshold table (blueprint v2, §Risk Engine) ---
THRESHOLDS: list[tuple[Decimal, Decimal, Decimal]] = [
    # (min_confidence, required_edge, max_position_pct)
    (Decimal("0.80"), Decimal("0.03"), Decimal("0.05")),
    (Decimal("0.60"), Decimal("0.05"), Decimal("0.03")),
    (Decimal("0.40"), Decimal("0.08"), Decimal("0.02")),
]

# Portfolio-level limits
MAX_OPEN_POSITIONS = 15
MAX_CATEGORY_EXPOSURE_PCT = Decimal("0.30")
CASH_RESERVE_PCT = Decimal("0.15")
DAILY_LOSS_HALT_PCT = Decimal("0.05")


@dataclass
class TradeSignal:
    """Uniform input from any strategy to the risk engine."""
    ticker: str
    side: str                    # "yes" or "no"
    strategy: str                # "weather_v1", "safe_compounder", ...
    edge: Decimal                # positive, in $ terms (probability units)
    confidence: Decimal          # 0..1
    cost_per_contract: Decimal   # $ per contract
    desired_contracts: int       # what the strategy wants to buy
    category: Optional[str] = None   # inferred from ticker if not set


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    approved_contracts: int = 0   # may be < desired_contracts
    approved_cost: Decimal = Decimal("0")
    max_position_pct: Decimal = Decimal("0")


def _threshold_row(confidence: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
    """Return (min_conf, required_edge, max_pct) row for this confidence level."""
    for row in THRESHOLDS:
        if confidence >= row[0]:
            return row
    return None


class RiskEngine:
    """
    Gate every trade through this. Inputs: the candidate signal + live
    portfolio state. Outputs: a RiskDecision with approved size (possibly
    smaller than desired) and a reason.
    """

    def __init__(self, portfolio_balance: Decimal):
        self.portfolio_balance = Decimal(str(portfolio_balance))

    # ------------------------------------------------------------------
    # Portfolio-state queries
    # ------------------------------------------------------------------

    def _open_positions(self) -> list[Position]:
        session = get_session()
        try:
            return session.query(Position).filter(Position.status == "open").all()
        finally:
            session.close()

    def _category_exposure(self, category: str, positions: list[Position]) -> Decimal:
        """Sum of entry cost for open positions in this category, as % of portfolio."""
        total = Decimal("0")
        for p in positions:
            if infer_category(p.ticker) != category:
                continue
            entry = Decimal(str(p.entry_price or 0))
            total += entry * p.contracts
        if self.portfolio_balance <= 0:
            return Decimal("0")
        return total / self.portfolio_balance

    def _cash_reserve_ok(self, new_cost: Decimal, positions: list[Position]) -> bool:
        """After committing new_cost, is cash still >= 15% of portfolio?"""
        committed = sum(
            (Decimal(str(p.entry_price or 0)) * p.contracts for p in positions),
            Decimal("0"),
        )
        remaining_cash = self.portfolio_balance - committed - new_cost
        min_required = self.portfolio_balance * CASH_RESERVE_PCT
        return remaining_cash >= min_required

    def _daily_loss_pct(self) -> Decimal:
        """Realized loss as fraction of portfolio over the last 24h (positive number)."""
        session = get_session()
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
            rows = (
                session.query(Position)
                .filter(Position.exit_time >= since, Position.realized_pnl.isnot(None))
                .all()
            )
            realized = sum(
                (Decimal(str(r.realized_pnl or 0)) for r in rows),
                Decimal("0"),
            )
            if realized >= 0 or self.portfolio_balance <= 0:
                return Decimal("0")
            return -realized / self.portfolio_balance
        finally:
            session.close()

    # ------------------------------------------------------------------
    # The one gate every strategy calls
    # ------------------------------------------------------------------

    def check_can_trade(self, signal: TradeSignal) -> RiskDecision:
        """Evaluate every risk rule in order. Return the first rejection, or approval."""
        # --- Gate 1: Confidence + edge from the threshold table ---
        row = _threshold_row(signal.confidence)
        if row is None:
            return RiskDecision(
                allowed=False,
                reason=f"Confidence {signal.confidence:.2f} < 0.40 (do-not-trade zone)",
            )
        _, required_edge, max_pct = row

        if signal.edge < required_edge:
            return RiskDecision(
                allowed=False,
                reason=(
                    f"Edge {signal.edge:.3f} < required {required_edge:.3f} "
                    f"at confidence {signal.confidence:.2f}"
                ),
                max_position_pct=max_pct,
            )

        # --- Gate 2: Daily loss circuit breaker ---
        daily_loss = self._daily_loss_pct()
        if daily_loss >= DAILY_LOSS_HALT_PCT:
            return RiskDecision(
                allowed=False,
                reason=f"Daily loss {daily_loss:.1%} >= halt {DAILY_LOSS_HALT_PCT:.0%}",
            )

        open_positions = self._open_positions()

        # --- Gate 3: Position count ---
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            return RiskDecision(
                allowed=False,
                reason=f"Already holding {len(open_positions)} positions (max {MAX_OPEN_POSITIONS})",
            )

        # --- Gate 4: Category exposure ---
        category = signal.category or infer_category(signal.ticker)
        existing_exposure = self._category_exposure(category, open_positions)
        desired_cost = signal.cost_per_contract * signal.desired_contracts
        new_category_exposure = existing_exposure + (desired_cost / self.portfolio_balance if self.portfolio_balance > 0 else Decimal("0"))

        approved_contracts = signal.desired_contracts
        if new_category_exposure > MAX_CATEGORY_EXPOSURE_PCT:
            # Scale down to fit under the cap
            remaining_category_budget = (MAX_CATEGORY_EXPOSURE_PCT - existing_exposure) * self.portfolio_balance
            if remaining_category_budget <= 0:
                return RiskDecision(
                    allowed=False,
                    reason=(
                        f"Category '{category}' already at "
                        f"{existing_exposure:.0%} (max {MAX_CATEGORY_EXPOSURE_PCT:.0%})"
                    ),
                )
            approved_contracts = min(
                approved_contracts,
                int(remaining_category_budget / signal.cost_per_contract),
            )

        # --- Gate 5: Per-position cap from threshold table ---
        max_position_cost = self.portfolio_balance * max_pct
        approved_contracts = min(
            approved_contracts,
            int(max_position_cost / signal.cost_per_contract),
        )

        if approved_contracts < 1:
            return RiskDecision(
                allowed=False,
                reason=f"Size shrunk to 0 contracts after position/category caps",
                max_position_pct=max_pct,
            )

        approved_cost = signal.cost_per_contract * approved_contracts

        # --- Gate 6: Cash reserve (15% always) ---
        if not self._cash_reserve_ok(approved_cost, open_positions):
            return RiskDecision(
                allowed=False,
                reason=f"Would breach 15% cash reserve",
                max_position_pct=max_pct,
            )

        return RiskDecision(
            allowed=True,
            reason=(
                f"OK edge={signal.edge:.3f} conf={signal.confidence:.2f} "
                f"→ {approved_contracts} contracts @ ${signal.cost_per_contract:.2f}"
            ),
            approved_contracts=approved_contracts,
            approved_cost=approved_cost,
            max_position_pct=max_pct,
        )
