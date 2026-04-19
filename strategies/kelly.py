"""
Position Sizing — Fractional Kelly Criterion for Kalshi Binary Markets

Binary contract math:
- Buy YES at price p → risk p, profit (1-p) if correct
- Buy NO  at price p → risk p, profit (1-p) if correct
- Kelly formula for binary bets:  f* = (p_model * b - q_model) / b
  where b = (1 - cost) / cost  (the odds), q_model = 1 - p_model

We use FRACTIONAL Kelly (default 0.25x) because:
1. Full Kelly assumes your model is perfectly calibrated (it's not)
2. Full Kelly has massive variance — 0.25x cuts variance by ~75%
3. Full Kelly maximizes growth rate; fractional Kelly maximizes Sharpe
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


@dataclass
class SizingResult:
    """Output of position sizing calculation."""
    should_trade: bool
    side: str                   # "yes" or "no"
    edge: Decimal               # model_prob - market_prob (signed)
    kelly_fraction: Decimal     # raw kelly bet fraction
    position_size_pct: Decimal  # fraction of portfolio to bet
    contracts: int              # number of contracts to buy
    cost_per_contract: Decimal  # price you'd pay per contract
    max_profit: Decimal         # profit if correct
    max_loss: Decimal           # loss if wrong
    expected_value: Decimal     # edge * contracts
    reason: str                 # why we did or didn't trade


def compute_kelly(
    model_prob: Decimal,
    market_yes_price: Decimal,
    portfolio_balance: Decimal,
    kelly_multiplier: Decimal = Decimal("0.25"),
    min_edge: Decimal = Decimal("0.05"),
    max_position_pct: Decimal = Decimal("0.02"),
    min_contracts: int = 1,
) -> SizingResult:
    """
    Compute optimal position size for a Kalshi binary market.

    Args:
        model_prob:        Your model's estimated probability (0.0 - 1.0)
        market_yes_price:  Current YES price in dollars (0.01 - 0.99)
        portfolio_balance: Your available cash
        kelly_multiplier:  Fraction of Kelly to use (0.25 = quarter Kelly)
        min_edge:          Minimum edge to trade (default $0.05)
        max_position_pct:  Maximum % of portfolio per trade
        min_contracts:     Minimum contracts to bother with

    Returns:
        SizingResult with trade decision, size, and risk metrics
    """
    model_prob = Decimal(str(model_prob))
    market_yes_price = Decimal(str(market_yes_price))

    # --- Determine side ---
    # If model_prob > market_yes_price → buy YES (event is underpriced)
    # If model_prob < market_yes_price → buy NO  (event is overpriced)
    yes_edge = model_prob - market_yes_price
    no_edge = (Decimal("1") - model_prob) - (Decimal("1") - market_yes_price)
    # no_edge simplifies to: market_yes_price - model_prob (just the negative of yes_edge)

    if yes_edge > 0:
        side = "yes"
        edge = yes_edge
        cost = market_yes_price
        win_prob = model_prob
    else:
        side = "no"
        edge = -yes_edge  # positive value for NO edge
        cost = Decimal("1") - market_yes_price
        win_prob = Decimal("1") - model_prob

    # --- Gate 1: Minimum edge ---
    if edge < min_edge:
        return SizingResult(
            should_trade=False, side=side, edge=edge,
            kelly_fraction=Decimal("0"), position_size_pct=Decimal("0"),
            contracts=0, cost_per_contract=cost,
            max_profit=Decimal("0"), max_loss=Decimal("0"),
            expected_value=Decimal("0"),
            reason=f"Edge {edge:.4f} below minimum {min_edge}",
        )

    # --- Gate 2: Sane cost ---
    if cost <= Decimal("0.01") or cost >= Decimal("0.99"):
        return SizingResult(
            should_trade=False, side=side, edge=edge,
            kelly_fraction=Decimal("0"), position_size_pct=Decimal("0"),
            contracts=0, cost_per_contract=cost,
            max_profit=Decimal("0"), max_loss=Decimal("0"),
            expected_value=Decimal("0"),
            reason=f"Price {cost} too extreme (near 0 or 1)",
        )

    # --- Kelly Criterion ---
    # For binary bet: f* = (p * b - q) / b
    # where b = payoff / cost = (1-cost)/cost, p = win_prob, q = 1-p
    payoff = Decimal("1") - cost
    b = payoff / cost
    q = Decimal("1") - win_prob

    kelly_raw = (win_prob * b - q) / b
    kelly_raw = max(Decimal("0"), kelly_raw)

    # Apply fractional Kelly
    kelly_bet = kelly_raw * kelly_multiplier

    # --- Position cap ---
    position_pct = min(kelly_bet, max_position_pct)

    # --- Convert to contracts ---
    dollar_amount = (portfolio_balance * position_pct).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    contracts = int(dollar_amount / cost)
    contracts = max(0, contracts)

    if contracts < min_contracts:
        return SizingResult(
            should_trade=False, side=side, edge=edge,
            kelly_fraction=kelly_raw, position_size_pct=position_pct,
            contracts=0, cost_per_contract=cost,
            max_profit=Decimal("0"), max_loss=Decimal("0"),
            expected_value=Decimal("0"),
            reason=f"Only {contracts} contracts (below min {min_contracts})",
        )

    # --- Risk metrics ---
    total_cost = cost * contracts
    max_profit = payoff * contracts
    max_loss = total_cost
    ev = edge * contracts

    return SizingResult(
        should_trade=True,
        side=side,
        edge=edge,
        kelly_fraction=kelly_raw,
        position_size_pct=position_pct,
        contracts=contracts,
        cost_per_contract=cost,
        max_profit=max_profit,
        max_loss=max_loss,
        expected_value=ev,
        reason=f"Edge={edge:.3f} Kelly={kelly_raw:.3f} → {contracts}x {side} @ ${cost}",
    )
