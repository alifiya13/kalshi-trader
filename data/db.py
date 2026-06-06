"""
Database models and connection.

Uses SQLAlchemy ORM. Starts with SQLite for simplicity —
swap DATABASE_URL to PostgreSQL when you're ready to scale.
"""

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Column, DateTime, Integer, Numeric, String, Text, JSON,
    ForeignKey, create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship

from config.settings import settings


class Base(DeclarativeBase):
    pass


class Market(Base):
    __tablename__ = "markets"

    ticker = Column(String, primary_key=True)
    series_ticker = Column(String, index=True)
    event_ticker = Column(String, index=True)
    title = Column(Text)
    category = Column(String, index=True)
    status = Column(String, index=True)
    yes_bid = Column(Numeric)
    no_bid = Column(Numeric)
    volume = Column(Numeric)
    close_time = Column(DateTime(timezone=True))
    result = Column(String, nullable=True)  # "yes", "no", or null
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_ticker = Column(String, ForeignKey("markets.ticker"), index=True)
    strategy = Column(String, index=True)
    model_prob = Column(Numeric)
    market_prob = Column(Numeric)
    edge = Column(Numeric)
    confidence = Column(Numeric)
    features = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True)
    client_order_id = Column(String, unique=True)
    market_ticker = Column(String, ForeignKey("markets.ticker"), index=True)
    strategy = Column(String)
    action = Column(String)  # "buy" or "sell"
    side = Column(String)    # "yes" or "no"
    price_cents = Column(Integer)
    count = Column(Integer)
    status = Column(String, index=True)
    fill_count = Column(Integer, default=0)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    fills = relationship("Fill", back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    fill_id = Column(String, primary_key=True)
    order_id = Column(String, ForeignKey("orders.order_id"), index=True)
    market_ticker = Column(String)
    side = Column(String)
    price_cents = Column(Integer)
    count = Column(Integer)
    trade_fee = Column(Numeric)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    order = relationship("Order", back_populates="fills")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Numeric)
    positions_value = Column(Numeric)
    unrealized_pnl = Column(Numeric)
    realized_pnl = Column(Numeric)
    snapshot_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PaperTrade(Base):
    """
    A logged but NOT-executed trade used for strategy validation.

    LEGACY: this table belonged to the old standalone paper-trader pipeline
    (since removed). It is retained because the dashboard still reads
    historical rows, but nothing writes to it anymore — the current flow logs
    decisions to `council_decisions` and paper positions to `positions`.
    """
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    strategy = Column(String, index=True)
    side = Column(String)                      # "yes" or "no"
    entry_price = Column(Numeric)              # per-contract dollar cost at entry
    contracts = Column(Integer)
    cost = Column(Numeric)                     # entry_price * contracts
    potential_profit = Column(Numeric)         # (1 - entry_price) * contracts

    # Signal snapshot — frozen at trade time for post-hoc analysis
    signal_edge = Column(Numeric)
    signal_confidence = Column(String)
    model_prob = Column(Numeric)
    market_prob = Column(Numeric)
    nws_high = Column(Integer, nullable=True)

    placed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    settled_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(String, nullable=True)     # "yes" / "no" (market result), or None
    pnl = Column(Numeric, nullable=True)       # signed dollars


class DebateLog(Base):
    """
    Every AI debate call is logged here for later calibration.

    Rows are written when the debate runs (outcome fields null). Once
    the market settles, a nightly job fills in `market_result` and
    `was_correct` so we can compute Brier scores per model weekly.
    """
    __tablename__ = "debate_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    market_title = Column(Text)

    bull_prob = Column(Numeric)
    bear_prob = Column(Numeric)
    judge_prob = Column(Numeric)
    disagreement = Column(Numeric)

    market_price = Column(Numeric)
    edge = Column(Numeric)
    side = Column(String)           # "yes" | "no" | "hold"
    confidence = Column(Numeric)    # post disagreement-penalty
    should_trade = Column(Integer)  # 0/1 as SQLite has no bool

    total_cost = Column(Numeric)    # USD spent on this debate

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Filled in after settlement
    market_result = Column(String, nullable=True)   # "yes" | "no"
    was_correct = Column(Integer, nullable=True)    # 0/1


class CouncilDecision(Base):
    """
    One TRADE recommended by a 3-stage WeatherCouncil run (agents/council.py).

    Event-level model (2026-06-06): the council runs ONCE per weather event,
    sees ALL brackets, and recommends 1+ trades. Each recommended trade gets
    its own row; rows from the same run share a `council_run_id` (UUID) and
    the same stage-1/2/3 reasoning blobs. Per-trade fields (`ticker`,
    `stage3_final_prob` = P(this trade wins), `stage3_side`,
    `trade_reasoning`, prices, `edge`) differ per row.

    The council (adapted from Karpathy's llm-council) is a research instrument
    for studying *failure modes*, so we persist every stage and every model's
    prediction + reasoning chain. The Stage-1/2 columns are POSITIONAL
    (model_a / model_b / model_c) — the A/B/C labels map to
    WeatherCouncil.council_models in order (also stored verbatim in the
    `council_models` JSON column). Outcome columns (market_result,
    was_correct) stay null until the market settles and a reconciliation job
    fills them in.
    """
    __tablename__ = "council_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    council_run_id = Column(String, index=True)   # UUID — groups trades from one run
    event_ticker = Column(String, index=True)     # e.g. KXHIGHCHI-26JUN07 — dedup key
    city = Column(String, index=True)             # parsed from the event title
    temp_type = Column(String)                    # "high" | "low"
    ticker = Column(String, index=True)
    market_title = Column(Text)

    # --- Event-level temperature predictions (shared across the run) ---
    # "temp" = the daily extreme the event asks about (high or low).
    predicted_temp_f = Column(Numeric, nullable=True)   # chairman's final prediction
    stage1_model_a_predicted_temp = Column(Numeric, nullable=True)
    stage1_model_b_predicted_temp = Column(Numeric, nullable=True)
    stage1_model_c_predicted_temp = Column(Numeric, nullable=True)
    stage2_model_a_updated_temp = Column(Numeric, nullable=True)
    stage2_model_b_updated_temp = Column(Numeric, nullable=True)
    stage2_model_c_updated_temp = Column(Numeric, nullable=True)

    # --- Stage 1: independent analysis (positional A/B/C) ---
    # prob/side are legacy (per-bracket era); reasoning now holds the model's
    # predicted high + full trade list as one blob.
    stage1_model_a_prob = Column(Numeric)
    stage1_model_a_side = Column(String)
    stage1_model_a_reasoning = Column(Text)
    stage1_model_b_prob = Column(Numeric)
    stage1_model_b_side = Column(String)
    stage1_model_b_reasoning = Column(Text)
    stage1_model_c_prob = Column(Numeric)
    stage1_model_c_side = Column(String)
    stage1_model_c_reasoning = Column(Text)

    # --- Stage 2: peer review (updated predictions) ---
    stage2_model_a_updated_prob = Column(Numeric)
    stage2_model_a_reasoning = Column(Text)
    stage2_model_b_updated_prob = Column(Numeric)
    stage2_model_b_reasoning = Column(Text)
    stage2_model_c_updated_prob = Column(Numeric)
    stage2_model_c_reasoning = Column(Text)

    # --- Stage 3: chairman synthesis ---
    stage3_final_prob = Column(Numeric)     # P(this trade WINS) — per trade
    stage3_confidence = Column(Numeric)
    stage3_should_trade = Column(Integer)   # always 1 now (council must trade)
    stage3_side = Column(String)            # "yes" | "no" — per trade
    stage3_reasoning = Column(Text)         # chairman's OVERALL synthesis (shared)
    stage3_dissent_summary = Column(Text)
    stage3_risk_factors = Column(Text)
    trade_reasoning = Column(Text)          # chairman's reasoning for THIS trade

    # --- Market snapshot + edge at decision time ---
    market_yes_price = Column(Numeric)
    market_no_price = Column(Numeric)
    edge = Column(Numeric)
    # NWS forecast for the event's variable (high OR low). Name kept from
    # the NYC-high era because the dashboard reads it.
    weather_nws_high = Column(Integer, nullable=True)

    # --- Bookkeeping ---
    council_models = Column(JSON, nullable=True)   # [model_a, model_b, model_c]
    chairman_model = Column(String, nullable=True)
    total_cost_usd = Column(Numeric)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # --- Filled in after settlement ---
    market_result = Column(String, nullable=True)   # "yes" | "no"
    was_correct = Column(Integer, nullable=True)    # 0/1


class Position(Base):
    """
    Active trading position tracked by the position manager.

    Lifecycle: open → closed_profit / closed_loss / settled
    """
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    strategy = Column(String, index=True)
    side = Column(String)                          # "yes" or "no"
    entry_price = Column(Numeric)                  # per-contract dollar cost
    contracts = Column(Integer)
    entry_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    current_price = Column(Numeric, nullable=True)
    unrealized_pnl = Column(Numeric, nullable=True)

    exit_price = Column(Numeric, nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    exit_reason = Column(String, nullable=True)    # "profit_target", "stop_loss", "settled", "manual"
    realized_pnl = Column(Numeric, nullable=True)

    status = Column(String, index=True, default="open")  # open, closed_profit, closed_loss, settled

    market_close_time = Column(DateTime(timezone=True), nullable=True)
    market_result = Column(String, nullable=True)  # "yes" / "no" after settlement


# ------------------------------------------------------------------
# Engine & Session
# ------------------------------------------------------------------

engine = create_engine(settings.database_url, echo=False)

# Enable WAL mode for SQLite (better concurrent reads)
if "sqlite" in settings.database_url:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session. Remember to close it."""
    return SessionLocal()
