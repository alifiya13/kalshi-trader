"""
Market Scanner

Pulls all open markets from Kalshi, stores them in the database,
and provides filtered views for strategy modules.

Kalshi API field-name note (as of 2026-04):
The `/markets` list endpoint uses these fields (all prices in dollars):
    volume_fp, volume_24h_fp         — lifetime / 24h traded contracts
    liquidity_dollars                — resting book notional
    yes_bid_dollars, yes_ask_dollars — top-of-book YES prices
    no_bid_dollars,  no_ask_dollars  — top-of-book NO prices
    yes_bid_size_fp, yes_ask_size_fp — top-of-book sizes
    open_interest_fp                 — open interest
    event_ticker                     — parent event (categories live on /events or /series)

There is NO `category` field on the market list response. We infer one from
the ticker prefix (see `_infer_category`).
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

import structlog

from core.rest_client import KalshiClient
from data.db import Market, get_session, init_db

logger = structlog.get_logger()


# ----------------------------------------------------------------------
# Field helpers — tolerate strings, None, and missing keys
# ----------------------------------------------------------------------

def _num(raw, default=0.0) -> float:
    """Safely coerce a Kalshi numeric field (often returned as a string)."""
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def mkt_volume(m: dict) -> float:
    return _num(m.get("volume_fp"))


def mkt_volume_24h(m: dict) -> float:
    return _num(m.get("volume_24h_fp"))


def mkt_liquidity(m: dict) -> float:
    return _num(m.get("liquidity_dollars"))


def mkt_yes_bid(m: dict) -> float:
    return _num(m.get("yes_bid_dollars"))


def mkt_yes_ask(m: dict) -> float:
    return _num(m.get("yes_ask_dollars"))


def mkt_no_bid(m: dict) -> float:
    return _num(m.get("no_bid_dollars"))


def mkt_no_ask(m: dict) -> float:
    return _num(m.get("no_ask_dollars"))


def mkt_total_resting_size(m: dict) -> float:
    """Top-of-book YES+NO resting size. Good proxy for 'is this book live'."""
    return _num(m.get("yes_bid_size_fp")) + _num(m.get("no_bid_size_fp"))


# ----------------------------------------------------------------------
# Category inference from ticker prefix
# ----------------------------------------------------------------------
# Kalshi doesn't return a category on list responses, so we bucket by
# ticker prefix. Ordered list: first match wins.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("parlay",   ("KXMVE",)),
    ("crypto",   ("KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXCRYPTO")),
    ("politics", ("KXPRES", "KXSENATE", "KXHOUSE", "KXGOV", "KXELECT", "KXPERSONPRES",
                  "KXG7", "KXNEXTISRAEL", "KXXI", "KXPOPE", "KXNATO")),
    ("macro",    ("KXCPI", "KXPAYROLLS", "KXFED", "KXRATE", "KXGDP", "KXPPI",
                  "KXUNEMP", "KXINFL", "KXJOBS")),
    ("sp500",    ("KXSPX", "KXSP500", "KXSPY", "KXSPD", "KXSPQ")),
    ("weather",  ("KXTEMP", "KXWEATHER", "KXHIGH", "KXLOW", "KXRAIN", "KXSNOW")),
    ("nba",      ("KXNBA",)),
    ("mlb",      ("KXMLB",)),
    ("nfl",      ("KXNFL",)),
    ("nhl",      ("KXNHL",)),
    ("tennis",   ("KXITF", "KXATP", "KXWTA")),
    ("golf",     ("KXPGA", "KXLPGA", "KXMASTERS")),
    ("soccer",   ("KXMLS", "KXEPL", "KXUEFA", "KXFIFA", "KXUCL")),
    ("ufc",      ("KXUFC", "KXMMA")),
]


def infer_category(ticker: str) -> str:
    if not ticker:
        return "other"
    for category, prefixes in _CATEGORY_RULES:
        if ticker.startswith(prefixes):
            return category
    return "other"


class MarketScanner:
    """Discovers and catalogs Kalshi markets."""

    def __init__(self, client: KalshiClient):
        self.client = client

    def scan_all_open(
        self,
        max_pages: int | None = None,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> list[dict]:
        """
        Pull open markets using cursor pagination.
        Returns raw market dicts and stores them in DB.

        Args:
            max_pages: hard cap on pages fetched. None means unlimited — use
                with care on prod, which has hundreds of thousands of open
                markets (mostly zero-volume parlay permutations).
            exclude_prefixes: ticker-prefix filter applied client-side as each
                page arrives. Matches via str.startswith(). Useful to drop
                ("KXMVE",) — the user-generated multi-game/cross-category
                parlay markets that dominate the prod catalog with no volume.
        """
        all_markets = []
        cursor = None
        page = 0
        raw_fetched = 0
        excluded = 0

        while True:
            page += 1
            if max_pages is not None and page > max_pages:
                logger.info("scan_hit_max_pages", max_pages=max_pages)
                break

            resp = self.client.get_markets(status="open", limit=200, cursor=cursor)
            markets = resp.get("markets", [])

            if not markets:
                break

            raw_fetched += len(markets)

            if exclude_prefixes:
                before = len(markets)
                markets = [
                    m for m in markets
                    if not m.get("ticker", "").startswith(exclude_prefixes)
                ]
                excluded += before - len(markets)

            all_markets.extend(markets)
            cursor = resp.get("cursor")
            logger.info(
                "scan_page",
                page=page,
                kept=len(markets),
                total_kept=len(all_markets),
                raw_fetched=raw_fetched,
                excluded=excluded,
            )

            if not cursor:
                break

        # Store in DB
        self._upsert_markets(all_markets)
        logger.info(
            "scan_complete",
            total_kept=len(all_markets),
            raw_fetched=raw_fetched,
            excluded=excluded,
        )
        return all_markets

    def scan_series(self, series_ticker: str) -> list[dict]:
        """Pull all open markets for a specific series."""
        resp = self.client.get_markets(series_ticker=series_ticker, status="open", limit=200)
        markets = resp.get("markets", [])
        self._upsert_markets(markets)
        logger.info("scan_series", series=series_ticker, count=len(markets))
        return markets

    @staticmethod
    def _parse_close_time(raw: Optional[str]) -> Optional[datetime]:
        """Parse ISO 8601 close_time string to datetime, or return None."""
        if not raw:
            return None
        try:
            # Handle both "2026-04-09T17:00:00Z" and "2026-04-09T17:00:00+00:00"
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _upsert_markets(self, markets: list[dict]):
        """
        Insert or update markets in database.

        yes_bid_dollars / no_bid_dollars / volume_fp come from the Kalshi
        list endpoint as strings in dollar units (e.g. "0.2900"). The DB
        stores them as Decimal dollars.
        """
        session = get_session()
        try:
            for m in markets:
                ticker = m["ticker"]
                yes_bid = Decimal(str(mkt_yes_bid(m)))
                no_bid = Decimal(str(mkt_no_bid(m)))
                volume = Decimal(str(mkt_volume(m)))

                existing = session.get(Market, ticker)
                if existing:
                    existing.status = m.get("status")
                    existing.yes_bid = yes_bid
                    existing.no_bid = no_bid
                    existing.volume = volume
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    session.add(Market(
                        ticker=ticker,
                        series_ticker=m.get("series_ticker"),
                        event_ticker=m.get("event_ticker"),
                        title=m.get("title"),
                        category=infer_category(ticker),
                        status=m.get("status"),
                        yes_bid=yes_bid,
                        no_bid=no_bid,
                        volume=volume,
                        close_time=self._parse_close_time(m.get("close_time")),
                    ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def filter_tradeable(markets: list[dict], min_volume: int = 50) -> list[dict]:
        """
        Filter markets that are actually worth trading:
        - Status is "active" (Kalshi) or "open" (legacy)
        - Lifetime volume above minimum OR has resting liquidity
        """
        out = []
        for m in markets:
            status = m.get("status")
            if status not in ("active", "open"):
                continue
            if mkt_volume(m) >= min_volume or mkt_liquidity(m) > 0:
                out.append(m)
        return out

    @staticmethod
    def filter_has_liquidity(markets: list[dict]) -> list[dict]:
        """Markets with any resting top-of-book size on either side."""
        return [m for m in markets if mkt_total_resting_size(m) > 0]

    @staticmethod
    def group_by_category(markets: list[dict]) -> dict[str, list[dict]]:
        """Group markets by inferred category (from ticker prefix)."""
        groups: dict[str, list[dict]] = {}
        for m in markets:
            cat = infer_category(m.get("ticker", ""))
            groups.setdefault(cat, []).append(m)
        return groups
