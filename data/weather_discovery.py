"""
Dynamic weather-event discovery — NO hardcoded tickers, cities, or coords.

Pipeline (researched 2026-06-06):

    GET /events?status=open (paginate ALL)        Kalshi's category query
        │  filter category == "Climate and Weather"   param is silently
        ▼                                             ignored — the FIELD
    parse city + temp type from the event title      on each event works
        │  "Highest temperature in Chicago on …" → ("Chicago", "high")
        ▼
    geocode city via Nominatim (cached, 1 req/s)
        │
        ▼
    GET /markets?event_ticker=… → brackets (drop any closing past DEADLINE)
        │
        ▼
    WeatherEvent{event_ticker, series, title, city, lat, lon, temp_type,
                 close_time, brackets[...]}

Why title-parsing instead of a ticker→city table: Kalshi's series naming is
inconsistent (KXHIGHNY vs KXHIGHTATL vs KXHIGHLAX), so any prefix/ticker
mapping would be hardcoding that silently misses new cities. Titles are
uniform: "Highest|Lowest temperature in <CITY> on <date>?".

The full /markets catalog is NOT swept — it's 200k+ open markets dominated
by parlay permutations. Per-event market fetches only.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

import requests
import structlog

from core.rest_client import KalshiClient

logger = structlog.get_logger()

WEATHER_CATEGORY = "Climate and Weather"

# Research window: the paper run ends June 14, 2026. Markets closing after
# this are excluded so every decision settles inside the study.
DEADLINE_UTC = datetime(2026, 6, 14, 23, 59, 0, tzinfo=timezone.utc)

# "Highest temperature in Chicago on Jun 7, 2026?" → ("Highest", "Chicago")
# "Lowest temperature in Washington DC on Jun 7, 2026?" → ("Lowest", "Washington DC")
_TITLE_RE = re.compile(
    r"^(Highest|Lowest)\s+temperature\s+in\s+(.+?)\s+on\s+", re.IGNORECASE
)

# Nominatim usage policy: identify yourself, max 1 req/s.
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_UA = "kalshi-trader/weather-research (contact: dev@localhost)"
_GEOCODE_INTERVAL_S = 1.1

# In-process geocode cache: city name → (lat, lon) or None (failed lookup,
# cached too so we don't hammer Nominatim re-failing every scan).
_geocode_cache: dict[str, Optional[tuple[float, float]]] = {}
_last_geocode_ts: float = 0.0

# "KXHIGHCHI-26JUN07" → date(2026, 6, 7). Event tickers end with the date
# block (market tickers add a -B63.5 suffix; this regex handles both).
_TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


@dataclass
class WeatherEvent:
    """One discovered weather event with everything the council needs."""
    event_ticker: str
    series_ticker: str
    title: str
    city: str
    lat: float
    lon: float
    temp_type: str                      # "high" | "low"
    close_time: Optional[str]           # earliest bracket close (ISO)
    event_date: Optional[date]          # observation day, from the ticker
    brackets: list[dict] = field(default_factory=list)

    def as_council_event(self) -> dict:
        """The event_data dict agents/council.py expects."""
        return {
            "event_ticker": self.event_ticker,
            "event_date": self.event_date,
            "series_ticker": self.series_ticker,
            "city": self.city,
            "temp_type": self.temp_type,
            "brackets": self.brackets,
        }


def parse_event_date(ticker: str) -> Optional[date]:
    """Extract the observation date from an event or market ticker."""
    m = _TICKER_DATE_RE.search(ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        return date(2000 + int(yy), month, int(dd))
    except ValueError:
        return None


def parse_title(title: str) -> tuple[Optional[str], Optional[str]]:
    """
    ("Chicago", "high") from "Highest temperature in Chicago on Jun 7, 2026?".
    Returns (None, None) for non-temperature titles (rain, hurricanes, …).
    """
    m = _TITLE_RE.match(title or "")
    if not m:
        return None, None
    kind, city = m.groups()
    return city.strip(), "high" if kind.lower() == "highest" else "low"


def geocode_city(city: str) -> Optional[tuple[float, float]]:
    """
    City name → (lat, lon) via Nominatim. Cached per city; throttled to
    respect the 1 req/s usage policy. Returns None on failure (cached).
    """
    global _last_geocode_ts
    if city in _geocode_cache:
        return _geocode_cache[city]

    wait = _GEOCODE_INTERVAL_S - (time.monotonic() - _last_geocode_ts)
    if wait > 0:
        time.sleep(wait)

    try:
        r = requests.get(
            _NOMINATIM_URL,
            params={"q": f"{city}, USA", "format": "json", "limit": 1},
            headers={"User-Agent": _NOMINATIM_UA},
            timeout=15,
        )
        _last_geocode_ts = time.monotonic()
        r.raise_for_status()
        hits = r.json()
        if not hits:
            logger.warning("geocode_no_result", city=city)
            _geocode_cache[city] = None
            return None
        result = (float(hits[0]["lat"]), float(hits[0]["lon"]))
        logger.info("geocode_ok", city=city, lat=result[0], lon=result[1],
                    display=hits[0].get("display_name", "")[:60])
    except Exception as e:
        logger.warning("geocode_failed", city=city, error=str(e)[:200])
        _last_geocode_ts = time.monotonic()
        _geocode_cache[city] = None
        return None

    _geocode_cache[city] = result
    return result


def _fetch_open_weather_events(client: KalshiClient) -> list[dict]:
    """Paginate ALL open events; keep only the weather category."""
    out: list[dict] = []
    cursor: Optional[str] = None
    pages = 0
    while pages < 300:
        resp = client.get_events(status="open", limit=200, cursor=cursor)
        batch = resp.get("events", [])
        out.extend(e for e in batch if e.get("category") == WEATHER_CATEGORY)
        cursor = resp.get("cursor")
        pages += 1
        if not cursor or not batch:
            break
    logger.info("weather_events_fetched", pages=pages, n_weather=len(out))
    return out


def _fetch_brackets(client: KalshiClient, event_ticker: str) -> list[dict]:
    """
    Fetch one event's markets as council-ready bracket dicts, sorted coldest
    → warmest. Brackets whose close_time is past DEADLINE_UTC are dropped.
    """
    resp = client.get_markets(event_ticker=event_ticker, status="open", limit=200)
    markets = resp.get("markets", [])
    cursor = resp.get("cursor")
    while cursor:
        resp = client.get_markets(
            event_ticker=event_ticker, status="open", limit=200, cursor=cursor,
        )
        markets.extend(resp.get("markets", []))
        cursor = resp.get("cursor")

    _TYPE_MAP = {"between": "band", "greater": "above", "less": "below"}

    brackets: list[dict] = []
    for m in markets:
        if m.get("status") not in ("open", "active"):
            continue

        close_time = m.get("close_time")
        if close_time:
            try:
                ct = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
                if ct > DEADLINE_UTC:
                    continue
            except (ValueError, AttributeError):
                pass

        strike_type = m.get("strike_type", "")
        floor_strike = m.get("floor_strike")
        cap_strike = m.get("cap_strike")

        if strike_type == "between":
            threshold = f"{int(floor_strike)}-{int(cap_strike)}°"
            sort_key = float(floor_strike)
        elif strike_type == "greater":
            threshold = f">{int(floor_strike)}°"
            sort_key = float(floor_strike) + 0.5
        elif strike_type == "less":
            threshold = f"<{int(cap_strike)}°"
            sort_key = float(cap_strike) - 99.0  # always coldest
        else:
            logger.warning("unknown_strike_type", ticker=m.get("ticker"),
                           strike_type=strike_type)
            continue

        yes_bid = Decimal(str(m.get("yes_bid_dollars") or "0"))
        yes_ask = Decimal(str(m.get("yes_ask_dollars") or "0"))
        if yes_bid > 0 and yes_ask > 0:
            market_prob = (yes_bid + yes_ask) / Decimal("2")
        else:
            market_prob = yes_ask if yes_ask > 0 else yes_bid
        yes_price = yes_ask if yes_ask > 0 else market_prob
        no_price = (Decimal("1") - yes_bid) if yes_bid > 0 else (Decimal("1") - market_prob)

        brackets.append({
            "ticker": m.get("ticker", ""),
            "title": m.get("title", ""),
            "threshold": threshold,
            "type": _TYPE_MAP[strike_type],
            "floor_strike": floor_strike,
            "cap_strike": cap_strike,
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_bid": yes_bid,
            "market_prob": market_prob,
            "volume": m.get("volume") or 0,
            "close_time": close_time,
            "_sort": sort_key,
        })

    brackets.sort(key=lambda b: b.pop("_sort"))
    return brackets


def discover_weather_events(client: KalshiClient) -> list[WeatherEvent]:
    """
    Discover every open Kalshi weather event, fully resolved: city, coords,
    temp type, and council-ready brackets (deadline-filtered).

    Non-temperature events (rain, hurricanes, drought, …) get temp_type
    parsed as None and are returned WITHOUT brackets/coords — callers filter
    on `temp_type` — so the discovery count still reflects the whole
    category. Temperature events that fail geocoding or have no brackets
    inside the deadline are dropped with a log line.
    """
    raw_events = _fetch_open_weather_events(client)

    out: list[WeatherEvent] = []
    for e in raw_events:
        title = e.get("title", "")
        event_ticker = e.get("event_ticker", "")
        city, temp_type = parse_title(title)

        if temp_type is None:
            # Not a temperature event — visible to callers, but unresolved.
            out.append(WeatherEvent(
                event_ticker=event_ticker,
                series_ticker=e.get("series_ticker", ""),
                title=title, city="", lat=0.0, lon=0.0,
                temp_type="", close_time=None,
                event_date=parse_event_date(event_ticker),
            ))
            continue

        coords = geocode_city(city)
        if coords is None:
            logger.warning("event_dropped_geocode", event=event_ticker, city=city)
            continue

        brackets = _fetch_brackets(client, event_ticker)
        if not brackets:
            logger.info("event_dropped_no_brackets_in_window",
                        event=event_ticker, deadline=DEADLINE_UTC.isoformat())
            continue

        close_times = [b["close_time"] for b in brackets if b.get("close_time")]
        out.append(WeatherEvent(
            event_ticker=event_ticker,
            series_ticker=e.get("series_ticker", ""),
            title=title,
            city=city,
            lat=coords[0],
            lon=coords[1],
            temp_type=temp_type,
            close_time=min(close_times) if close_times else None,
            event_date=parse_event_date(event_ticker),
            brackets=brackets,
        ))

    n_temp = sum(1 for ev in out if ev.temp_type)
    logger.info("weather_discovery_done", total=len(out), temperature=n_temp)
    return out
