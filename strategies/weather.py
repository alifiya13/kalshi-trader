"""
Strategy 1 — Weather market trading (KXHIGHNY and friends)

Thesis
------
NWS-settled daily high-temperature brackets on Kalshi are a rare clean
signal: the settlement source is public, deterministic, and forecastable
by ensemble numerical weather models. Kalshi's books on these markets
quote 1¢ spreads with real volume, so execution risk is low. Our edge
comes from a better probability estimate than whatever the market
implies after fee/vig, driven by a free public ensemble forecast.

Pipeline
--------
    Open-Meteo ensemble forecast  ──►  P(daily_max in bracket)
    Kalshi KXHIGHNY markets       ──►  implied P from YES price
                                          │
                                          ▼
                                      edge = model_p - market_p
                                      Kelly sizer → contracts

Ticker format (discovered empirically, 2026-04)
-----------------------------------------------
The /markets list response carries three first-class fields we key on
instead of parsing ticker suffixes:

    strike_type   "between"  → bracket  (floor_strike..cap_strike, integer °F, inclusive)
    strike_type   "greater"  → upper tail, strictly > floor_strike
    strike_type   "less"     → lower tail, strictly < cap_strike

Example tickers (title shown for clarity):
    KXHIGHNY-26APR11-B63.5   between  floor=63 cap=64   "63-64°"
    KXHIGHNY-26APR11-T68     greater  floor=68          ">68°"
    KXHIGHNY-26APR11-T61     less     cap=61            "<61°"

NWS daily-high observations are whole integer °F, so bracket "63-64"
settles YES iff observed_int ∈ {63, 64}. Under standard rounding of a
continuous forecast this is equivalent to 62.5 ≤ raw < 64.5. The
greater/less tails map the same way:

    ">68°"  ≡  observed_int ≥ 69   ≡  raw ≥ 68.5
    "<61°"  ≡  observed_int ≤ 60   ≡  raw < 60.5

We use this half-degree offset when scoring ensemble members.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from statistics import mean, pstdev
from typing import Optional

import requests
import structlog

from core.rest_client import KalshiClient

logger = structlog.get_logger()


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Central Park, NYC — NWS siting for daily high at KNYC
NYC_LAT = 40.7831
NYC_LON = -73.9712

OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
NWS_USER_AGENT = "kalshi-trader/weather-v1 (contact: dev@localhost)"

# Ensemble systems we query, in priority order. GFS (NOAA GEFS ~30 members)
# is primary because Kalshi settles on NWS, which is a US dataset. ICON
# (DWD ~40 members) is our cross-check — distinct physics, independent
# training. ECMWF IFS is often the best in the world but Open-Meteo's free
# tier has flaky ensemble support for it; kept last and skipped if thin.
ENSEMBLE_MODELS: tuple[str, ...] = (
    "gfs_seamless",
    "icon_seamless",
    "ecmwf_ifs04",
)

# A model must return at least this many ensemble members or we skip it.
# One-member "ensembles" are just the deterministic run, which defeats
# the entire point of asking for a probability distribution.
MIN_ENSEMBLE_MEMBERS = 10

# If the ensemble consensus and the NWS official forecast disagree by more
# than this many °F, we veto all trades on the day. Big disagreement means
# one of the two inputs is wrong and we don't know which.
NWS_VETO_THRESHOLD_F = 3.0

# If the NWS forecast high is within this many °F of a market's strike
# boundary, we boost that market's confidence — NWS nailing the threshold
# is exactly the situation where its special weight matters most.
NWS_BOOST_DISTANCE_F = 1.0


# ----------------------------------------------------------------------
# Forecast fetching
# ----------------------------------------------------------------------

def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _fetch_single_ensemble(
    latitude: float,
    longitude: float,
    hours_ahead: int = 48,
    model: str = "gfs_seamless",
    timezone_name: str = "America/New_York",
    timeout: float = 20.0,
) -> dict:
    """
    Fetch one ensemble 2m temperature forecast from Open-Meteo.

    Returns a dict with:
        {
            "times":    [datetime, ...]    length N (hourly, local time)
            "members":  { "01": [float, ...], "02": [...], ... }
                        each list is length N, in °F
            "units":    "°F"
            "model":    str
            "n_members": int
        }

    We request `temperature_unit=fahrenheit` so Open-Meteo does the conversion
    server-side. Because the public Open-Meteo API has at least once returned
    Celsius despite the fahrenheit flag (documented footgun), we inspect
    `hourly_units` on the response and convert defensively if it comes back
    as °C. That keeps downstream code always in °F.
    """
    # Open-Meteo's ensemble endpoint bills in forecast_days, not hours.
    # Round up so we cover at least `hours_ahead` of forecast horizon.
    forecast_days = max(2, (hours_ahead + 23) // 24)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "models": model,
        "temperature_unit": "fahrenheit",
        "forecast_days": forecast_days,
        "timezone": timezone_name,
    }

    logger.info(
        "weather_forecast_request",
        lat=latitude, lon=longitude, model=model, forecast_days=forecast_days,
    )
    resp = requests.get(OPEN_METEO_ENSEMBLE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    units = data.get("hourly_units", {})

    raw_times = hourly.get("time", [])
    # Timestamps are naive ISO strings in the requested timezone.
    times = [datetime.fromisoformat(t) for t in raw_times]

    # --- Defensive unit handling ---
    # If ANY temperature column came back as °C, flip a conversion flag.
    # (Open-Meteo either honors the param for every column or none.)
    needs_c_to_f = False
    for key, unit in units.items():
        if key.startswith("temperature_2m") and unit and "C" in unit.upper():
            needs_c_to_f = True
            break

    # Collect every ensemble member column. The deterministic base run is
    # published as plain "temperature_2m"; numbered members are
    # "temperature_2m_memberNN". We treat the deterministic run as
    # member "00" so callers get one combined distribution.
    members: dict[str, list[float]] = {}
    for key, series in hourly.items():
        if not key.startswith("temperature_2m"):
            continue
        if key == "temperature_2m":
            member_id = "00"
        else:
            # "temperature_2m_member07" → "07"
            member_id = key.rsplit("_member", 1)[-1]

        if needs_c_to_f:
            members[member_id] = [
                _c_to_f(float(v)) if v is not None else None for v in series
            ]
        else:
            members[member_id] = [
                float(v) if v is not None else None for v in series
            ]

    logger.info(
        "weather_forecast_ok",
        n_times=len(times),
        n_members=len(members),
        unit_in=units.get("temperature_2m"),
        converted_c_to_f=needs_c_to_f,
    )

    return {
        "times": times,
        "members": members,
        "units": "°F",
        "model": model,
        "n_members": len(members),
    }


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    hours_ahead: int = 48,
    models: tuple[str, ...] = ENSEMBLE_MODELS,
    timezone_name: str = "America/New_York",
    timeout: float = 20.0,
) -> dict[str, dict]:
    """
    Fetch ensemble forecasts from MULTIPLE models and return a dict keyed
    by model name. Models that return fewer than MIN_ENSEMBLE_MEMBERS
    usable members are dropped with a warning — a 1-member "ensemble" is
    just a deterministic run mislabeled, and mixing it into an average
    would pretend we have evidence we don't.

    Returns:
        { "gfs_seamless": {...forecast dict...},
          "icon_seamless": {...forecast dict...} }

    If ALL models fail, returns an empty dict (caller must handle).
    Individual model failures (network, bad response) are caught and
    logged so one flaky model can't take the whole strategy down.
    """
    out: dict[str, dict] = {}
    for model in models:
        try:
            fc = _fetch_single_ensemble(
                latitude, longitude,
                hours_ahead=hours_ahead, model=model,
                timezone_name=timezone_name, timeout=timeout,
            )
        except Exception as e:
            logger.warning("ensemble_fetch_failed", model=model, error=str(e))
            continue

        if fc["n_members"] < MIN_ENSEMBLE_MEMBERS:
            logger.warning(
                "ensemble_too_thin",
                model=model,
                n_members=fc["n_members"],
                min_required=MIN_ENSEMBLE_MEMBERS,
            )
            continue

        out[model] = fc

    logger.info(
        "ensembles_fetched",
        n_valid=len(out),
        models=list(out.keys()),
    )
    return out


def fetch_nws_forecast(
    latitude: float,
    longitude: float,
    target_date: Optional[date] = None,
    timeout: float = 15.0,
) -> Optional[dict]:
    """
    Fetch the OFFICIAL NWS forecast high for `target_date` (default: tomorrow
    in NYC local time). This is the literal settlement source for Kalshi's
    KXHIGHNY markets, so when we have it, it carries special weight over
    any raw ensemble prediction.

    Returns a dict:
        {
            "forecast_high_f": int,      # tomorrow's high in °F (integer)
            "period_name":    str,       # e.g. "Saturday"
            "short_forecast": str,       # e.g. "Partly Sunny"
            "wfo":            str,       # forecast office, e.g. "OKX"
            "grid":           str,       # e.g. "33,37"
            "target_date":    str,       # ISO
        }
    or None if anything along the chain fails (no NWS high → caller treats
    as "NWS says nothing, don't apply the veto").

    The NWS API is a two-hop flow:
        1. GET /points/{lat},{lon}  → returns the forecast office + gridpoint URL
        2. GET that forecast URL    → returns a list of daytime/nighttime periods

    NWS requires a User-Agent header; requests without one get 403'd.
    """
    if target_date is None:
        now_nyc = datetime.now(timezone(timedelta(hours=-4)))
        target_date = (now_nyc + timedelta(days=1)).date()

    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}

    try:
        # --- Hop 1: resolve lat/lon to a gridpoint forecast URL ---
        pts_url = NWS_POINTS_URL.format(lat=latitude, lon=longitude)
        r = requests.get(pts_url, headers=headers, timeout=timeout)
        r.raise_for_status()
        props = r.json().get("properties", {})
        forecast_url = props.get("forecast")
        wfo = props.get("gridId")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        if not forecast_url:
            logger.warning("nws_no_forecast_url", pts_url=pts_url)
            return None

        # --- Hop 2: fetch the period list ---
        r2 = requests.get(forecast_url, headers=headers, timeout=timeout)
        r2.raise_for_status()
        periods = r2.json().get("properties", {}).get("periods", [])
    except Exception as e:
        logger.warning("nws_fetch_failed", error=str(e))
        return None

    # Find the DAYTIME period whose startTime lands on target_date.
    # NWS daytime periods cover roughly 06:00–18:00 local and carry
    # the high; nighttime periods carry the low. `isDaytime=True` is
    # the authoritative flag.
    target_iso = target_date.isoformat()
    for p in periods:
        if not p.get("isDaytime"):
            continue
        start = p.get("startTime", "")
        if not start.startswith(target_iso):
            continue
        temp = p.get("temperature")
        unit = (p.get("temperatureUnit") or "F").upper()
        if temp is None:
            continue
        temp_f = float(temp) if unit == "F" else _c_to_f(float(temp))
        result = {
            "forecast_high_f": round(temp_f),
            "period_name": p.get("name", ""),
            "short_forecast": p.get("shortForecast", ""),
            "wfo": wfo or "",
            "grid": f"{grid_x},{grid_y}" if grid_x is not None else "",
            "target_date": target_iso,
        }
        logger.info("nws_forecast_ok", **result)
        return result

    logger.warning(
        "nws_no_period_for_date",
        target_date=target_iso,
        n_periods=len(periods),
    )
    return None


# ----------------------------------------------------------------------
# Ensemble → probability
# ----------------------------------------------------------------------

def _daily_max_per_member(
    forecast: dict, target_date
) -> list[float]:
    """
    Collapse the hourly ensemble forecast to one daily max per member for
    `target_date`. Returns a list (one value per member) of °F daily highs.
    Members with no hourly data for that date are skipped.
    """
    times: list[datetime] = forecast["times"]
    members: dict[str, list[float]] = forecast["members"]

    # Indices for hours that fall on the target date (local time).
    idxs = [i for i, t in enumerate(times) if t.date() == target_date]
    if not idxs:
        return []

    out: list[float] = []
    for member_id, series in members.items():
        vals = [series[i] for i in idxs if series[i] is not None]
        if vals:
            out.append(max(vals))
    return out


def compute_exceedance_probability(
    ensemble_temps: list[float], threshold_f: float
) -> float:
    """
    P(daily_max > threshold) = (# members with max > threshold) / total.

    `ensemble_temps` is a flat list of per-member daily highs already
    collapsed for the day of interest. Threshold is strict "greater than"
    to match the semantics of Kalshi's ">N°" tickers; see the rounding
    math in the module docstring for how bracket markets use this.
    """
    if not ensemble_temps:
        return 0.0
    hits = sum(1 for t in ensemble_temps if t > threshold_f)
    return hits / len(ensemble_temps)


def _bracket_probability(
    ensemble_temps: list[float], lo_raw: float, hi_raw: float
) -> float:
    """P(lo_raw <= daily_max < hi_raw) across ensemble members."""
    if not ensemble_temps:
        return 0.0
    hits = sum(1 for t in ensemble_temps if lo_raw <= t < hi_raw)
    return hits / len(ensemble_temps)


# ----------------------------------------------------------------------
# Market parsing + edge finder
# ----------------------------------------------------------------------

@dataclass
class WeatherSignal:
    ticker: str
    title: str
    strike_type: str          # "between" | "greater" | "less"
    floor_strike: Optional[int]
    cap_strike: Optional[int]
    threshold_label: str      # human-readable range, e.g. "63-64°" or ">68°"
    model_prob: Decimal       # AVERAGE probability across valid ensembles
    market_yes_bid: Decimal
    market_yes_ask: Decimal
    market_prob: Decimal      # midpoint of bid/ask as implied prob
    edge: Decimal             # signed: model_p - cost on the chosen side
    side: str                 # "yes" or "no"
    confidence: str           # "high" | "medium" | "low"
    tradeable: bool           # passes edge ≥ 5¢ AND confidence != "low"
    # Per-model diagnostics — keeping these makes the signal auditable
    # after the fact when a trade goes wrong.
    per_model_probs: dict     # { "gfs_seamless": 0.64, "icon_seamless": 0.02 }
    ensemble_mean_f: Optional[float]  # mean daily max across all members
    nws_forecast_high: Optional[int]  # integer °F from NWS, None if unavailable
    n_members: int            # TOTAL members across all valid ensembles
    n_models: int             # count of valid ensembles
    veto_reason: Optional[str]  # why tradeable=False (if applicable)


def _market_midpoint(yes_bid: Decimal, yes_ask: Decimal) -> Decimal:
    """Implied prob from the YES side — midpoint of bid/ask."""
    if yes_bid <= 0 and yes_ask <= 0:
        return Decimal("0")
    if yes_bid <= 0:
        return yes_ask
    if yes_ask <= 0:
        return yes_bid
    return (yes_bid + yes_ask) / Decimal("2")


def _score_market(market: dict, daily_maxes: list[float]) -> tuple[Decimal, str]:
    """
    Compute model probability + human-readable threshold label for one market.
    See module docstring for the half-degree rounding math — NWS integers
    round to [lo-0.5, hi+0.5) in continuous raw-forecast terms.
    """
    strike_type = market.get("strike_type")
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    if strike_type == "between":
        lo = float(floor_strike)
        hi = float(cap_strike)
        # integer observations {lo, lo+1, ..., hi}  ⇔  raw in [lo-0.5, hi+0.5)
        p = _bracket_probability(daily_maxes, lo - 0.5, hi + 0.5)
        label = f"{int(lo)}-{int(hi)}°"
    elif strike_type == "greater":
        n = float(floor_strike)
        # ">N°" integer observations ≥ N+1  ⇔  raw ≥ N+0.5
        p = compute_exceedance_probability(daily_maxes, n + 0.5 - 1e-9)
        label = f">{int(n)}°"
    elif strike_type == "less":
        n = float(cap_strike)
        # "<N°" integer observations ≤ N-1  ⇔  raw < N-0.5
        p = 1.0 - compute_exceedance_probability(daily_maxes, n - 0.5 - 1e-9)
        label = f"<{int(n)}°"
    else:
        p = 0.0
        label = "?"

    return Decimal(str(round(p, 4))), label


def _distance_to_strike(market: dict, temp_f: float) -> float:
    """
    Distance in °F from `temp_f` to the nearest edge of this market's
    strike range. Smaller = NWS is closer to the decision boundary for
    this bucket, which is either the best case (NWS lands squarely inside)
    or the worst case (NWS is right at the edge and could go either way).
    We use it for confidence boosting — see _apply_nws_confidence.
    """
    strike_type = market.get("strike_type")
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    if strike_type == "between":
        lo = float(floor_strike)
        hi = float(cap_strike)
        if lo <= temp_f <= hi:
            return 0.0  # NWS lands inside the bucket
        return min(abs(temp_f - lo), abs(temp_f - hi))
    if strike_type == "greater":
        n = float(floor_strike)
        return abs(temp_f - n)
    if strike_type == "less":
        n = float(cap_strike)
        return abs(temp_f - n)
    return float("inf")


def _nws_agrees_with_side(
    market: dict, nws_high: float, side: str
) -> bool:
    """
    Does the NWS forecast high actually support the side we want to take?

    - For a "between" bucket: NWS inside the range → YES should win;
      NWS outside → NO should win.
    - For a "greater" (>N) market: NWS > N → YES; else NO.
    - For a "less" (<N) market: NWS < N → YES; else NO.

    Returns True if NWS agrees with our chosen side.
    """
    strike_type = market.get("strike_type")
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    if strike_type == "between":
        lo = float(floor_strike)
        hi = float(cap_strike)
        nws_says_yes = lo <= nws_high <= hi
    elif strike_type == "greater":
        nws_says_yes = nws_high > float(floor_strike)
    elif strike_type == "less":
        nws_says_yes = nws_high < float(cap_strike)
    else:
        return False

    return (side == "yes" and nws_says_yes) or (side == "no" and not nws_says_yes)


def _base_confidence(
    n_members: int, n_models: int, model_prob: Decimal
) -> str:
    """
    Pre-NWS confidence based purely on ensemble signal strength:
    - high:   ≥2 models AND model prob far from 50% AND ≥20 members total
    - low:    prob in [0.4, 0.6] — coin-flip zone is where we're most
              exposed to ensemble noise
    - medium: everything else
    """
    sharpness = abs(float(model_prob) - 0.5) * 2  # 0..1
    if sharpness > 0.7 and n_members >= 20 and n_models >= 2:
        return "high"
    if sharpness < 0.2:
        return "low"
    return "medium"


# "KXHIGHNY-26APR11-B63.5"  →  date(2026, 4, 11)
_TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})-")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_ticker_date(ticker: str) -> Optional[date]:
    """
    Extract the observation date from a weather ticker like
    `KXHIGHNY-26APR11-B63.5`. Returns None on any parse failure.

    The ticker date is the measurement day itself — *not* the close_time,
    which lands the following calendar day in UTC because NYC daily-high
    observations are finalized at midnight local (04:00 UTC). Using the
    ticker date avoids the off-by-one that close_time would cause.
    """
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


def get_weather_context(
    target_date: Optional[date] = None,
    latitude: float = NYC_LAT,
    longitude: float = NYC_LON,
) -> dict:
    """
    Package the day's weather forecast data into a clean dict for the LLM
    council (agents/council.py). This is the WEATHER half of the council's
    context packet; the caller pairs it with live Kalshi market data.

    Returns:
        {
            "target_date":       "2026-06-06",
            "gfs_forecast":      {mean, min, max, stdev, n_members, members[°F]} | None,
            "icon_forecast":     {...} | None,
            "nws_high":          int | None,     # official NWS forecast high
            "nws_short_forecast": str | None,    # e.g. "Partly Sunny"
            "ensemble_mean":     float | None,   # across ALL members of ALL models
            "ensemble_spread":   float,          # population stdev across all members
            "all_member_temps":  [float, ...],   # every member's daily-high °F
            "n_members":         int,            # total members across all models
            "n_models":          int,            # count of valid ensembles
        }

    Mirrors the data find_weather_edge() consumes (GFS/ICON ensembles + NWS),
    but returns descriptive statistics + raw members instead of per-market
    probabilities — the council reasons over the distribution itself.
    """
    if target_date is None:
        now_nyc = datetime.now(timezone(timedelta(hours=-4)))
        target_date = (now_nyc + timedelta(days=1)).date()

    ensembles = fetch_weather_forecast(latitude, longitude, hours_ahead=48)

    # Collapse each ensemble to per-member daily highs for the target date.
    per_model: dict[str, list[float]] = {}
    for model, fc in ensembles.items():
        maxes = _daily_max_per_member(fc, target_date)
        if maxes:
            per_model[model] = maxes

    def _stats(vals: list[float]) -> Optional[dict]:
        if not vals:
            return None
        return {
            "mean": round(mean(vals), 2),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "stdev": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "n_members": len(vals),
            "members": [round(v, 1) for v in vals],
        }

    all_temps = [t for lst in per_model.values() for t in lst]
    ensemble_mean = round(mean(all_temps), 2) if all_temps else None
    ensemble_spread = round(pstdev(all_temps), 2) if len(all_temps) > 1 else 0.0

    nws = fetch_nws_forecast(latitude, longitude, target_date=target_date)

    return {
        "target_date": target_date.isoformat(),
        "gfs_forecast": _stats(per_model.get("gfs_seamless", [])),
        "icon_forecast": _stats(per_model.get("icon_seamless", [])),
        "nws_high": nws["forecast_high_f"] if nws else None,
        "nws_short_forecast": nws["short_forecast"] if nws else None,
        "ensemble_mean": ensemble_mean,
        "ensemble_spread": ensemble_spread,
        "all_member_temps": [round(v, 1) for v in all_temps],
        "n_members": len(all_temps),
        "n_models": len(per_model),
    }


def find_weather_edge(
    client: KalshiClient,
    series_ticker: str = "KXHIGHNY",
    latitude: float = NYC_LAT,
    longitude: float = NYC_LON,
    target_date: Optional[date] = None,
    min_edge: Decimal = Decimal("0.05"),
) -> list[WeatherSignal]:
    """
    Full multi-source pipeline:

        1. Fetch KXHIGHNY markets, filter to target_date (tomorrow by default)
        2. Fetch N ensemble forecasts (GFS, ICON, ECMWF if available)
        3. Fetch NWS official forecast high (the settlement source)
        4. For each market:
             - Compute probability under EACH ensemble
             - Average to get model_prob
             - Compare NWS high vs ensemble mean
             - Apply NWS confidence boost / veto
        5. Mark signal.tradeable iff edge >= min_edge AND confidence != "low"

    Returns WeatherSignal list sorted by (signed) edge descending.
    """
    # --- 1. Pull all markets in the series, any status ---
    resp = client.get_markets(series_ticker=series_ticker, status=None, limit=200)
    all_markets = resp.get("markets", [])
    active = [m for m in all_markets if m.get("status") == "active"]

    # --- 2. Pick target date ---
    if target_date is None:
        now_nyc = datetime.now(timezone(timedelta(hours=-4)))
        target_date = (now_nyc + timedelta(days=1)).date()

    # --- 3. Filter active markets to the target date by TICKER DATE ---
    target_markets: list[dict] = [
        m for m in active
        if parse_ticker_date(m.get("ticker", "")) == target_date
    ]

    logger.info(
        "weather_markets_filtered",
        series=series_ticker,
        total=len(all_markets),
        active=len(active),
        target_date=str(target_date),
        target_count=len(target_markets),
    )

    if not target_markets:
        return []

    # --- 4. Fetch all available ensembles ---
    ensembles = fetch_weather_forecast(latitude, longitude, hours_ahead=48)
    if not ensembles:
        logger.warning("no_ensembles_available")
        return []

    # Collapse each ensemble to per-member daily maxes for the target date.
    # Skip any that return nothing for this date (ensemble horizon too short).
    per_model_maxes: dict[str, list[float]] = {}
    for model, fc in ensembles.items():
        maxes = _daily_max_per_member(fc, target_date)
        if maxes:
            per_model_maxes[model] = maxes

    if not per_model_maxes:
        logger.warning(
            "no_ensemble_has_target_date",
            target_date=str(target_date),
        )
        return []

    n_members_total = sum(len(m) for m in per_model_maxes.values())
    n_models = len(per_model_maxes)

    # Ensemble mean daily high across all members of all models — a single
    # point estimate we can compare to the NWS number.
    all_maxes = [t for lst in per_model_maxes.values() for t in lst]
    ensemble_mean_f = sum(all_maxes) / len(all_maxes) if all_maxes else None

    # --- 5. Fetch NWS official forecast ---
    nws = fetch_nws_forecast(latitude, longitude, target_date=target_date)
    nws_high: Optional[int] = nws["forecast_high_f"] if nws else None

    # Day-level veto: if NWS and the ensemble mean disagree by more than
    # NWS_VETO_THRESHOLD_F, we don't know which to trust and we refuse to
    # trade anything for the day. This is exactly the case that blew up
    # on our first run: GFS said ~60°F, NWS said 63°F, we had no way to
    # know GFS was the cold outlier until we ran the second model.
    day_veto: Optional[str] = None
    if nws_high is not None and ensemble_mean_f is not None:
        disagreement = abs(float(nws_high) - ensemble_mean_f)
        if disagreement > NWS_VETO_THRESHOLD_F:
            day_veto = (
                f"NWS ({nws_high}°F) vs ensemble mean "
                f"({ensemble_mean_f:.1f}°F) disagree by "
                f"{disagreement:.1f}°F (> {NWS_VETO_THRESHOLD_F}°F)"
            )
            logger.warning("day_level_veto", reason=day_veto)

    # --- 6. Score every market ---
    signals: list[WeatherSignal] = []
    for m in target_markets:
        # Per-model probability, then average.
        per_model: dict[str, float] = {}
        label = ""
        for model, maxes in per_model_maxes.items():
            p, label = _score_market(m, maxes)
            per_model[model] = float(p)

        avg_prob = sum(per_model.values()) / len(per_model)
        model_prob = Decimal(str(round(avg_prob, 4)))

        # Market prices & costs
        yes_bid = Decimal(str(m.get("yes_bid_dollars") or "0"))
        yes_ask = Decimal(str(m.get("yes_ask_dollars") or "0"))
        market_prob = _market_midpoint(yes_bid, yes_ask)

        yes_cost = yes_ask if yes_ask > 0 else market_prob
        no_cost = Decimal("1") - yes_bid if yes_bid > 0 else Decimal("1") - market_prob

        yes_edge = model_prob - yes_cost
        no_edge = (Decimal("1") - model_prob) - no_cost

        if yes_edge >= no_edge:
            side = "yes"
            edge = yes_edge
        else:
            side = "no"
            edge = no_edge

        # --- Confidence logic ---
        confidence = _base_confidence(n_members_total, n_models, model_prob)
        veto_reason: Optional[str] = None

        if day_veto:
            # Day-level veto overrides everything.
            confidence = "low"
            veto_reason = day_veto
        elif nws_high is not None:
            # Boost or cut based on how NWS treats this specific market.
            dist = _distance_to_strike(m, float(nws_high))
            nws_agrees = _nws_agrees_with_side(m, float(nws_high), side)

            if not nws_agrees:
                # NWS — the literal settlement source — disagrees with the
                # side we'd take. We don't care how far it is from the
                # boundary; if NWS is calling for 63°F and we'd sell NO on
                # the 63-64° bucket, we'd be fighting the thing that
                # settles the market. Hard veto.
                confidence = "low"
                veto_reason = (
                    f"NWS high {nws_high}°F contradicts {side.upper()} "
                    f"side on {label}"
                )
            elif dist <= NWS_BOOST_DISTANCE_F:
                # NWS agrees with us AND is inside/adjacent to the bucket
                # — this is our strongest confirmation.
                confidence = "high"
            # NWS agrees but is far from the boundary → keep base confidence.

        # --- Final tradeability gate ---
        tradeable = (edge >= min_edge) and (confidence != "low")

        signals.append(WeatherSignal(
            ticker=m.get("ticker", ""),
            title=m.get("title", ""),
            strike_type=m.get("strike_type", ""),
            floor_strike=m.get("floor_strike"),
            cap_strike=m.get("cap_strike"),
            threshold_label=label,
            model_prob=model_prob,
            market_yes_bid=yes_bid,
            market_yes_ask=yes_ask,
            market_prob=market_prob,
            edge=edge,
            side=side,
            confidence=confidence,
            tradeable=tradeable,
            per_model_probs={k: round(v, 4) for k, v in per_model.items()},
            ensemble_mean_f=round(ensemble_mean_f, 2) if ensemble_mean_f else None,
            nws_forecast_high=nws_high,
            n_members=n_members_total,
            n_models=n_models,
            veto_reason=veto_reason,
        ))

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals
