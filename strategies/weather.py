"""
Weather forecasting for the LLM-council research project.

Builds the WEATHER half of the council's context packet for ANY city and
either daily extreme (high or low):

    Open-Meteo ensembles (GFS + ICON)  ──►  per-member daily max/min °F
    NWS official forecast              ──►  the literal settlement source

No market logic lives here anymore — event/bracket discovery is
data/weather_discovery.py and the decision-making is agents/council.py.

Unit footgun (kept from the original implementation): Open-Meteo has at
least once returned Celsius despite `temperature_unit=fahrenheit`, so we
inspect `hourly_units` on every response and convert defensively.
"""

from __future__ import annotations

from datetime import date, datetime
from statistics import mean, pstdev
from typing import Optional

import requests
import structlog

logger = structlog.get_logger()


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
NWS_USER_AGENT = "kalshi-trader/weather-research (contact: dev@localhost)"

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

# NWS /points grid lookups never change for a coordinate — cache them so a
# multi-city scan does one lookup per city per process, not per scan.
_nws_grid_cache: dict[tuple[float, float], Optional[dict]] = {}


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
    timezone_name: str = "auto",
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

    `timezone=auto` makes Open-Meteo resolve the coordinate's local
    timezone, so "daily" maxima/minima are computed on the city's own
    calendar day — which is how NWS observes and Kalshi settles.
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
    # Timestamps are naive ISO strings in the resolved local timezone.
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
    timeout: float = 20.0,
) -> dict[str, dict]:
    """
    Fetch ensemble forecasts from MULTIPLE models and return a dict keyed
    by model name. Models that return fewer than MIN_ENSEMBLE_MEMBERS
    usable members are dropped with a warning — a 1-member "ensemble" is
    just a deterministic run mislabeled, and mixing it into an average
    would pretend we have evidence we don't.

    If ALL models fail, returns an empty dict (caller must handle).
    Individual model failures (network, bad response) are caught and
    logged so one flaky model can't take the whole strategy down.
    """
    out: dict[str, dict] = {}
    for model in models:
        try:
            fc = _fetch_single_ensemble(
                latitude, longitude,
                hours_ahead=hours_ahead, model=model, timeout=timeout,
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


def _nws_grid(latitude: float, longitude: float, timeout: float) -> Optional[dict]:
    """
    Resolve lat/lon → NWS gridpoint metadata ({forecast_url, wfo, grid}).
    Cached per coordinate — the grid mapping is static.
    """
    key = (round(latitude, 4), round(longitude, 4))
    if key in _nws_grid_cache:
        return _nws_grid_cache[key]

    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    try:
        pts_url = NWS_POINTS_URL.format(lat=latitude, lon=longitude)
        r = requests.get(pts_url, headers=headers, timeout=timeout)
        r.raise_for_status()
        props = r.json().get("properties", {})
        forecast_url = props.get("forecast")
        if not forecast_url:
            logger.warning("nws_no_forecast_url", pts_url=pts_url)
            _nws_grid_cache[key] = None
            return None
        grid = {
            "forecast_url": forecast_url,
            "wfo": props.get("gridId") or "",
            "grid": (
                f"{props.get('gridX')},{props.get('gridY')}"
                if props.get("gridX") is not None else ""
            ),
        }
    except Exception as e:
        logger.warning("nws_points_failed", error=str(e))
        # Don't cache transient network failures — only definitive "no grid".
        return None

    _nws_grid_cache[key] = grid
    return grid


def fetch_nws_forecast(
    latitude: float,
    longitude: float,
    target_date: date,
    temp_type: str = "high",
    timeout: float = 15.0,
) -> Optional[dict]:
    """
    Fetch the OFFICIAL NWS forecast high or low for `target_date` at a
    coordinate. NWS is the literal settlement source for Kalshi's
    temperature markets, so when we have it, it carries special weight
    over any raw ensemble prediction.

    temp_type="high": the DAYTIME period starting on target_date carries
    the day's forecast high.
    temp_type="low":  lows happen in the early morning, so the relevant
    period is the NIGHTTIME period that STARTS the evening BEFORE
    target_date (e.g. "Saturday Night" carries Sunday's low). If that
    period has already rolled off the forecast, fall back to the
    nighttime period starting on target_date itself.

    Returns:
        {
            "forecast_temp_f": int,      # forecast high/low in °F (integer)
            "period_name":    str,       # e.g. "Saturday" / "Saturday Night"
            "short_forecast": str,       # e.g. "Partly Sunny"
            "wfo":            str,       # forecast office, e.g. "OKX"
            "grid":           str,       # e.g. "33,37"
            "target_date":    str,       # ISO
        }
    or None if anything along the chain fails (caller treats as "NWS says
    nothing").

    The grid lookup (hop 1 of NWS's two-hop API) is cached per coordinate.
    NWS requires a User-Agent header; requests without one get 403'd.
    """
    grid = _nws_grid(latitude, longitude, timeout)
    if not grid:
        return None

    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    try:
        r = requests.get(grid["forecast_url"], headers=headers, timeout=timeout)
        r.raise_for_status()
        periods = r.json().get("properties", {}).get("periods", [])
    except Exception as e:
        logger.warning("nws_fetch_failed", error=str(e))
        return None

    want_daytime = temp_type == "high"
    if want_daytime:
        # The daytime period (≈06:00–18:00 local) on target_date carries the high.
        wanted_starts = [target_date.isoformat()]
    else:
        from datetime import timedelta
        wanted_starts = [
            (target_date - timedelta(days=1)).isoformat(),  # preferred: night before
            target_date.isoformat(),                        # fallback
        ]

    for start_iso in wanted_starts:
        for p in periods:
            if bool(p.get("isDaytime")) != want_daytime:
                continue
            if not (p.get("startTime", "")).startswith(start_iso):
                continue
            temp = p.get("temperature")
            if temp is None:
                continue
            unit = (p.get("temperatureUnit") or "F").upper()
            temp_f = float(temp) if unit == "F" else _c_to_f(float(temp))
            result = {
                "forecast_temp_f": round(temp_f),
                "period_name": p.get("name", ""),
                "short_forecast": p.get("shortForecast", ""),
                "wfo": grid["wfo"],
                "grid": grid["grid"],
                "target_date": target_date.isoformat(),
            }
            logger.info("nws_forecast_ok", temp_type=temp_type, **result)
            return result

    logger.warning(
        "nws_no_period_for_date",
        target_date=target_date.isoformat(),
        temp_type=temp_type,
        n_periods=len(periods),
    )
    return None


# ----------------------------------------------------------------------
# Ensemble → daily extreme distribution
# ----------------------------------------------------------------------

def _daily_extreme_per_member(
    forecast: dict, target_date: date, temp_type: str,
) -> list[float]:
    """
    Collapse the hourly ensemble forecast to one daily max ("high") or min
    ("low") per member for `target_date`. Returns a list (one value per
    member) of °F daily extremes. Members with no hourly data for that
    date are skipped.
    """
    times: list[datetime] = forecast["times"]
    members: dict[str, list[float]] = forecast["members"]

    # Indices for hours that fall on the target date (local time).
    idxs = [i for i, t in enumerate(times) if t.date() == target_date]
    if not idxs:
        return []

    pick = max if temp_type == "high" else min
    out: list[float] = []
    for member_id, series in members.items():
        vals = [series[i] for i in idxs if series[i] is not None]
        if vals:
            out.append(pick(vals))
    return out


def get_weather_context(
    latitude: float,
    longitude: float,
    target_date: date,
    temp_type: str = "high",
    city: str = "",
) -> dict:
    """
    Package one city/day's forecast data into a clean dict for the LLM
    council (agents/council.py). This is the WEATHER half of the council's
    context packet; the caller pairs it with the event's bracket data.

    Returns:
        {
            "city":              "Chicago",
            "temp_type":         "high" | "low",
            "target_date":       "2026-06-07",
            "gfs_forecast":      {mean, min, max, stdev, n_members, members[°F]} | None,
            "icon_forecast":     {...} | None,
            "nws_temp":          int | None,     # official NWS forecast high/low
            "nws_short_forecast": str | None,    # e.g. "Partly Sunny"
            "ensemble_mean":     float | None,   # across ALL members of ALL models
            "ensemble_spread":   float,          # population stdev across all members
            "all_member_temps":  [float, ...],   # every member's daily extreme °F
            "n_members":         int,            # total members across all models
            "n_models":          int,            # count of valid ensembles
        }

    Descriptive statistics + raw members, not per-market probabilities —
    the council reasons over the distribution itself.
    """
    if temp_type not in ("high", "low"):
        raise ValueError(f"temp_type must be 'high' or 'low', got {temp_type!r}")

    ensembles = fetch_weather_forecast(latitude, longitude, hours_ahead=48)

    # Collapse each ensemble to per-member daily extremes for the target date.
    per_model: dict[str, list[float]] = {}
    for model, fc in ensembles.items():
        extremes = _daily_extreme_per_member(fc, target_date, temp_type)
        if extremes:
            per_model[model] = extremes

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

    nws = fetch_nws_forecast(latitude, longitude, target_date=target_date,
                             temp_type=temp_type)

    return {
        "city": city,
        "temp_type": temp_type,
        "target_date": target_date.isoformat(),
        "gfs_forecast": _stats(per_model.get("gfs_seamless", [])),
        "icon_forecast": _stats(per_model.get("icon_seamless", [])),
        "nws_temp": nws["forecast_temp_f"] if nws else None,
        "nws_short_forecast": nws["short_forecast"] if nws else None,
        "ensemble_mean": ensemble_mean,
        "ensemble_spread": ensemble_spread,
        "all_member_temps": [round(v, 1) for v in all_temps],
        "n_members": len(all_temps),
        "n_models": len(per_model),
    }
