"""
FastAPI dashboard server for the Kalshi trading system.

Reads from the same SQLite DB the active trader writes to. All endpoints
are read-only — this process never places orders or mutates state.
"""

import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from sqlalchemy import func

from config.settings import settings
from data.db import CouncilDecision, PaperTrade, Position, get_session, init_db


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Kalshi Trader Dashboard")

_START_TIME = time.time()


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _iso(v: datetime | None) -> str | None:
    if v is None:
        return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.isoformat()


def _uptime_seconds() -> int:
    return int(time.time() - _START_TIME)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/status")
def api_status() -> dict:
    session = get_session()
    try:
        total = session.query(Position).count()
        open_count = session.query(Position).filter(Position.status == "open").count()
        realized = session.query(func.sum(Position.realized_pnl)).filter(
            Position.status.in_(["closed_profit", "closed_loss", "settled"])
        ).scalar() or 0

        # Best-effort live balance — reading the balance hits the Kalshi
        # trade API, which may be slow or unavailable; fall back gracefully.
        balance: float | None = None
        try:
            from core.rest_client import get_trade_client
            resp = get_trade_client().get_balance()
            balance = _num(resp.get("balance"))
        except Exception:
            balance = None

        return {
            "env": settings.kalshi_env,
            "balance": balance,
            "uptime_seconds": _uptime_seconds(),
            "total_positions": total,
            "open_positions": open_count,
            "realized_pnl": _num(realized) or 0.0,
        }
    finally:
        session.close()


@app.get("/api/positions")
def api_positions() -> list[dict]:
    session = get_session()
    try:
        rows = session.query(Position).order_by(Position.entry_time.desc()).all()
        out: list[dict] = []
        for p in rows:
            entry = _num(p.entry_price) or 0.0
            current = _num(p.current_price) or entry
            realized = _num(p.realized_pnl)
            unrealized = _num(p.unrealized_pnl)
            if p.status == "open":
                pnl = unrealized if unrealized is not None else (current - entry) * (p.contracts or 0)
            else:
                pnl = realized if realized is not None else 0.0
            out.append({
                "id": p.id,
                "ticker": p.ticker,
                "strategy": p.strategy,
                "side": p.side,
                "entry_price": entry,
                "current_price": current,
                "contracts": p.contracts,
                "status": p.status,
                "pnl": pnl,
                "entry_time": _iso(p.entry_time),
                "exit_time": _iso(p.exit_time),
                "exit_reason": p.exit_reason,
            })
        return out
    finally:
        session.close()


@app.get("/api/paper_trades")
def api_paper_trades() -> list[dict]:
    session = get_session()
    try:
        rows = (
            session.query(PaperTrade)
            .order_by(PaperTrade.placed_at.desc())
            .limit(200)
            .all()
        )
        return [
            {
                "id": t.id,
                "ticker": t.ticker,
                "strategy": t.strategy,
                "side": t.side,
                "entry_price": _num(t.entry_price),
                "contracts": t.contracts,
                "cost": _num(t.cost),
                "potential_profit": _num(t.potential_profit),
                "signal_edge": _num(t.signal_edge),
                "placed_at": _iso(t.placed_at),
                "settled_at": _iso(t.settled_at),
                "result": t.result,
                "pnl": _num(t.pnl),
            }
            for t in rows
        ]
    finally:
        session.close()


@app.get("/api/council_decisions")
def api_council_decisions() -> list[dict]:
    """Summary row per council run — newest first. Detail is fetched lazily."""
    session = get_session()
    try:
        rows = (
            session.query(CouncilDecision)
            .order_by(CouncilDecision.created_at.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "id": d.id,
                "ticker": d.ticker,
                "market_title": d.market_title,
                "nws_high": d.weather_nws_high,
                "market_yes_price": _num(d.market_yes_price),
                "final_prob": _num(d.stage3_final_prob),
                "confidence": _num(d.stage3_confidence),
                "edge": _num(d.edge),
                "should_trade": bool(d.stage3_should_trade),
                "side": d.stage3_side,
                "created_at": _iso(d.created_at),
                "market_result": d.market_result,
                "was_correct": None if d.was_correct is None else bool(d.was_correct),
            }
            for d in rows
        ]
    finally:
        session.close()


@app.get("/api/council_decision/{decision_id}")
def api_council_decision(decision_id: int) -> dict:
    """Full 3-stage detail for one decision: every model, every reasoning chain."""
    session = get_session()
    try:
        d = session.get(CouncilDecision, decision_id)
        if d is None:
            return {"error": "not found", "id": decision_id}

        models = d.council_models or []

        def model_name(i: int) -> str:
            return models[i] if i < len(models) else f"Model {chr(65 + i)}"

        return {
            "id": d.id,
            "ticker": d.ticker,
            "market_title": d.market_title,
            "nws_high": d.weather_nws_high,
            "market_yes_price": _num(d.market_yes_price),
            "market_no_price": _num(d.market_no_price),
            "edge": _num(d.edge),
            "created_at": _iso(d.created_at),
            "total_cost_usd": _num(d.total_cost_usd),
            "chairman_model": d.chairman_model,
            "council_models": models,
            "market_result": d.market_result,
            "was_correct": None if d.was_correct is None else bool(d.was_correct),
            "stage1": [
                {"label": "Model A", "model": model_name(0), "prob": _num(d.stage1_model_a_prob),
                 "side": d.stage1_model_a_side, "reasoning": d.stage1_model_a_reasoning},
                {"label": "Model B", "model": model_name(1), "prob": _num(d.stage1_model_b_prob),
                 "side": d.stage1_model_b_side, "reasoning": d.stage1_model_b_reasoning},
                {"label": "Model C", "model": model_name(2), "prob": _num(d.stage1_model_c_prob),
                 "side": d.stage1_model_c_side, "reasoning": d.stage1_model_c_reasoning},
            ],
            "stage2": [
                {"label": "Model A", "model": model_name(0),
                 "updated_prob": _num(d.stage2_model_a_updated_prob), "reasoning": d.stage2_model_a_reasoning},
                {"label": "Model B", "model": model_name(1),
                 "updated_prob": _num(d.stage2_model_b_updated_prob), "reasoning": d.stage2_model_b_reasoning},
                {"label": "Model C", "model": model_name(2),
                 "updated_prob": _num(d.stage2_model_c_updated_prob), "reasoning": d.stage2_model_c_reasoning},
            ],
            "stage3": {
                "model": d.chairman_model,
                "final_prob": _num(d.stage3_final_prob),
                "confidence": _num(d.stage3_confidence),
                "should_trade": bool(d.stage3_should_trade),
                "side": d.stage3_side,
                "reasoning": d.stage3_reasoning,
                "dissent_summary": d.stage3_dissent_summary,
                "risk_factors": d.stage3_risk_factors,
            },
        }
    finally:
        session.close()


@app.get("/api/council_accuracy")
def api_council_accuracy() -> dict:
    """How the council is doing: trade/skip split + accuracy on settled markets."""
    session = get_session()
    try:
        rows = session.query(CouncilDecision).all()
        total = len(rows)
        trade = sum(1 for d in rows if d.stage3_should_trade)
        skip = total - trade

        settled = [d for d in rows if d.market_result is not None and d.was_correct is not None]
        correct = sum(1 for d in settled if d.was_correct)
        accuracy = (correct / len(settled)) if settled else None

        win_confs = [_num(d.stage3_confidence) for d in settled
                     if d.was_correct and d.stage3_confidence is not None]
        lose_confs = [_num(d.stage3_confidence) for d in settled
                      if not d.was_correct and d.stage3_confidence is not None]
        avg_conf_win = (sum(win_confs) / len(win_confs)) if win_confs else None
        avg_conf_lose = (sum(lose_confs) / len(lose_confs)) if lose_confs else None

        return {
            "total_decisions": total,
            "trade": trade,
            "skip": skip,
            "settled": len(settled),
            "correct": correct,
            "accuracy": accuracy,
            "avg_confidence_winning": avg_conf_win,
            "avg_confidence_losing": avg_conf_lose,
        }
    finally:
        session.close()


@app.get("/api/stats")
def api_stats() -> dict:
    session = get_session()
    try:
        closed = session.query(Position).filter(
            Position.status.in_(["closed_profit", "closed_loss", "settled"])
        ).all()
        total_trades = len(closed)
        wins = sum(1 for p in closed if (_num(p.realized_pnl) or 0.0) > 0)
        losses = sum(1 for p in closed if (_num(p.realized_pnl) or 0.0) < 0)
        win_rate = (wins / total_trades) if total_trades else 0.0
        pnls = [_num(p.realized_pnl) or 0.0 for p in closed]
        total_pnl = sum(pnls)
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0

        # avg edge from council decisions (the research signal lives there now)
        avg_edge_raw = session.query(func.avg(CouncilDecision.edge)).scalar()
        avg_edge = _num(avg_edge_raw) or 0.0

        # per-strategy breakdown — council research is weather_council only
        strategies: dict[str, dict[str, float]] = {
            "weather_council": {"count": 0, "pnl": 0.0},
        }
        for p in closed:
            key = (p.strategy or "other").lower()
            bucket = strategies.setdefault(key, {"count": 0, "pnl": 0.0})
            bucket["count"] += 1
            bucket["pnl"] += _num(p.realized_pnl) or 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_edge": avg_edge,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "strategies": strategies,
        }
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=False)
