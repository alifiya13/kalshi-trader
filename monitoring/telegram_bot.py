"""
Telegram alerts for the trading system.

All send functions fail silently (log a warning) if credentials are missing
or the HTTP call errors — the trader must never crash on a broken notifier.
"""

import logging
from decimal import Decimal
from typing import Optional

import requests

from config.settings import settings


logger = logging.getLogger(__name__)

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT = 5  # seconds


def send_alert(message: str) -> bool:
    """Send a Telegram message via the Bot API. Returns True on success."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        logger.warning("Telegram alert skipped: bot token or chat id not configured")
        return False

    try:
        resp = requests.post(
            _API_URL.format(token=token),
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as e:
        logger.warning("Telegram send exception: %s", e)
        return False


def _fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def alert_startup(env: str, balance) -> bool:
    msg = (
        f"<b>🚀 Trader started</b>\n"
        f"Env: <code>{env}</code>\n"
        f"Balance: <b>{_fmt_money(balance)}</b>"
    )
    return send_alert(msg)


def alert_new_position(
    ticker: str,
    side: str,
    contracts: int,
    entry_price,
    cost,
    strategy: str,
    edge,
) -> bool:
    try:
        edge_pct = f"{float(edge) * 100:+.1f}%"
    except Exception:
        edge_pct = str(edge)
    msg = (
        f"<b>🟢 NEW POSITION</b>\n"
        f"<code>{ticker}</code>\n"
        f"Side: <b>{side.upper()}</b>  x{contracts}\n"
        f"Entry: {_fmt_money(entry_price)}\n"
        f"Cost: <b>{_fmt_money(cost)}</b>\n"
        f"Strategy: <i>{strategy}</i>\n"
        f"Edge: {edge_pct}"
    )
    return send_alert(msg)


def alert_exit(ticker: str, side: str, exit_price, pnl, reason: str) -> bool:
    try:
        pnl_f = float(pnl)
    except Exception:
        pnl_f = 0.0
    icon = "✅" if pnl_f >= 0 else "🔴"
    msg = (
        f"<b>{icon} EXIT</b>\n"
        f"<code>{ticker}</code>\n"
        f"Side: <b>{side.upper()}</b>\n"
        f"Exit: {_fmt_money(exit_price)}\n"
        f"P&amp;L: <b>{'+' if pnl_f >= 0 else ''}{_fmt_money(pnl_f)}</b>\n"
        f"Reason: <i>{reason}</i>"
    )
    return send_alert(msg)


def alert_scan_summary(
    num_markets: int,
    num_signals: int,
    num_trades: int,
    balance,
) -> bool:
    msg = (
        f"<b>📊 Scan summary</b>\n"
        f"Markets: {num_markets}\n"
        f"Signals: {num_signals}\n"
        f"Trades: <b>{num_trades}</b>\n"
        f"Balance: {_fmt_money(balance)}"
    )
    return send_alert(msg)


def alert_error(error_message: str) -> bool:
    # Truncate to keep under Telegram's 4096-char limit even with formatting.
    trimmed = error_message if len(error_message) <= 3500 else error_message[:3500] + "…"
    msg = f"<b>⚠️ TRADER ERROR</b>\n<pre>{trimmed}</pre>"
    return send_alert(msg)


if __name__ == "__main__":
    # Manual smoke test — `python -m monitoring.telegram_bot`
    logging.basicConfig(level=logging.INFO)
    ok = send_alert(
        "<b>🧪 Telegram test</b>\n"
        "If you see this, the Kalshi trader alerts channel is wired up."
    )
    print("sent=" + str(ok))
