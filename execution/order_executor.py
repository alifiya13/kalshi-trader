"""
Order Executor — handles order placement on Kalshi demo.

All orders go to the demo environment. Every order is logged to the
database for audit trail.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from core.rest_client import KalshiClient
from data.db import Order, get_session

logger = structlog.get_logger()


class OrderExecutor:
    """
    Places buy/sell orders on Kalshi demo and tracks them in the DB.
    In dry_run mode, logs what would happen without hitting the API.
    """

    def __init__(self, client: KalshiClient, dry_run: bool = True):
        self.client = client
        self.dry_run = dry_run

    def place_buy(
        self, ticker: str, side: str, price_cents: int, count: int
    ) -> dict | None:
        """
        Place a limit buy order.

        Args:
            ticker: market ticker
            side: "yes" or "no"
            price_cents: price in cents (1-99)
            count: number of contracts

        Returns:
            Order response dict, or None in dry_run mode.
        """
        client_order_id = str(uuid.uuid4())

        if self.dry_run:
            logger.info(
                "dry_run_buy",
                ticker=ticker,
                side=side,
                price_cents=price_cents,
                count=count,
                client_order_id=client_order_id,
            )
            self._log_order_to_db(
                order_id=f"DRY-{client_order_id[:8]}",
                client_order_id=client_order_id,
                ticker=ticker,
                action="buy",
                side=side,
                price_cents=price_cents,
                count=count,
                status="dry_run",
            )
            return {"order_id": f"DRY-{client_order_id[:8]}", "status": "dry_run"}

        try:
            resp = self.client.create_order(
                ticker=ticker,
                action="buy",
                side=side,
                count=count,
                yes_price=price_cents if side == "yes" else None,
                no_price=price_cents if side == "no" else None,
                order_type="limit",
                client_order_id=client_order_id,
            )
            order = resp.get("order", resp)
            order_id = order.get("order_id", "unknown")

            self._log_order_to_db(
                order_id=order_id,
                client_order_id=client_order_id,
                ticker=ticker,
                action="buy",
                side=side,
                price_cents=price_cents,
                count=count,
                status=order.get("status", "pending"),
            )

            logger.info("order_placed", order_id=order_id, ticker=ticker, side=side)
            return order

        except Exception as e:
            logger.error("order_failed", ticker=ticker, error=str(e))
            return None

    def place_sell(
        self, ticker: str, side: str, price_cents: int, count: int
    ) -> dict | None:
        """Place a limit sell order to close a position."""
        client_order_id = str(uuid.uuid4())

        if self.dry_run:
            logger.info(
                "dry_run_sell",
                ticker=ticker,
                side=side,
                price_cents=price_cents,
                count=count,
            )
            self._log_order_to_db(
                order_id=f"DRY-{client_order_id[:8]}",
                client_order_id=client_order_id,
                ticker=ticker,
                action="sell",
                side=side,
                price_cents=price_cents,
                count=count,
                status="dry_run",
            )
            return {"order_id": f"DRY-{client_order_id[:8]}", "status": "dry_run"}

        try:
            resp = self.client.create_order(
                ticker=ticker,
                action="sell",
                side=side,
                count=count,
                yes_price=price_cents if side == "yes" else None,
                no_price=price_cents if side == "no" else None,
                order_type="limit",
                client_order_id=client_order_id,
            )
            order = resp.get("order", resp)
            order_id = order.get("order_id", "unknown")

            self._log_order_to_db(
                order_id=order_id,
                client_order_id=client_order_id,
                ticker=ticker,
                action="sell",
                side=side,
                price_cents=price_cents,
                count=count,
                status=order.get("status", "pending"),
            )

            logger.info("sell_placed", order_id=order_id, ticker=ticker)
            return order

        except Exception as e:
            logger.error("sell_failed", ticker=ticker, error=str(e))
            return None

    def check_order_status(self, order_id: str, timeout: int = 30) -> str:
        """
        Poll order status until filled or timeout.

        Returns final status string: "filled", "resting", "cancelled", etc.
        """
        if order_id.startswith("DRY-"):
            return "dry_run"

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = self.client.get_orders(limit=1)
                orders = resp.get("orders", [])
                for o in orders:
                    if o.get("order_id") == order_id:
                        status = o.get("status", "unknown")
                        if status in ("filled", "cancelled", "expired"):
                            return status
                time.sleep(2)
            except Exception as e:
                logger.warning("order_status_check_failed", error=str(e))
                time.sleep(2)

        return "timeout"

    def _log_order_to_db(
        self, order_id: str, client_order_id: str, ticker: str,
        action: str, side: str, price_cents: int, count: int, status: str,
    ):
        session = get_session()
        try:
            session.add(Order(
                order_id=order_id,
                client_order_id=client_order_id,
                market_ticker=ticker,
                strategy="active_trader",
                action=action,
                side=side,
                price_cents=price_cents,
                count=count,
                status=status,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning("order_db_log_failed", error=str(e))
        finally:
            session.close()
