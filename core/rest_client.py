"""
Kalshi REST Client

Wraps every API call with:
- RSA-PSS authentication
- Rate limiting (token bucket)
- Automatic retries with backoff
- Structured error handling
- Decimal-safe price parsing

This is the ONLY module that touches the network for REST calls.
Everything else calls methods on this client.
"""

import time
import uuid
from decimal import Decimal
from typing import Any, Optional

import requests
import structlog

from config.settings import settings
from core.auth import KalshiAuth
from core.rate_limiter import RateLimiter

logger = structlog.get_logger()


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an error."""

    def __init__(self, status_code: int, message: str, response: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.response = response or {}
        super().__init__(f"[{status_code}] {message}")


class KalshiClient:
    """
    Dual-mode Kalshi REST API client.

    Public endpoints (get_markets, get_orderbook, get_trades, ...) ALWAYS hit
    production, because the demo sandbox has no meaningful liquidity or volume.

    Authenticated endpoints (get_balance, get_positions, create_order, ...) hit
    whatever environment this client was constructed for (`trade_env`):
      - trade_env="demo" → demo sandbox with demo credentials
      - trade_env="prod" → real prod with prod credentials
      - trade_env=None   → read-only; any authenticated call raises

    Prefer the module-level factories:
        data_client  = get_data_client()    # public market data only
        trade_client = get_trade_client()   # public data + active-env trading

    Usage:
        client = get_trade_client()
        markets = client.get_markets(series_ticker="KXBTC15M", status="open")
        balance = client.get_balance()
        order = client.create_order(ticker="...", action="buy", side="yes", count=5, yes_price=45)
    """

    # Sentinel meaning "use settings.kalshi_env" — distinct from an explicit None
    # which means "no auth at all" (read-only data client).
    _USE_ACTIVE_ENV = object()

    def __init__(self, trade_env=_USE_ACTIVE_ENV):
        # Public endpoints always use production
        self.public_base_url = settings.prod_base_url

        # Default: follow the active trading env from settings.
        # Pass trade_env=None explicitly for a read-only client.
        if trade_env is KalshiClient._USE_ACTIVE_ENV:
            trade_env = settings.kalshi_env

        # Authenticated endpoints use the requested trading env
        self.trade_env = trade_env
        if trade_env is None:
            self.private_base_url: str | None = None
            self.auth: KalshiAuth | None = None
        else:
            if trade_env not in ("demo", "prod"):
                raise ValueError(f"trade_env must be 'demo' or 'prod', got {trade_env!r}")
            self.private_base_url = settings.base_url_for(trade_env)
            api_key_id, private_key_path = settings.creds_for(trade_env)
            if not api_key_id:
                raise ValueError(
                    f"Missing API key ID for trade_env={trade_env!r}. "
                    f"Set KALSHI_{trade_env.upper()}_API_KEY_ID in .env."
                )
            self.auth = KalshiAuth(api_key_id, private_key_path)

        # Backwards-compat: callers and log lines still look at `base_url`.
        # Point it at the private URL (trading target); if read-only, point at prod public.
        self.base_url = self.private_base_url or self.public_base_url

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Rate limiters — 80% of Basic tier as safety margin
        self._read_limiter = RateLimiter(max_per_second=16)
        self._write_limiter = RateLimiter(max_per_second=8)

        # Track for diagnostics
        self._request_count = 0
        self._error_count = 0

        logger.info(
            "kalshi_client_init",
            trade_env=trade_env or "read-only",
            public_base_url=self.public_base_url,
            private_base_url=self.private_base_url,
        )

    # ------------------------------------------------------------------
    # Internal request machinery
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        authenticated: bool = True,
        retries: int = 3,
    ) -> dict:
        """
        Make a rate-limited, authenticated request with retries.

        Args:
            method: GET, POST, PUT, DELETE
            path: e.g. "/markets" — will be prefixed with /trade-api/v2 if needed
            params: query parameters
            json_body: request body for POST/PUT
            authenticated: whether to add auth headers (False for public endpoints)
            retries: max retry attempts on transient errors
        """
        # Pick the base URL for this request:
        #   - public endpoints (authenticated=False) always go to prod
        #   - authenticated endpoints go to whichever env this client was built for
        if authenticated:
            if self.auth is None or self.private_base_url is None:
                raise RuntimeError(
                    "This client is read-only (trade_env=None). "
                    "Use get_trade_client() for authenticated calls."
                )
            request_base_url = self.private_base_url
        else:
            request_base_url = self.public_base_url

        # Build full URL and the exact path that will appear in the HTTP request.
        # The signed path MUST match the request path, including the /trade-api/v2 prefix.
        if path.startswith("/trade-api/"):
            host = request_base_url.split("/trade-api/")[0]
            url = host + path
            signed_path = path
        else:
            url = request_base_url + path
            # Derive the full path from request_base_url (e.g. "/trade-api/v2") + relative path
            base_path = "/trade-api/" + request_base_url.split("/trade-api/", 1)[1]
            signed_path = base_path + path

        # Rate limit
        is_write = method.upper() in ("POST", "PUT", "DELETE") and "orders" in path
        limiter = self._write_limiter if is_write else self._read_limiter
        limiter.wait()

        # Build headers — sign the full path (without query params; auth.get_headers strips them)
        headers = {}
        if authenticated:
            headers = self.auth.get_headers(method.upper(), signed_path)

        # Retry loop
        last_error = None
        for attempt in range(retries):
            try:
                self._request_count += 1

                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=10,
                )

                # Success
                if resp.status_code in (200, 201):
                    return resp.json()

                # Rate limited — wait and retry
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("rate_limited", wait_seconds=wait, attempt=attempt + 1)
                    time.sleep(wait)
                    # Re-sign with fresh timestamp
                    if authenticated:
                        headers = self.auth.get_headers(method.upper(), signed_path)
                    continue

                # Client errors — don't retry
                if 400 <= resp.status_code < 500:
                    self._error_count += 1
                    error_body = {}
                    try:
                        error_body = resp.json()
                    except Exception:
                        pass
                    raise KalshiAPIError(
                        resp.status_code,
                        error_body.get("message", resp.text[:200]),
                        error_body,
                    )

                # Server errors — retry
                if resp.status_code >= 500:
                    last_error = KalshiAPIError(resp.status_code, f"Server error: {resp.text[:200]}")
                    wait = 2 ** attempt
                    logger.warning("server_error", status=resp.status_code, wait=wait)
                    time.sleep(wait)
                    if authenticated:
                        headers = self.auth.get_headers(method.upper(), signed_path)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("request_exception", error=str(e), wait=wait)
                time.sleep(wait)
                if authenticated:
                    headers = self.auth.get_headers(method.upper(), path)
                continue

        self._error_count += 1
        raise last_error or KalshiAPIError(0, "Max retries exceeded")

    # ------------------------------------------------------------------
    # Public Market Data (no auth needed)
    # ------------------------------------------------------------------

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = "open",
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Fetch markets with optional filters."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params, authenticated=False)

    def get_market(self, ticker: str) -> dict:
        """Fetch a single market by ticker."""
        return self._request("GET", f"/markets/{ticker}", authenticated=False)

    def get_orderbook(self, ticker: str) -> dict:
        """Fetch the orderbook for a market. Returns yes_dollars and no_dollars arrays."""
        return self._request("GET", f"/markets/{ticker}/orderbook", authenticated=False)

    def get_series(self, ticker: str) -> dict:
        """Fetch series info."""
        return self._request("GET", f"/series/{ticker}", authenticated=False)

    def get_event(self, ticker: str) -> dict:
        """Fetch event info."""
        return self._request("GET", f"/events/{ticker}", authenticated=False)

    def get_trades(self, ticker: str | None = None, limit: int = 100) -> dict:
        """Fetch public trades."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/markets/trades", params=params, authenticated=False)

    # ------------------------------------------------------------------
    # Authenticated — Portfolio
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """
        Get your cash balance.

        Kalshi returns `balance` and `portfolio_value` in integer cents.
        We convert both to Decimal dollars here so every caller downstream
        can treat them as dollar amounts without re-doing the math.
        """
        resp = self._request("GET", "/portfolio/balance")
        if "balance" in resp:
            resp["balance"] = (Decimal(resp["balance"]) / Decimal(100)).quantize(Decimal("0.01"))
        if "portfolio_value" in resp:
            resp["portfolio_value"] = (Decimal(resp["portfolio_value"]) / Decimal(100)).quantize(Decimal("0.01"))
        return resp

    def get_positions(self, limit: int = 100) -> dict:
        """Get current positions."""
        return self._request("GET", "/portfolio/positions", params={"limit": limit})

    def get_fills(self, limit: int = 100, ticker: str | None = None) -> dict:
        """Get your fill history."""
        params: dict = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/portfolio/fills", params=params)

    def get_orders(
        self, status: str | None = None, ticker: str | None = None, limit: int = 100
    ) -> dict:
        """Get your orders."""
        params: dict = {"limit": limit}
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/portfolio/orders", params=params)

    # ------------------------------------------------------------------
    # Authenticated — Order Management
    # ------------------------------------------------------------------

    def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        yes_price: int | None = None,
        no_price: int | None = None,
        order_type: str = "limit",
        client_order_id: str | None = None,
    ) -> dict:
        """
        Place an order.

        Args:
            ticker: market ticker
            action: "buy" or "sell"
            side: "yes" or "no"
            count: number of contracts
            yes_price: price in cents (1-99) for YES side
            no_price: price in cents (1-99) for NO side
            order_type: "limit" or "market"
            client_order_id: unique ID for deduplication (auto-generated if None)

        Returns:
            Order response dict with order_id, status, etc.
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price

        logger.info(
            "create_order",
            ticker=ticker,
            action=action,
            side=side,
            count=count,
            price=yes_price or no_price,
        )

        return self._request("POST", "/portfolio/orders", json_body=body)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order."""
        logger.info("cancel_order", order_id=order_id)
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def amend_order(
        self,
        order_id: str,
        count: int | None = None,
        price: int | None = None,
    ) -> dict:
        """Amend a resting order's count or price."""
        body: dict = {}
        if count is not None:
            body["count"] = count
        if price is not None:
            body["price"] = price
        logger.info("amend_order", order_id=order_id, **body)
        return self._request("PUT", f"/portfolio/orders/{order_id}", json_body=body)

    def batch_create_orders(self, orders: list[dict]) -> dict:
        """Create multiple orders in one call."""
        logger.info("batch_create_orders", count=len(orders))
        return self._request("POST", "/portfolio/orders/batched", json_body={"orders": orders})

    def batch_cancel_orders(self, order_ids: list[str]) -> dict:
        """Cancel multiple orders. Each cancel counts as 0.2 write transactions."""
        logger.info("batch_cancel_orders", count=len(order_ids))
        return self._request(
            "DELETE", "/portfolio/orders/batched", json_body={"order_ids": order_ids}
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_orderbook(raw: dict) -> dict:
        """
        Parse raw orderbook into useful structure.

        Returns:
            {
                "best_yes_bid": Decimal,
                "best_yes_ask": Decimal,
                "best_no_bid": Decimal,
                "best_no_ask": Decimal,
                "spread": Decimal,
                "yes_bids": [(price, quantity), ...],
                "no_bids": [(price, quantity), ...],
            }
        """
        ob = raw.get("orderbook_fp", raw.get("orderbook", {}))
        yes_dollars = ob.get("yes_dollars", [])
        no_dollars = ob.get("no_dollars", [])

        # Parse into Decimal tuples — arrays are sorted ascending, last = best
        yes_bids = [(Decimal(p), Decimal(q)) for p, q in yes_dollars]
        no_bids = [(Decimal(p), Decimal(q)) for p, q in no_dollars]

        best_yes_bid = yes_bids[-1][0] if yes_bids else Decimal("0")
        best_no_bid = no_bids[-1][0] if no_bids else Decimal("0")

        # Implied asks via the $1.00 reciprocal
        best_yes_ask = Decimal("1.00") - best_no_bid if no_bids else Decimal("1.00")
        best_no_ask = Decimal("1.00") - best_yes_bid if yes_bids else Decimal("1.00")

        spread = best_yes_ask - best_yes_bid

        return {
            "best_yes_bid": best_yes_bid,
            "best_yes_ask": best_yes_ask,
            "best_no_bid": best_no_bid,
            "best_no_ask": best_no_ask,
            "spread": spread,
            "yes_bids": yes_bids,
            "no_bids": no_bids,
        }

    @staticmethod
    def compute_depth(bids: list[tuple[Decimal, Decimal]], levels: int = 5) -> Decimal:
        """Sum quantity of the top N price levels."""
        top = bids[-levels:] if len(bids) >= levels else bids
        return sum(q for _, q in top)

    def get_stats(self) -> dict:
        """Return client diagnostics."""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "trade_env": self.trade_env,
            "public_base_url": self.public_base_url,
            "private_base_url": self.private_base_url,
        }


# ----------------------------------------------------------------------
# Module-level factories
# ----------------------------------------------------------------------

def get_data_client() -> KalshiClient:
    """
    Read-only client for public market data. Always reads from production.

    Use this when you just need real prices, orderbooks, volume, etc.
    Calling any authenticated method (get_balance, create_order, ...) will raise.
    """
    return KalshiClient(trade_env=None)


def get_trade_client() -> KalshiClient:
    """
    Authenticated client for placing orders in the active trading env
    (`settings.kalshi_env` — "demo" or "prod").

    Public endpoints on this client still read from production, so you can
    freely mix price reads and order placement without juggling two instances.
    """
    return KalshiClient(trade_env=settings.kalshi_env)
