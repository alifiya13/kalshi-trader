"""
Central configuration. Loaded from .env file.
Every setting is validated at startup — no silent failures.

Dual-mode credentials:
  - PROD credentials are used only for reading authenticated production data
    (rarely needed — public prod endpoints don't require auth).
  - DEMO credentials are used for placing test orders on the sandbox.
  - KALSHI_ENV selects which environment is "active" for authenticated
    trading endpoints (create_order, get_balance, etc.).
  - Public endpoints (get_markets, get_orderbook, get_trades) ALWAYS read
    from production regardless of KALSHI_ENV, because the demo sandbox
    has no real liquidity or volume data.
"""

from decimal import Decimal
from pydantic_settings import BaseSettings
from pydantic import field_validator


PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"


class Settings(BaseSettings):
    # --- Kalshi credentials (two sets, one per environment) ---
    kalshi_prod_api_key_id: str = ""
    kalshi_prod_private_key_path: str = "./keys/kalshi-prod.pem"

    kalshi_demo_api_key_id: str = ""
    kalshi_demo_private_key_path: str = "./keys/kalshi-demo.pem"

    # Which env is "active" for authenticated trading endpoints.
    # Public endpoints always hit prod regardless of this setting.
    kalshi_env: str = "demo"  # "demo" or "prod"

    # --- Database ---
    database_url: str = "sqlite:///kalshi_trader.db"

    # --- Risk ---
    max_position_pct: Decimal = Decimal("0.02")
    daily_loss_limit_pct: Decimal = Decimal("0.05")
    weekly_loss_limit_pct: Decimal = Decimal("0.10")
    max_category_exposure_pct: Decimal = Decimal("0.30")
    kelly_fraction: Decimal = Decimal("0.25")
    min_edge_to_trade: Decimal = Decimal("0.05")

    # --- LLM Gateway (OpenAI-compatible) ---
    llm_gateway_api_key: str = ""
    llm_gateway_base_url: str = "https://api.llmgateway.io/v1"

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "logs/trader.log"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ------------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------------
    @property
    def prod_base_url(self) -> str:
        """Always production — used for public market data reads."""
        return PROD_BASE_URL

    @property
    def active_base_url(self) -> str:
        """Base URL for authenticated trading, follows KALSHI_ENV."""
        return PROD_BASE_URL if self.kalshi_env == "prod" else DEMO_BASE_URL

    @property
    def active_ws_url(self) -> str:
        return PROD_WS_URL if self.kalshi_env == "prod" else DEMO_WS_URL

    # Backwards-compat alias — existing callers (scan_markets, watch_orderbook)
    # still reference settings.base_url. Points at the active trading env.
    @property
    def base_url(self) -> str:
        return self.active_base_url

    @property
    def ws_url(self) -> str:
        return self.active_ws_url

    # ------------------------------------------------------------------
    # Active-env credential accessors
    # ------------------------------------------------------------------
    @property
    def trade_api_key_id(self) -> str:
        """API key ID for the active trading environment."""
        return self.kalshi_prod_api_key_id if self.kalshi_env == "prod" else self.kalshi_demo_api_key_id

    @property
    def trade_private_key_path(self) -> str:
        """Private key PEM path for the active trading environment."""
        return (
            self.kalshi_prod_private_key_path
            if self.kalshi_env == "prod"
            else self.kalshi_demo_private_key_path
        )

    def creds_for(self, env: str) -> tuple[str, str]:
        """
        Return (api_key_id, private_key_path) for a specific env.
        Raises ValueError if env is unknown.
        """
        if env == "prod":
            return self.kalshi_prod_api_key_id, self.kalshi_prod_private_key_path
        if env == "demo":
            return self.kalshi_demo_api_key_id, self.kalshi_demo_private_key_path
        raise ValueError(f"unknown env '{env}', expected 'demo' or 'prod'")

    def base_url_for(self, env: str) -> str:
        if env == "prod":
            return PROD_BASE_URL
        if env == "demo":
            return DEMO_BASE_URL
        raise ValueError(f"unknown env '{env}', expected 'demo' or 'prod'")

    @field_validator("kalshi_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ("demo", "prod"):
            raise ValueError(f"kalshi_env must be 'demo' or 'prod', got '{v}'")
        return v

    def is_demo(self) -> bool:
        return self.kalshi_env == "demo"


# Singleton — import this everywhere
settings = Settings()
