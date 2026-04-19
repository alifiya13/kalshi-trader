"""
Kalshi RSA-PSS Authentication

Every authenticated request must be signed with your private key.
Signature = RSA-PSS(timestamp_ms + HTTP_METHOD + path_without_query_params)

This module handles:
- Loading your private key from disk
- Generating signed headers for any request
- Timestamp generation in milliseconds
"""

import base64
import time
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiAuth:
    """Manages RSA-PSS request signing for Kalshi API."""

    def __init__(self, api_key_id: str, private_key_path: str):
        self.api_key_id = api_key_id
        self._private_key_path = private_key_path
        self._private_key = None

    def _ensure_key_loaded(self) -> rsa.RSAPrivateKey:
        """Lazy-load the private key on first use."""
        if self._private_key is None:
            self._private_key = self._load_key(self._private_key_path)
        return self._private_key

    @staticmethod
    def _load_key(path: str) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        key_path = Path(path)
        if not key_path.exists():
            raise FileNotFoundError(
                f"Private key not found at {key_path.resolve()}. "
                f"Generate one at kalshi.com/account/profile → API Keys."
            )
        with open(key_path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def _sign(self, message: str) -> str:
        """Sign a message string with RSA-PSS, return base64."""
        key = self._ensure_key_loaded()
        signature = key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """
        Generate signed auth headers for a Kalshi API request.

        Args:
            method: HTTP method (GET, POST, DELETE, PUT)
            path: Request path WITHOUT base URL, e.g. /trade-api/v2/portfolio/balance
                  Query params are stripped automatically before signing.

        Returns:
            Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP
        """
        timestamp_ms = str(int(time.time() * 1000))

        # CRITICAL: strip query params before signing
        clean_path = path.split("?")[0]
        message = timestamp_ms + method.upper() + clean_path

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(message),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
