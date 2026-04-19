"""
Exchange constants. These rarely change — update when Kalshi docs change.
"""

# Rate limits by tier (requests per second)
RATE_LIMITS = {
    "basic":    {"read": 20, "write": 10},
    "advanced": {"read": 30, "write": 30},
    "premier":  {"read": 100, "write": 100},
    "prime":    {"read": 400, "write": 400},
}

# Write-limited endpoints (everything else is a read)
WRITE_ENDPOINTS = frozenset([
    "/portfolio/orders",          # POST = create, DELETE = cancel
    "/portfolio/orders/batched",  # POST = batch create, DELETE = batch cancel
])

# Market categories
CATEGORIES = [
    "sports", "crypto", "economics", "politics",
    "climate", "culture", "entertainment", "other",
]

# High-value series tickers to watch
WATCHED_SERIES = {
    "crypto": [
        "KXBTC15M",    # BTC 15-minute
        "KXBTC1H",     # BTC hourly
        "KXETH15M",    # ETH 15-minute
        "KXETH1H",     # ETH hourly
    ],
    "weather": [
        "KXHIGHNY",    # NYC high temp
        "KXHIGHCH",    # Chicago high temp
        "KXHIGHLA",    # LA high temp
    ],
    "economics": [
        "CPI",         # Consumer Price Index
        "FED",         # Fed rate decisions
        "GDPNOW",     # GDP nowcast
    ],
}

# Minimum spread to consider a market tradeable (in dollars)
MIN_TRADEABLE_SPREAD = 0.05  # $0.05

# Minimum volume to consider a market (in contracts)
MIN_TRADEABLE_VOLUME = 100
