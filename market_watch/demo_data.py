from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import numpy as np
import pandas as pd


BASE_PRICES = {
    "SPY": 612.0, "QQQ": 548.0, "DIA": 445.0, "IWM": 227.0,
    "XLK": 252.0, "XLF": 53.0, "XLE": 91.0, "XLV": 151.0,
    "XLY": 221.0, "XLP": 84.0, "XLI": 143.0, "XLB": 94.0,
    "XLU": 81.0, "XLRE": 43.0, "XLC": 105.0,
    "AAPL": 228.0, "MSFT": 502.0, "NVDA": 164.0, "AMZN": 233.0,
    "GOOGL": 191.0, "META": 708.0, "TSLA": 349.0, "IONQ": 47.0,
    "PLTR": 141.0, "RKLB": 39.0, "AMD": 167.0, "AVGO": 281.0,
}


def _seed(symbol: str) -> int:
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def bars(symbol: str, days: int = 260) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    rng = np.random.default_rng(_seed(symbol))
    dates = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=max(days, 65))
    base = BASE_PRICES.get(symbol, 40.0 + (_seed(symbol) % 240))
    drift = 0.00035 + ((_seed(symbol) % 9) - 4) * 0.00006
    vol = 0.014 + (_seed(symbol) % 11) * 0.0008
    returns = rng.normal(drift, vol, len(dates))
    close = base * np.exp(np.cumsum(returns))
    close = close * (base / close[-1])
    overnight = rng.normal(0, vol / 3, len(dates))
    open_ = close * (1 + overnight)
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, len(dates)))
    volume_base = 2_000_000 + (_seed(symbol) % 45_000_000)
    volume = rng.lognormal(np.log(volume_base), 0.28, len(dates)).astype(int)

    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return frame.tail(days).reset_index(drop=True)


def snapshots(symbols: list[str]) -> pd.DataFrame:
    records: list[dict] = []
    for symbol in symbols:
        frame = bars(symbol, 65)
        latest = frame.iloc[-1]
        previous = frame.iloc[-2]
        records.append(
            {
                "symbol": symbol.upper(),
                "price": float(latest["close"]),
                "previous_close": float(previous["close"]),
                "change": float(latest["close"] - previous["close"]),
                "change_pct": float((latest["close"] / previous["close"] - 1) * 100),
                "volume": int(latest["volume"]),
                "timestamp": latest["timestamp"],
                "source": "Demo simulation",
            }
        )
    return pd.DataFrame(records)


def news(symbols: list[str] | None = None, limit: int = 12) -> list[dict]:
    symbols = symbols or ["SPY", "QQQ"]
    now = datetime.now(timezone.utc)
    templates = [
        ("Market breadth improves as investors evaluate the next catalyst", "Market context"),
        ("Technology shares show relative strength while rates remain in focus", "Sector trend"),
        ("Investors prepare for earnings and upcoming economic data", "Calendar watch"),
        ("High-growth stocks remain volatile near key technical levels", "Risk watch"),
        ("Analysts debate whether the latest move has durable participation", "Opinion roundup"),
    ]
    output = []
    for index in range(limit):
        symbol = symbols[index % len(symbols)].upper()
        headline, category = templates[index % len(templates)]
        output.append(
            {
                "id": f"demo-{index}",
                "headline": f"{symbol}: {headline}",
                "summary": (
                    "Demonstration article generated for the no-key version of Market Watch. "
                    "Connect Alpaca to replace this with current market news."
                ),
                "source": "Market Watch Demo",
                "url": "",
                "created_at": (now - pd.Timedelta(minutes=37 * index)).isoformat(),
                "symbols": [symbol],
                "category": category,
                "is_demo": True,
            }
        )
    return output
