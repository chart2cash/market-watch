from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

from .config import Settings
from . import demo_data


class MarketDataError(RuntimeError):
    pass


class MarketDataService:
    """Alpaca-backed market data with a deterministic demo fallback."""

    BASE_URL = "https://data.alpaca.markets"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.live = settings.live_market_data_enabled

    @property
    def mode_label(self) -> str:
        return "Live Alpaca data" if self.live else "Demo data"

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_api_secret,
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            response = requests.get(
                f"{self.BASE_URL}{path}",
                headers=self._headers(),
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise MarketDataError(f"Market-data request failed: {exc}") from exc

    def snapshots(self, symbols: Iterable[str]) -> pd.DataFrame:
        clean = sorted({s.upper().strip() for s in symbols if s and s.strip()})
        if not clean:
            return pd.DataFrame()
        if not self.live:
            return demo_data.snapshots(clean)

        payload = self._get(
            "/v2/stocks/snapshots",
            {"symbols": ",".join(clean), "feed": self.settings.alpaca_data_feed},
        )
        records = []
        for symbol in clean:
            snap = payload.get(symbol, {})
            daily = snap.get("dailyBar") or snap.get("daily_bar") or {}
            previous = snap.get("prevDailyBar") or snap.get("previousDailyBar") or snap.get("prev_daily_bar") or {}
            trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
            price = trade.get("p") or trade.get("price") or daily.get("c") or daily.get("close")
            previous_close = previous.get("c") or previous.get("close")
            if price is None:
                continue
            change = (float(price) - float(previous_close)) if previous_close else 0.0
            change_pct = (change / float(previous_close) * 100) if previous_close else 0.0
            records.append(
                {
                    "symbol": symbol,
                    "price": float(price),
                    "previous_close": float(previous_close) if previous_close else None,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": int(daily.get("v") or daily.get("volume") or 0),
                    "timestamp": trade.get("t") or daily.get("t"),
                    "source": f"Alpaca ({self.settings.alpaca_data_feed})",
                }
            )
        return pd.DataFrame(records)

    def bars(self, symbol: str, days: int = 260) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        if not self.live:
            return demo_data.bars(symbol, days)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(int(days * 1.75), 120))
        payload = self._get(
            f"/v2/stocks/{symbol}/bars",
            {
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": "all",
                "feed": self.settings.alpaca_data_feed,
                "limit": 10000,
                "sort": "asc",
            },
        )
        raw_bars = payload.get("bars", [])
        rows = []
        for bar in raw_bars:
            rows.append(
                {
                    "timestamp": pd.to_datetime(bar.get("t"), utc=True),
                    "open": float(bar.get("o")),
                    "high": float(bar.get("h")),
                    "low": float(bar.get("l")),
                    "close": float(bar.get("c")),
                    "volume": int(bar.get("v", 0)),
                }
            )
        return pd.DataFrame(rows).tail(days).reset_index(drop=True)

    def news(self, symbols: Iterable[str] | None = None, limit: int = 15) -> list[dict]:
        clean = sorted({s.upper().strip() for s in (symbols or []) if s and s.strip()})
        if not self.live:
            return demo_data.news(clean or ["SPY", "QQQ"], limit)

        params = {"limit": min(limit, 50), "sort": "desc"}
        if clean:
            params["symbols"] = ",".join(clean)
        payload = self._get("/v1beta1/news", params)
        articles = payload.get("news", payload if isinstance(payload, list) else [])
        output = []
        for item in articles[:limit]:
            output.append(
                {
                    "id": item.get("id"),
                    "headline": item.get("headline", "Untitled"),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "Alpaca News"),
                    "url": item.get("url", ""),
                    "created_at": item.get("created_at") or item.get("updated_at"),
                    "symbols": item.get("symbols", []),
                    "category": "Market news",
                    "is_demo": False,
                }
            )
        return output
