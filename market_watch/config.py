from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _secret(name: str, default: str = "") -> str:
    """Read a setting from env first, then Streamlit secrets when available."""
    value = os.getenv(name)
    if value is not None:
        return str(value).strip()
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default.strip()


@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str = _secret("ALPACA_API_KEY")
    alpaca_api_secret: str = _secret("ALPACA_API_SECRET")
    alpaca_data_feed: str = _secret("ALPACA_DATA_FEED", "iex") or "iex"
    openai_api_key: str = _secret("OPENAI_API_KEY")
    openai_model: str = _secret("OPENAI_MODEL", "gpt-5.5") or "gpt-5.5"
    database_url: str = _secret("DATABASE_URL")
    app_password: str = _secret("APP_PASSWORD")
    database_path: Path = ROOT / (_secret("MARKET_WATCH_DB", "data/market_watch.db") or "data/market_watch.db")

    @property
    def live_market_data_enabled(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_api_secret)

    @property
    def ai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def hosted_database_enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def storage_label(self) -> str:
        return "Hosted PostgreSQL" if self.hosted_database_enabled else "Local SQLite"


settings = Settings()
