from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy().sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return df
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["avg_volume20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["avg_volume20"]
    return df


def stock_metrics(frame: pd.DataFrame, symbol: str) -> dict:
    df = add_indicators(frame)
    if df.empty:
        return {"symbol": symbol.upper()}
    latest = df.iloc[-1]

    def period_return(period: int) -> float | None:
        if len(df) <= period:
            return None
        return float((latest["close"] / df.iloc[-period - 1]["close"] - 1) * 100)

    sma20 = latest.get("sma20")
    sma50 = latest.get("sma50")
    return {
        "symbol": symbol.upper(),
        "price": float(latest["close"]),
        "return_1d_pct": period_return(1),
        "return_5d_pct": period_return(5),
        "return_20d_pct": period_return(20),
        "sma20": None if pd.isna(sma20) else float(sma20),
        "sma50": None if pd.isna(sma50) else float(sma50),
        "distance_sma20_pct": None if pd.isna(sma20) else float((latest["close"] / sma20 - 1) * 100),
        "distance_sma50_pct": None if pd.isna(sma50) else float((latest["close"] / sma50 - 1) * 100),
        "rsi14": None if pd.isna(latest.get("rsi14")) else float(latest["rsi14"]),
        "volume": int(latest["volume"]),
        "volume_ratio": None if pd.isna(latest.get("volume_ratio")) else float(latest["volume_ratio"]),
    }


def market_regime(spy_bars: pd.DataFrame) -> tuple[str, str]:
    df = add_indicators(spy_bars)
    if len(df) < 50:
        return "Insufficient data", "Not enough history to evaluate the trend."
    latest = df.iloc[-1]
    close = latest["close"]
    sma20 = latest["sma20"]
    sma50 = latest["sma50"]
    volatility = df["close"].pct_change().tail(20).std() * np.sqrt(252) * 100

    if close > sma20 > sma50:
        regime = "Risk-on / constructive"
        note = "Price is above both major trend averages, with the shorter average leading."
    elif close < sma20 < sma50:
        regime = "Risk-off / defensive"
        note = "Price is below both trend averages, and near-term momentum is weaker."
    else:
        regime = "Mixed / transition"
        note = "Trend signals disagree, so confirmation and position sizing matter more."
    return regime, f"{note} Estimated 20-day annualized volatility: {volatility:.1f}%."


def calculate_positions(trades: pd.DataFrame, prices: dict[str, float]) -> pd.DataFrame:
    columns = [
        "symbol", "quantity", "average_cost", "cost_basis", "current_price",
        "market_value", "unrealized_pl", "unrealized_pl_pct", "realized_pl"
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for symbol, group in trades.sort_values("trade_date").groupby("symbol"):
        qty = 0.0
        avg_cost = 0.0
        realized = 0.0
        for _, trade in group.iterrows():
            tqty = float(trade["quantity"])
            price = float(trade["price"])
            fees = float(trade.get("fees", 0) or 0)
            if trade["side"].upper() == "BUY":
                new_qty = qty + tqty
                avg_cost = ((qty * avg_cost) + (tqty * price) + fees) / new_qty if new_qty else 0
                qty = new_qty
            else:
                sell_qty = min(tqty, qty)
                realized += sell_qty * (price - avg_cost) - fees
                qty -= sell_qty
                if qty <= 1e-10:
                    qty = 0.0
                    avg_cost = 0.0
        current_price = float(prices.get(symbol, avg_cost or 0))
        market_value = qty * current_price
        cost_basis = qty * avg_cost
        unrealized = market_value - cost_basis
        records.append(
            {
                "symbol": symbol,
                "quantity": qty,
                "average_cost": avg_cost,
                "cost_basis": cost_basis,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pl": unrealized,
                "unrealized_pl_pct": (unrealized / cost_basis * 100) if cost_basis else 0.0,
                "realized_pl": realized,
            }
        )
    return pd.DataFrame(records, columns=columns)
