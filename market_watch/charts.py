from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .analytics import add_indicators


def price_chart(frame: pd.DataFrame, symbol: str) -> go.Figure:
    df = add_indicators(frame)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol.upper(),
        )
    )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sma20"], name="SMA 20", line={"width": 1.5}))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sma50"], name="SMA 50", line={"width": 1.5}))
    fig.update_layout(
        height=560,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h"},
        title=f"{symbol.upper()} daily price",
    )
    return fig


def sector_chart(frame: pd.DataFrame) -> go.Figure:
    ordered = frame.sort_values("change_pct")
    fig = go.Figure(go.Bar(x=ordered["change_pct"], y=ordered["symbol"], orientation="h"))
    fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="Daily change (%)",
        yaxis_title="",
        title="Sector ETF performance",
    )
    return fig
