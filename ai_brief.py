from __future__ import annotations

import json
from typing import Literal

from .config import Settings

BriefLength = Literal["Concise", "Detailed"]


def _client(settings: Settings):
    if not settings.ai_enabled:
        return None
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


def create_market_brief(settings: Settings, market_payload: dict, length: BriefLength = "Concise") -> str:
    client = _client(settings)
    if client is None:
        return (
            "AI commentary is disabled. Add OPENAI_API_KEY to Streamlit Secrets to generate a market brief. "
            "The dashboard remains fully usable without it."
        )

    word_limit = 260 if length == "Concise" else 520
    instructions = f"""
You are the analytical briefing layer in a personal stock-market dashboard. Use only the supplied data.
Write for an active investor who holds speculative growth names and also watches the broad market.
Separate observations from interpretation. Never promise returns and never issue an unconditional buy or sell order.
Do not pad the response. Maximum {word_limit} words.

Use exactly these headings:
## Market status
State Risk-on, Neutral, or Risk-off, followed by a brief explanation grounded in the supplied data.
## What is moving
Give no more than three important index, sector, rate, volatility, or news developments.
## What matters to your watchlist
Explain only items that connect to the supplied action queue, watchlist, positions, or news.
## Opportunities to research
Give zero to three candidates. Explain why each surfaced; label them as research candidates, not recommendations.
## Key risks
List the most important near-term risks or conflicting signals.
## Today's action list
Give no more than five concrete monitoring or research actions. Say “No action warranted” when appropriate.

Be direct, specific, and easy to scan on a phone.
""".strip()
    response = client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=json.dumps(market_payload, default=str),
    )
    return response.output_text.strip()


def create_stock_analysis(
    settings: Settings,
    stock_payload: dict,
    length: BriefLength = "Concise",
) -> str:
    client = _client(settings)
    if client is None:
        return "AI commentary is disabled. Add OPENAI_API_KEY to Streamlit Secrets."

    word_limit = 320 if length == "Concise" else 650
    instructions = f"""
You are the stock-research layer in a personal market dashboard. Analyze only the supplied ticker data,
technical metrics, news, saved watchlist levels, and user notes. Do not invent fundamentals, analyst targets,
earnings dates, or events that are not in the payload. Clearly say when data is insufficient.
Never promise returns and never issue an unconditional buy or sell instruction. Maximum {word_limit} words.

Use exactly these headings:
## Snapshot
Summarize the current setup and trend in two or three sentences.
## Bull case
Identify the strongest evidence supporting upside or improving conditions.
## Bear case
Identify the strongest evidence supporting downside or deterioration.
## Catalysts and news
Summarize only meaningful supplied headlines and distinguish facts from possible implications.
## Technical setup
Interpret trend, momentum, volume, moving-average distance, and any saved levels.
## Confirmation and invalidation
State what observable developments would strengthen or weaken the setup.
## Personalized action plan
Give up to four research or monitoring actions tied to the user's saved levels and notes. Do not tell the user to buy or sell.

Keep the writing practical, balanced, and optimized for phone reading.
""".strip()
    response = client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=json.dumps(stock_payload, default=str),
    )
    return response.output_text.strip()
