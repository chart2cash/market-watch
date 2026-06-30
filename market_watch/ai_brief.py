from __future__ import annotations

import json

from .config import Settings


def create_market_brief(settings: Settings, market_payload: dict) -> str:
    if not settings.ai_enabled:
        return (
            "AI commentary is disabled. Add OPENAI_API_KEY to .env to generate a personalized market brief. "
            "The dashboard remains fully usable without it."
        )

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    instructions = """
You write a cautious market-monitoring brief for one investor. Use only the supplied data.
Separate verified observations from interpretation. Never claim certainty, never promise returns,
and never issue an unconditional buy or sell instruction. Highlight portfolio/watchlist relevance,
conflicting signals, and concrete items to verify. Keep it under 450 words with headings:
Facts, Interpretation, Watchlist Relevance, Risks, Next Checks.
""".strip()
    response = client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=json.dumps(market_payload, default=str),
    )
    return response.output_text
