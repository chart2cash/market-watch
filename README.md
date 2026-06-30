# Market Watch — Cloud-Ready MVP

A private, browser-based stock-market dashboard with persistent online storage, watchlists, stock research, screening, portfolio tracking, alerts, news, and optional AI-assisted commentary.

## Cloud-ready additions

- Persistent Supabase/PostgreSQL storage through `DATABASE_URL`
- Local SQLite fallback for testing
- Password gate through `APP_PASSWORD`
- Streamlit Secrets support
- Automatic database-table creation
- Database-independent CSV/ZIP backups
- Deployment configuration for Streamlit Community Cloud

## Main features

- Market overview with indexes, sectors, trend interpretation, and action queue
- Alpaca market data and news with clearly labeled demo fallbacks
- Watchlists with targets, alerts, theses, and conviction scores
- Stock pages with candlesticks, SMA 20, SMA 50, RSI, volume, notes, and news
- Explainable technical screener
- Trade ledger and open-position calculations
- Saved price alerts
- CSV trade import and full backup download
- Optional OpenAI-generated market brief

## Online deployment

Follow [DEPLOY_ONLINE.md](DEPLOY_ONLINE.md). The recommended stack is:

- GitHub private repository
- Streamlit Community Cloud
- Supabase Shared Pooler/PostgreSQL

The app creates the required tables automatically after `DATABASE_URL` is configured.

## Local testing

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Without `DATABASE_URL`, the app stores local test data in `data/market_watch.db`.

## Secrets

For Streamlit Community Cloud, use `.streamlit/secrets.example.toml` as the template in the platform's Secrets editor. Do not commit real secrets.

Required for durable online storage:

```text
DATABASE_URL
```

Recommended for private access:

```text
APP_PASSWORD
```

Optional:

```text
ALPACA_API_KEY
ALPACA_API_SECRET
ALPACA_DATA_FEED
OPENAI_API_KEY
OPENAI_MODEL
```

## Tests

```bash
pytest
```

## Safety boundary

This application is a research, organization, and monitoring tool. It does not place trades and should not be treated as personalized financial advice. Verify live data and material company information with authoritative sources before acting.
