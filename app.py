from __future__ import annotations

from datetime import date, datetime
import hmac
import html
import io
import json
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.exc import IntegrityError

from market_watch.analytics import add_indicators, calculate_positions, market_regime, stock_metrics
from market_watch.charts import price_chart, sector_chart
from market_watch.config import settings
from market_watch.db import Database
from market_watch.demo_data import bars as demo_bars
from market_watch.demo_data import news as demo_news
from market_watch.demo_data import snapshots as demo_snapshots
from market_watch.market_data import MarketDataError, MarketDataService



def _ai_client(settings):
    if not settings.ai_enabled:
        return None
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


def create_market_brief(settings, market_payload: dict, length: str = "Concise") -> str:
    client = _ai_client(settings)
    if client is None:
        return "AI commentary is disabled. Add OPENAI_API_KEY to Streamlit Secrets."
    word_limit = 260 if length == "Concise" else 520
    instructions = f"""
You are the analytical briefing layer in a personal stock-market dashboard. Use only the supplied data.
Write for an active investor who holds speculative growth names and also watches the broad market.
Separate observations from interpretation. Never promise returns and never issue an unconditional buy or sell order.
Do not pad the response. Maximum {word_limit} words.

Use exactly these headings:
## Market status
## What is moving
## What matters to your watchlist
## Opportunities to research
## Key risks
## Today's action list

Be direct, specific, and easy to scan on a phone.
""".strip()
    response = client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=json.dumps(market_payload, default=str),
    )
    return response.output_text.strip()


def create_stock_analysis(settings, stock_payload: dict, length: str = "Concise") -> str:
    client = _ai_client(settings)
    if client is None:
        return "AI commentary is disabled. Add OPENAI_API_KEY to Streamlit Secrets."
    word_limit = 320 if length == "Concise" else 650
    instructions = f"""
You are the stock-research layer in a personal market dashboard. Analyze only the supplied ticker data,
technical metrics, news, saved watchlist levels, and user notes. Do not invent missing facts.
Never promise returns and never issue an unconditional buy or sell instruction. Maximum {word_limit} words.

Use exactly these headings:
## Snapshot
## Bull case
## Bear case
## Catalysts and news
## Technical setup
## Confirmation and invalidation
## Personalized action plan

Keep the writing practical, balanced, and optimized for phone reading.
""".strip()
    response = client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=json.dumps(stock_payload, default=str),
    )
    return response.output_text.strip()


st.set_page_config(
    page_title="Market Watch",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 10px;}
.mw-card {background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); padding: 14px 16px; border-radius: 10px; margin-bottom: 10px;}
.mw-muted {opacity: 0.72; font-size: 0.9rem;}
.mw-live {display:inline-block; padding:4px 9px; border-radius:999px; border:1px solid rgba(255,255,255,.18); font-size:.82rem;}
</style>
""",
    unsafe_allow_html=True,
)


def require_app_password() -> None:
    """Simple single-user password gate for hosted deployments."""
    if not settings.app_password or st.session_state.get("mw_authenticated"):
        return

    st.title("📈 Market Watch")
    st.caption("Enter the dashboard password to continue.")
    password = st.text_input("Password", type="password", key="mw_password")
    if st.button("Unlock", type="primary"):
        if hmac.compare_digest(password, settings.app_password):
            st.session_state["mw_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_app_password()


@st.cache_resource
def get_database() -> Database:
    return Database(path=settings.database_path, database_url=settings.database_url)


@st.cache_resource
def get_market_service() -> MarketDataService:
    return MarketDataService(settings)


db = get_database()
market = get_market_service()


@st.cache_data(ttl=90, show_spinner=False)
def cached_snapshots(symbols: tuple[str, ...]) -> pd.DataFrame:
    return market.snapshots(list(symbols))


@st.cache_data(ttl=900, show_spinner=False)
def cached_bars(symbol: str, days: int) -> pd.DataFrame:
    return market.bars(symbol, days)


@st.cache_data(ttl=300, show_spinner=False)
def cached_news(symbols: tuple[str, ...], limit: int) -> list[dict]:
    return market.news(list(symbols), limit)


def get_snapshots(symbols: list[str]) -> pd.DataFrame:
    clean = tuple(sorted({s.upper().strip() for s in symbols if s and s.strip()}))
    try:
        return cached_snapshots(clean)
    except MarketDataError as exc:
        st.warning(f"Live market request failed, so this section is using demo data. {exc}")
        return demo_snapshots(list(clean))


def get_bars(symbol: str, days: int) -> pd.DataFrame:
    try:
        return cached_bars(symbol.upper(), days)
    except MarketDataError as exc:
        st.warning(f"Live history request failed, so this chart is using demo data. {exc}")
        return demo_bars(symbol, days)


def get_news(symbols: list[str], limit: int = 12) -> list[dict]:
    clean = tuple(sorted({s.upper().strip() for s in symbols if s and s.strip()}))
    try:
        return cached_news(clean, limit)
    except MarketDataError as exc:
        st.warning(f"Live news request failed, so this section is using demo articles. {exc}")
        return demo_news(list(clean), limit)


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${value:,.2f}"


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.2f}%"


def display_news(articles: list[dict]) -> None:
    if not articles:
        st.info("No articles were returned for this selection.")
        return
    for article in articles:
        created = article.get("created_at") or ""
        try:
            created = pd.to_datetime(created).strftime("%b %d, %Y %I:%M %p")
        except Exception:
            pass
        symbols = html.escape(", ".join(article.get("symbols", [])))
        headline = html.escape(str(article.get("headline", "Untitled")))
        url = html.escape(str(article.get("url", "")), quote=True)
        title = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{headline}</a>' if url else headline
        source = html.escape(str(article.get("source", "")))
        summary = html.escape(str(article.get("summary", "")))
        created_text = html.escape(str(created))
        st.markdown(
            f"""
<div class="mw-card">
<strong>{title}</strong><br>
<span class="mw-muted">{source} · {created_text}{' · ' + symbols if symbols else ''}</span><br>
{summary}
</div>
""",
            unsafe_allow_html=True,
        )


def action_queue(items: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    if items.empty or snapshots.empty:
        return pd.DataFrame()
    merged = items.merge(snapshots[["symbol", "price", "change_pct"]], on="symbol", how="left")
    rows = []
    for _, row in merged.iterrows():
        price = row.get("price")
        if pd.isna(price):
            continue
        reasons = []
        priority = 0
        target = row.get("target_buy")
        below = row.get("alert_below")
        above = row.get("alert_above")
        if pd.notna(target):
            distance = (price / target - 1) * 100
            if abs(distance) <= 5:
                reasons.append(f"Within {abs(distance):.1f}% of target buy level")
                priority += 3
            elif price < target:
                reasons.append("Trading below target buy level")
                priority += 4
        if pd.notna(below) and price <= below:
            reasons.append("Below downside alert")
            priority += 5
        if pd.notna(above) and price >= above:
            reasons.append("Above upside alert")
            priority += 5
        if abs(float(row.get("change_pct", 0) or 0)) >= 4:
            reasons.append("Large daily move")
            priority += 2
        if reasons:
            rows.append(
                {
                    "priority": priority,
                    "symbol": row["symbol"],
                    "price": price,
                    "daily_change": row.get("change_pct"),
                    "watchlist": row.get("watchlist_name"),
                    "reason": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows).sort_values(["priority", "symbol"], ascending=[False, True]) if rows else pd.DataFrame()


def overview_page() -> None:
    st.title("Market Watch")
    st.caption("Personal market dashboard, watchlist monitor, screener, portfolio tracker, and research journal.")
    badge = "LIVE" if market.live else "DEMO"
    st.markdown(f'<span class="mw-live">{badge} · {market.mode_label}</span>', unsafe_allow_html=True)

    index_symbols = ["SPY", "QQQ", "DIA", "IWM"]
    sector_symbols = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    index_data = get_snapshots(index_symbols)
    sector_data = get_snapshots(sector_symbols)

    st.subheader("Market pulse")
    cols = st.columns(4)
    lookup = index_data.set_index("symbol") if not index_data.empty else pd.DataFrame()
    for col, symbol in zip(cols, index_symbols):
        if symbol in lookup.index:
            row = lookup.loc[symbol]
            col.metric(symbol, money(row["price"]), pct(row["change_pct"]))
        else:
            col.metric(symbol, "—", "—")

    left, right = st.columns([1.15, 1])
    with left:
        spy_bars = get_bars("SPY", 130)
        regime, regime_note = market_regime(spy_bars)
        st.subheader("Trend interpretation")
        st.markdown(f"<div class='mw-card'><strong>{regime}</strong><br>{regime_note}</div>", unsafe_allow_html=True)
        if not spy_bars.empty:
            chart_df = add_indicators(spy_bars)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["close"], name="SPY"))
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["sma20"], name="SMA 20"))
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["sma50"], name="SMA 50"))
            fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 20, "b": 10}, legend={"orientation": "h"})
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if not sector_data.empty:
            st.plotly_chart(sector_chart(sector_data), use_container_width=True)

    st.subheader("Personal action queue")
    items = db.watchlist_items()
    watch_symbols = items["symbol"].tolist() if not items.empty else []
    watch_snaps = get_snapshots(watch_symbols) if watch_symbols else pd.DataFrame()
    queue = action_queue(items, watch_snaps)
    if queue.empty:
        st.success("No saved target or alert rule is currently demanding attention.")
    else:
        display = queue[["symbol", "price", "daily_change", "watchlist", "reason"]].copy()
        display.columns = ["Symbol", "Price", "Daily %", "Watchlist", "Why it surfaced"]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Daily %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.subheader("Latest market and watchlist news")
    news_symbols = watch_symbols[:12] or index_symbols
    articles = get_news(news_symbols, 10)
    display_news(articles)

    with st.expander("AI-assisted market brief", expanded=bool(st.session_state.get("market_brief"))):
        st.caption("Choose a compact phone-friendly brief or a deeper review. AI output is analysis, not an automated trade instruction.")
        brief_length = st.radio("Brief length", ["Concise", "Detailed"], horizontal=True, key="market_brief_length")
        if st.button("Generate market brief", type="primary"):
            payload = {
                "generated_at": datetime.now().isoformat(),
                "data_mode": market.mode_label,
                "market_regime": {"label": regime, "detail": regime_note},
                "indexes": index_data.to_dict("records"),
                "sectors": sector_data.to_dict("records"),
                "action_queue": queue.to_dict("records") if not queue.empty else [],
                "watchlist": items.to_dict("records") if not items.empty else [],
                "news": articles[:8],
            }
            with st.spinner("Generating commentary..."):
                try:
                    st.session_state["market_brief"] = create_market_brief(settings, payload, brief_length)
                    st.session_state["market_brief_generated_at"] = datetime.now().strftime("%b %d, %Y %I:%M %p")
                except Exception as exc:
                    st.error(f"AI brief could not be generated: {exc}")

        brief = st.session_state.get("market_brief")
        if brief:
            st.caption(f"Generated {st.session_state.get('market_brief_generated_at', '')} · {market.mode_label}")
            st.markdown(brief)
            st.markdown("**Copy or download**")
            st.code(brief, language=None, wrap_lines=True)
            st.download_button(
                "Download market brief",
                data=brief,
                file_name=f"market_brief_{date.today().isoformat()}.txt",
                mime="text/plain",
                use_container_width=True,
            )


def watchlists_page() -> None:
    st.title("Watchlists")
    st.caption("Store target levels, thesis notes, conviction, and alert thresholds.")
    lists = db.watchlists()

    with st.expander("Create a watchlist"):
        with st.form("create_watchlist"):
            name = st.text_input("Watchlist name")
            submitted = st.form_submit_button("Create")
            if submitted and name.strip():
                try:
                    db.execute("INSERT INTO watchlists(name) VALUES (?)", (name.strip(),))
                    st.success("Watchlist created.")
                    st.rerun()
                except IntegrityError:
                    st.error("A watchlist with that name already exists.")

    list_lookup = dict(zip(lists["name"], lists["id"]))
    selected_name = st.selectbox("Watchlist", list_lookup.keys())
    selected_id = int(list_lookup[selected_name])
    items = db.watchlist_items(selected_id)

    symbols = items["symbol"].tolist() if not items.empty else []
    snaps = get_snapshots(symbols) if symbols else pd.DataFrame()
    view = items.merge(snaps, on="symbol", how="left") if not items.empty else pd.DataFrame()
    if not view.empty:
        shown = view[["id", "symbol", "price", "change_pct", "target_buy", "alert_below", "alert_above", "conviction", "thesis"]].copy()
        shown.columns = ["ID", "Symbol", "Price", "Daily %", "Target Buy", "Alert Below", "Alert Above", "Conviction", "Thesis"]
        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Daily %": st.column_config.NumberColumn(format="%.2f%%"),
                "Target Buy": st.column_config.NumberColumn(format="$%.2f"),
                "Alert Below": st.column_config.NumberColumn(format="$%.2f"),
                "Alert Above": st.column_config.NumberColumn(format="$%.2f"),
                "Conviction": st.column_config.ProgressColumn(min_value=1, max_value=5),
            },
        )
    else:
        st.info("This watchlist is empty.")

    left, right = st.columns(2)
    with left:
        st.subheader("Add or update a symbol")
        with st.form("upsert_watch_item"):
            symbol = st.text_input("Symbol").upper().strip()
            c1, c2, c3 = st.columns(3)
            target = c1.number_input("Target buy", min_value=0.0, value=0.0, step=0.25)
            alert_below = c2.number_input("Alert below", min_value=0.0, value=0.0, step=0.25)
            alert_above = c3.number_input("Alert above", min_value=0.0, value=0.0, step=0.25)
            conviction = st.slider("Conviction", 1, 5, 3)
            thesis = st.text_area("Thesis / notes")
            save = st.form_submit_button("Save symbol", type="primary")
            if save and symbol:
                db.execute(
                    """
                    INSERT INTO watchlist_items(watchlist_id,symbol,target_buy,alert_below,alert_above,thesis,conviction)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(watchlist_id,symbol) DO UPDATE SET
                    target_buy=excluded.target_buy, alert_below=excluded.alert_below,
                    alert_above=excluded.alert_above, thesis=excluded.thesis, conviction=excluded.conviction
                    """,
                    (
                        selected_id, symbol, target or None, alert_below or None,
                        alert_above or None, thesis.strip(), conviction,
                    ),
                )
                st.success(f"{symbol} saved to {selected_name}.")
                st.rerun()
    with right:
        st.subheader("Remove a symbol")
        if items.empty:
            st.caption("No symbols to remove.")
        else:
            remove_symbol = st.selectbox("Symbol to remove", items["symbol"].tolist(), key="remove_watch_symbol")
            if st.button("Remove from watchlist"):
                db.execute("DELETE FROM watchlist_items WHERE watchlist_id=? AND symbol=?", (selected_id, remove_symbol))
                st.success(f"{remove_symbol} removed.")
                st.rerun()


def research_page() -> None:
    st.title("Stock Research")
    watch_items = db.watchlist_items()
    default_symbols = watch_items["symbol"].tolist() if not watch_items.empty else ["SPY", "QQQ", "IONQ"]
    c1, c2 = st.columns([2, 1])
    symbol = c1.text_input("Ticker", value=default_symbols[0] if default_symbols else "SPY").upper().strip()
    period = c2.selectbox("History", [65, 130, 260, 520], index=2, format_func=lambda x: f"{x} trading days")
    if not symbol:
        st.stop()

    frame = get_bars(symbol, period)
    if frame.empty:
        st.error("No price history was returned.")
        st.stop()
    metrics = stock_metrics(frame, symbol)
    cols = st.columns(6)
    cols[0].metric("Price", money(metrics.get("price")), pct(metrics.get("return_1d_pct")))
    cols[1].metric("5-day", pct(metrics.get("return_5d_pct")))
    cols[2].metric("20-day", pct(metrics.get("return_20d_pct")))
    cols[3].metric("RSI 14", f"{metrics.get('rsi14', 0):.1f}" if metrics.get("rsi14") is not None else "—")
    cols[4].metric("vs SMA 20", pct(metrics.get("distance_sma20_pct")))
    cols[5].metric("Volume ratio", f"{metrics.get('volume_ratio', 0):.2f}x" if metrics.get("volume_ratio") is not None else "—")
    st.plotly_chart(price_chart(frame, symbol), use_container_width=True)

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Research notes")
        with st.form("new_note"):
            note_type = st.selectbox("Type", ["General", "Bull case", "Bear case", "Entry plan", "Exit plan", "Earnings", "Risk"])
            content = st.text_area("Note")
            save = st.form_submit_button("Save note")
            if save and content.strip():
                db.execute(
                    "INSERT INTO research_notes(symbol,note_type,content) VALUES (?,?,?)",
                    (symbol, note_type, content.strip()),
                )
                st.success("Note saved.")
                st.rerun()
        notes = db.notes(symbol)
        if notes.empty:
            st.caption("No notes saved for this symbol.")
        else:
            for _, note in notes.head(15).iterrows():
                st.markdown(
                    f"<div class='mw-card'><strong>{note['note_type']}</strong><br>{note['content']}<br>"
                    f"<span class='mw-muted'>{note['created_at']}</span></div>",
                    unsafe_allow_html=True,
                )
    with right:
        st.subheader(f"{symbol} news")
        symbol_news = get_news([symbol], 8)
        display_news(symbol_news)

    st.subheader(f"AI analysis: {symbol}")
    st.caption("Balanced analysis of the supplied chart, metrics, saved levels, notes, and news. It does not place trades or guarantee outcomes.")
    analysis_length = st.radio("Analysis length", ["Concise", "Detailed"], horizontal=True, key=f"stock_analysis_length_{symbol}")
    analysis_key = f"stock_analysis_{symbol}"
    generated_key = f"stock_analysis_generated_{symbol}"
    if st.button(f"Generate {symbol} analysis", type="primary", key=f"generate_analysis_{symbol}"):
        matching_watch = watch_items[watch_items["symbol"] == symbol] if not watch_items.empty else pd.DataFrame()
        saved_notes = db.notes(symbol)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "data_mode": market.mode_label,
            "symbol": symbol,
            "technical_metrics": metrics,
            "recent_price_bars": frame.tail(30).to_dict("records"),
            "saved_watchlist_rules": matching_watch.to_dict("records") if not matching_watch.empty else [],
            "saved_research_notes": saved_notes.head(20).to_dict("records") if not saved_notes.empty else [],
            "news": symbol_news,
        }
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                st.session_state[analysis_key] = create_stock_analysis(settings, payload, analysis_length)
                st.session_state[generated_key] = datetime.now().strftime("%b %d, %Y %I:%M %p")
            except Exception as exc:
                st.error(f"AI stock analysis could not be generated: {exc}")

    analysis = st.session_state.get(analysis_key)
    if analysis:
        st.caption(f"Generated {st.session_state.get(generated_key, '')} · {market.mode_label}")
        st.markdown(analysis)
        st.markdown("**Copy or download**")
        st.code(analysis, language=None, wrap_lines=True)
        st.download_button(
            f"Download {symbol} analysis",
            data=analysis,
            file_name=f"{symbol.lower()}_analysis_{date.today().isoformat()}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def screener_page() -> None:
    st.title("Personal Screener")
    st.caption("Run technical filters across a focused universe. The result explains why each symbol passed.")
    default_universe = "SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, IONQ, PLTR, RKLB, AMD, AVGO"
    universe_text = st.text_area("Symbols (comma-separated, maximum 30)", value=default_universe)
    c1, c2, c3, c4 = st.columns(4)
    max_rsi = c1.slider("Maximum RSI", 10, 100, 70)
    min_return20 = c2.slider("Minimum 20-day return", -50, 50, -20, format="%d%%")
    max_sma20_distance = c3.slider("Maximum % above SMA 20", -20, 50, 20, format="%d%%")
    min_volume_ratio = c4.slider("Minimum volume ratio", 0.0, 5.0, 0.0, 0.1)

    if st.button("Run screen", type="primary"):
        symbols = list(dict.fromkeys(s.upper().strip() for s in universe_text.split(",") if s.strip()))[:30]
        progress = st.progress(0, text="Calculating indicators...")
        rows = []
        for index, symbol in enumerate(symbols):
            frame = get_bars(symbol, 90)
            metric = stock_metrics(frame, symbol)
            if metric.get("price") is not None:
                rows.append(metric)
            progress.progress((index + 1) / max(len(symbols), 1), text=f"Analyzed {symbol}")
        progress.empty()
        results = pd.DataFrame(rows)
        if results.empty:
            st.warning("No usable data was returned.")
            return
        mask = (
            results["rsi14"].fillna(101).le(max_rsi)
            & results["return_20d_pct"].fillna(-999).ge(min_return20)
            & results["distance_sma20_pct"].fillna(999).le(max_sma20_distance)
            & results["volume_ratio"].fillna(0).ge(min_volume_ratio)
        )
        passed = results.loc[mask].copy()
        if passed.empty:
            st.info("No symbols passed every selected rule.")
            st.dataframe(results, use_container_width=True, hide_index=True)
            return

        def explain(row: pd.Series) -> str:
            reasons = [f"RSI {row['rsi14']:.1f}", f"20-day {row['return_20d_pct']:+.1f}%", f"vs SMA20 {row['distance_sma20_pct']:+.1f}%"]
            if row.get("volume_ratio", 0) >= 1.25:
                reasons.append(f"volume {row['volume_ratio']:.2f}x")
            return "; ".join(reasons)

        passed["why_passed"] = passed.apply(explain, axis=1)
        passed = passed.sort_values(["return_20d_pct", "volume_ratio"], ascending=[False, False])
        shown = passed[["symbol", "price", "return_1d_pct", "return_5d_pct", "return_20d_pct", "rsi14", "distance_sma20_pct", "volume_ratio", "why_passed"]]
        st.success(f"{len(passed)} of {len(results)} symbols passed.")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download results CSV", shown.to_csv(index=False), "screener_results.csv", "text/csv")


def portfolio_page() -> None:
    st.title("Portfolio & Trade Tracker")
    st.caption("Trades are stored locally. Positions and average cost are calculated from the ledger.")

    with st.expander("Record a trade", expanded=True):
        with st.form("trade_form"):
            c1, c2, c3, c4 = st.columns(4)
            symbol = c1.text_input("Symbol").upper().strip()
            side = c2.selectbox("Side", ["BUY", "SELL"])
            quantity = c3.number_input("Quantity", min_value=0.000001, value=1.0, step=1.0)
            price = c4.number_input("Price", min_value=0.0, value=0.0, step=0.01)
            c5, c6, c7 = st.columns(3)
            trade_date = c5.date_input("Trade date", value=date.today())
            fees = c6.number_input("Fees", min_value=0.0, value=0.0, step=0.01)
            account = c7.text_input("Account", value="Primary")
            strategy = st.text_input("Strategy / setup")
            notes = st.text_area("Trade notes")
            submit = st.form_submit_button("Save trade", type="primary")
            if submit and symbol and price > 0:
                db.execute(
                    """INSERT INTO trades(symbol,side,quantity,price,fees,trade_date,account,strategy,notes)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (symbol, side, quantity, price, fees, trade_date.isoformat(), account, strategy, notes),
                )
                st.success("Trade recorded.")
                st.rerun()

    trades = db.trades()
    if trades.empty:
        st.info("No trades have been recorded yet.")
        return

    symbols = trades["symbol"].unique().tolist()
    snaps = get_snapshots(symbols)
    prices = dict(zip(snaps["symbol"], snaps["price"])) if not snaps.empty else {}
    positions = calculate_positions(trades, prices)
    open_positions = positions[positions["quantity"] > 0].copy()

    if not open_positions.empty:
        total_value = open_positions["market_value"].sum()
        total_cost = open_positions["cost_basis"].sum()
        unrealized = open_positions["unrealized_pl"].sum()
        realized = positions["realized_pl"].sum()
        cols = st.columns(4)
        cols[0].metric("Market value", money(total_value))
        cols[1].metric("Open cost basis", money(total_cost))
        cols[2].metric("Unrealized P/L", money(unrealized), pct(unrealized / total_cost * 100 if total_cost else 0))
        cols[3].metric("Realized P/L", money(realized))
        st.subheader("Open positions")
        st.dataframe(
            open_positions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "average_cost": st.column_config.NumberColumn("Average cost", format="$%.2f"),
                "cost_basis": st.column_config.NumberColumn("Cost basis", format="$%.2f"),
                "current_price": st.column_config.NumberColumn("Current price", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Market value", format="$%.2f"),
                "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L", format="$%.2f"),
                "unrealized_pl_pct": st.column_config.NumberColumn("Unrealized %", format="%.2f%%"),
                "realized_pl": st.column_config.NumberColumn("Realized P/L", format="$%.2f"),
            },
        )
    else:
        st.info("There are no open positions in the ledger.")

    st.subheader("Trade ledger")
    st.dataframe(trades, use_container_width=True, hide_index=True)
    delete_id = st.selectbox("Delete trade ID", trades["id"].tolist())
    if st.button("Delete selected trade"):
        db.execute("DELETE FROM trades WHERE id=?", (int(delete_id),))
        st.success("Trade deleted.")
        st.rerun()


def alerts_page() -> None:
    st.title("Alerts")
    st.caption("Evaluate saved price rules on demand. Hosted scheduling and email/push delivery are Phase 2 additions.")

    with st.form("alert_form"):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("Symbol").upper().strip()
        alert_type = c2.selectbox("Condition", ["BELOW", "ABOVE"])
        threshold = c3.number_input("Threshold", min_value=0.01, value=1.0, step=0.25)
        notes = st.text_input("Notes")
        submit = st.form_submit_button("Create alert")
        if submit and symbol:
            db.execute(
                "INSERT INTO alerts(symbol,alert_type,threshold,notes) VALUES (?,?,?,?)",
                (symbol, alert_type, threshold, notes),
            )
            st.success("Alert created.")
            st.rerun()

    alerts = db.alerts()
    watch_items = db.watchlist_items()
    rule_rows = []
    for _, item in watch_items.iterrows():
        if pd.notna(item["alert_below"]):
            rule_rows.append({"source": item["watchlist_name"], "symbol": item["symbol"], "condition": "BELOW", "threshold": item["alert_below"], "notes": item["thesis"]})
        if pd.notna(item["alert_above"]):
            rule_rows.append({"source": item["watchlist_name"], "symbol": item["symbol"], "condition": "ABOVE", "threshold": item["alert_above"], "notes": item["thesis"]})
    for _, item in alerts[alerts["active"] == 1].iterrows():
        rule_rows.append({"source": "Standalone", "symbol": item["symbol"], "condition": item["alert_type"], "threshold": item["threshold"], "notes": item["notes"]})

    rules = pd.DataFrame(rule_rows)
    if rules.empty:
        st.info("No active alert rules exist.")
        return
    snaps = get_snapshots(rules["symbol"].unique().tolist())
    evaluated = rules.merge(snaps[["symbol", "price", "change_pct"]], on="symbol", how="left")
    evaluated["triggered"] = evaluated.apply(
        lambda row: bool(row["price"] <= row["threshold"]) if row["condition"] == "BELOW" else bool(row["price"] >= row["threshold"]),
        axis=1,
    )
    evaluated["status"] = evaluated["triggered"].map({True: "TRIGGERED", False: "Pending"})
    evaluated["distance_pct"] = evaluated.apply(
        lambda row: (row["price"] / row["threshold"] - 1) * 100 if row["threshold"] else None,
        axis=1,
    )
    st.dataframe(evaluated, use_container_width=True, hide_index=True)
    triggered = evaluated[evaluated["triggered"]]
    if not triggered.empty:
        st.error(f"{len(triggered)} alert rule(s) are currently triggered.")

    if not alerts.empty:
        alert_id = st.selectbox("Standalone alert ID", alerts["id"].tolist())
        c1, c2 = st.columns(2)
        if c1.button("Toggle active/inactive"):
            current = int(alerts.loc[alerts["id"] == alert_id, "active"].iloc[0])
            db.execute("UPDATE alerts SET active=? WHERE id=?", (0 if current else 1, int(alert_id)))
            st.rerun()
        if c2.button("Delete standalone alert"):
            db.execute("DELETE FROM alerts WHERE id=?", (int(alert_id),))
            st.rerun()


def make_backup_zip() -> bytes:
    buffer = io.BytesIO()
    tables = {
        "watchlists": db.watchlists(),
        "watchlist_items": db.watchlist_items(),
        "trades": db.trades(),
        "research_notes": db.notes(),
        "alerts": db.alerts(),
    }
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
        if not settings.hosted_database_enabled and settings.database_path.exists():
            archive.write(settings.database_path, arcname="market_watch.db")
    return buffer.getvalue()


def settings_page() -> None:
    st.title("Settings & Data")
    st.subheader("Connection status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Market data", "Alpaca" if market.live else "Demo")
    c2.metric("AI commentary", "Enabled" if settings.ai_enabled else "Disabled")
    c3.metric("Storage", db.storage_label)

    st.markdown(
        """
### Hosted configuration
For Streamlit Community Cloud, edit the app's **Secrets** and store `DATABASE_URL`, `APP_PASSWORD`, and any API credentials there. Never place real keys in GitHub.

### Enable live market data
Add `ALPACA_API_KEY` and `ALPACA_API_SECRET`. Keep `ALPACA_DATA_FEED=iex` for the typical free-data setup.

### Enable AI commentary
Add `OPENAI_API_KEY`. This is optional and billed separately from a ChatGPT subscription.
"""
    )

    st.subheader("Backup")
    st.download_button(
        "Download full backup ZIP",
        data=make_backup_zip(),
        file_name=f"market_watch_backup_{date.today().isoformat()}.zip",
        mime="application/zip",
    )

    st.subheader("Import trades")
    st.caption("CSV columns required: symbol, side, quantity, price, trade_date. Optional: fees, account, strategy, notes.")
    upload = st.file_uploader("Trade CSV", type=["csv"])
    if upload is not None:
        incoming = pd.read_csv(upload)
        st.dataframe(incoming.head(20), use_container_width=True)
        required = {"symbol", "side", "quantity", "price", "trade_date"}
        missing = required - set(incoming.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        elif st.button("Import these trades"):
            rows = []
            for _, row in incoming.iterrows():
                rows.append(
                    (
                        str(row["symbol"]).upper().strip(), str(row["side"]).upper().strip(),
                        float(row["quantity"]), float(row["price"]), float(row.get("fees", 0) or 0),
                        str(row["trade_date"]), str(row.get("account", "")), str(row.get("strategy", "")),
                        str(row.get("notes", "")),
                    )
                )
            db.executemany(
                """INSERT INTO trades(symbol,side,quantity,price,fees,trade_date,account,strategy,notes)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            st.success(f"Imported {len(rows)} trades.")
            st.rerun()

    st.subheader("Maintenance")
    if st.button("Clear market-data cache"):
        st.cache_data.clear()
        st.success("Cached prices, charts, and news cleared.")
    if settings.hosted_database_enabled:
        st.caption("Persistent database: hosted PostgreSQL")
        st.success("Your watchlists, trades, notes, and alerts are stored outside the Streamlit container and survive app restarts.")
    else:
        st.caption(f"Local database: {settings.database_path}")
        st.warning("Local SQLite is suitable for personal testing, but hosted deployments should configure DATABASE_URL for durable storage.")


PAGES = {
    "Overview": overview_page,
    "Watchlists": watchlists_page,
    "Stock Research": research_page,
    "Screener": screener_page,
    "Portfolio & Trades": portfolio_page,
    "Alerts": alerts_page,
    "Settings & Data": settings_page,
}

with st.sidebar:
    st.title("📈 Market Watch")
    selected_page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"Mode: {market.mode_label}")
    st.caption(f"Storage: {db.storage_label}")
    if settings.app_password and st.button("Lock dashboard"):
        st.session_state["mw_authenticated"] = False
        st.rerun()
    st.caption("Research and monitoring tool—not personalized financial advice or an automated trading system.")

PAGES[selected_page]()
