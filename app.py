import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client

from strategy import (
    ASSETS,
    STARTING_CASH,
    USER_KEY,
    CRYPTO_TICKERS,
    apply_cycle,
    get_crypto_candles,
    get_crypto_quotes,
    get_data,
    get_live_quotes,
    is_bullish,
    movers_table,
    period_returns,
    portfolio_value,
    position_price,
    positions_table,
    utc_now,
)

st.set_page_config(
    page_title="JARVIS Trading AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def fmt_time(value):
    if not value:
        return "Not recorded"
    try:
        return pd.to_datetime(value, utc=True).strftime(
            "%d %b %Y, %H:%M UTC"
        )
    except Exception:
        return str(value)


st.title("🤖 JARVIS")
st.caption("Trading dashboard · Virtual money only")
st.warning("Paper trading only. No real orders are placed.")

# ---------- SECRETS ----------
required = [
    "JARVIS_APP_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
]
missing = [
    key for key in required
    if key not in st.secrets or not str(st.secrets[key]).strip()
]
if missing:
    st.error("Missing Streamlit Secrets: " + ", ".join(missing))
    st.stop()

APP_PASSWORD = str(st.secrets["JARVIS_APP_PASSWORD"])
SUPABASE_URL = str(st.secrets["SUPABASE_URL"]).strip().rstrip("/")
SUPABASE_SECRET_KEY = str(st.secrets["SUPABASE_SECRET_KEY"]).strip()

if not SUPABASE_URL.startswith("https://"):
    st.error("Check SUPABASE_URL in Streamlit Secrets.")
    st.stop()

# ---------- LOGIN ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔐 Unlock JARVIS")
    with st.form("jarvis_login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Unlock",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ---------- DATABASE ----------
@st.cache_resource
def get_supabase(url, key):
    return create_client(url, key)


try:
    db = get_supabase(SUPABASE_URL, SUPABASE_SECRET_KEY)
except Exception as exc:
    st.error("Could not connect to Supabase.")
    st.caption(str(exc)[:300])
    st.stop()


def load_state():
    result = (
        db.table("paper_trading")
        .select("data,updated_at")
        .eq("user_key", USER_KEY)
        .limit(1)
        .execute()
    )
    if not result.data:
        st.error(
            "No saved portfolio was found. To avoid accidentally "
            "creating a fresh portfolio, this app will not initialize one."
        )
        st.stop()

    saved = result.data[0].get("data")
    if not isinstance(saved, dict):
        st.error("Saved portfolio data is not valid JSON.")
        st.stop()

    state = dict(saved)
    state.setdefault("cash", STARTING_CASH)
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    state.setdefault("equity_snapshots", [])
    state["_db_updated_at"] = result.data[0].get("updated_at")
    return state


if "jarvis_state" not in st.session_state:
    try:
        st.session_state.jarvis_state = load_state()
    except Exception as exc:
        st.error("Could not load the saved portfolio.")
        st.code(f"{type(exc).__name__}: {str(exc)[:300]}")
        st.stop()


def refresh_saved_state():
    try:
        st.session_state.jarvis_state = load_state()
        st.session_state.last_refresh = utc_now()
        st.rerun()
    except Exception as exc:
        st.error("Refresh failed.")
        st.caption(str(exc)[:300])


state = st.session_state.jarvis_state


def save_state(new_state):
    # Read the newest saved JSON first so unknown fields and worker
    # metadata are not discarded by the app.
    result = (
        db.table("paper_trading")
        .select("data")
        .eq("user_key", USER_KEY)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise RuntimeError("Saved portfolio row is missing.")

    latest = result.data[0].get("data")
    if not isinstance(latest, dict):
        raise RuntimeError("Saved portfolio data is invalid.")

    # Merge the app's portfolio fields into the saved state.
    merged = dict(latest)
    for key in ("cash", "positions", "trades"):
        if key in new_state:
            merged[key] = new_state[key]

    db.table("paper_trading").upsert(
        {
            "user_key": USER_KEY,
            "data": merged,
            "updated_at": utc_now(),
        },
        on_conflict="user_key",
    ).execute()
    st.session_state.jarvis_state = merged


# ---------- MARKET DATA ----------
@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(ticker):
    return get_data(ticker)


# ---------- LIVE PANEL ----------
@st.cache_data(ttl=10, show_spinner=False)
def cached_live_quotes(tickers):
    # Yahoo: stocks and forex (and crypto fallback).
    return get_live_quotes(tickers)


@st.cache_data(ttl=1, show_spinner=False)
def cached_crypto_quotes(tickers):
    # Coinbase: near-live crypto.
    return get_crypto_quotes(tickers)


@st.cache_data(ttl=30, show_spinner=False)
def cached_candles(ticker):
    # Coinbase asks apps not to poll candle history often.
    return get_crypto_candles(ticker)


STOCK_TICKERS = tuple(t for t in ASSETS.values() if t not in CRYPTO_TICKERS)


def all_quotes():
    """Coinbase for crypto, Yahoo for the rest (and as a crypto fallback)."""
    quotes = dict(cached_live_quotes(STOCK_TICKERS))
    crypto = cached_crypto_quotes(CRYPTO_TICKERS)
    quotes.update(crypto)
    missing = tuple(t for t in CRYPTO_TICKERS if t not in crypto)
    if missing:
        quotes.update(cached_live_quotes(missing))
    return quotes


LIVE_SECONDS = 2


def live_dashboard():
    """Winning/losing banner, movers and open positions.

    Runs as a fragment, so it refreshes by itself without reloading
    the rest of the page.
    """
    live_state = st.session_state.jarvis_state
    quotes = all_quotes()
    live_prices = {t: q["price"] for t, q in quotes.items()}

    live_cash = float(live_state.get("cash", STARTING_CASH))
    live_positions = live_state.get("positions", {})
    live_trades = live_state.get("trades", [])

    equity = portfolio_value(live_cash, live_positions, live_prices)
    start_value = float(live_state.get("starting_equity", STARTING_CASH))
    total_pnl = equity - start_value
    total_return = total_pnl / start_value * 100 if start_value else 0

    unrealized = sum(
        float(p["units"]) * position_price(t, p, live_prices) - float(p["cost"])
        for t, p in live_positions.items()
    )
    closed = [
        t for t in live_trades
        if t.get("action") == "SELL" and t.get("pnl") is not None
    ]
    realized = sum(float(t["pnl"]) for t in closed)
    wins = sum(1 for t in closed if float(t["pnl"]) > 0)
    win_rate = wins / len(closed) * 100 if closed else 0

    st.subheader("⚡ Live")
    if total_pnl > 0:
        st.success(f"🟢 Winning: {total_pnl:+,.2f} USD ({total_return:+.2f}%)")
    elif total_pnl < 0:
        st.error(f"🔴 At a loss: {total_pnl:+,.2f} USD ({total_return:+.2f}%)")
    else:
        st.info("⚪ Break-even so far.")

    a, b = st.columns(2)
    a.metric("Portfolio value", f"${equity:,.2f}", f"{total_pnl:+.2f} USD")
    b.metric("Cash", f"${live_cash:,.2f}")
    c, d = st.columns(2)
    c.metric("Unrealized P/L", f"${unrealized:+,.2f}")
    d.metric(
        "Realized P/L",
        f"${realized:+,.2f}",
        f"{win_rate:.0f}% win rate" if closed else None,
        delta_color="off",
    )

    st.markdown(
        "**What's up and down** · stocks/forex vs previous close, "
        "crypto vs 24 hours ago"
    )
    movers = movers_table(quotes)
    if movers.empty:
        st.warning("Live prices unavailable right now; retrying.")
    else:
        st.dataframe(movers, use_container_width=True, hide_index=True)

    st.markdown("**Live crypto chart** · 1-minute, last 3 hours")
    coin_names = {t: n for n, t in ASSETS.items()}
    coin_tabs = st.tabs([coin_names.get(t, t) for t in CRYPTO_TICKERS])
    for tab, ticker in zip(coin_tabs, CRYPTO_TICKERS):
        with tab:
            candles = cached_candles(ticker)
            if candles is None or candles.empty:
                st.caption("Live chart unavailable right now.")
            else:
                st.line_chart(candles["Close"])

    st.markdown("**Open positions**")
    if live_positions:
        st.dataframe(
            positions_table(live_positions, live_prices),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No open positions right now.")

    missing_live = [n for n, t in ASSETS.items() if t not in quotes]
    if missing_live:
        st.caption("No live price for: " + ", ".join(missing_live))
    st.caption(
        f"Updated {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC · "
        "Crypto: Coinbase, refreshed every few seconds. Stocks and forex: "
        "Yahoo Finance, which can be delayed ~15 minutes and shows the "
        "last close when the market is shut."
    )


live_on = st.toggle(
    f"⚡ Live updates (every {LIVE_SECONDS} seconds)", value=True
)
st.fragment(run_every=LIVE_SECONDS if live_on else None)(live_dashboard)()

st.divider()
head_left, head_right = st.columns([2, 1])
with head_left:
    st.subheader("📊 Daily charts & trend")
with head_right:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state.jarvis_state = load_state()
        st.session_state.last_refresh = utc_now()
        st.rerun()

market = {}
prices = {}
errors = []

with st.spinner("Loading market data..."):
    for name, ticker in ASSETS.items():
        try:
            df = get_market_data(ticker)
            if df is None:
                errors.append(f"{name}: no data")
                continue
            usable = df.dropna(subset=["Close"])
            if usable.empty:
                errors.append(f"{name}: no data")
                continue
            market[ticker] = df
            prices[ticker] = float(usable["Close"].iloc[-1])
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")

for name, ticker in ASSETS.items():
    if ticker not in market:
        continue

    df = market[ticker]
    valid = df.dropna(subset=["Close"])
    last = valid.iloc[-1]
    prev = valid.iloc[-2] if len(valid) > 1 else last
    price = prices[ticker]

    change = (
        (price / float(prev["Close"]) - 1) * 100
        if float(prev["Close"]) else 0
    )

    # Same rule the trading cycle uses: last completed daily candle.
    trend = "Bullish" if is_bullish(df) else "Bearish / unclear"

    with st.container(border=True):
        st.markdown(f"**{name}** · `{ticker}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest available price", f"{price:,.4f}", f"{change:+.2f}%")
        rsi = last.get("RSI")
        c2.metric("RSI", f"{rsi:.1f}" if pd.notna(rsi) else "—")
        c3.metric("EMA trend", trend)

        with st.expander("View chart"):
            st.line_chart(valid[["Close", "EMA20", "EMA50"]].tail(90))

st.caption(
    "Daily historical data from Yahoo Finance; "
    "not guaranteed to be real-time. "
    "Trend uses the last completed daily candle."
)
for error in errors:
    st.warning(error)

# ---------- PORTFOLIO DATA (used by reports below) ----------
trades = state.get("trades", [])
initial_value = float(state.get("starting_equity", STARTING_CASH))

st.caption("Last database update: " + fmt_time(state.get("_db_updated_at")))

# ---------- DAILY / MONTHLY REPORTS ----------
st.divider()
st.subheader("📈 Performance history")

snapshots = state.get("equity_snapshots", [])

if snapshots:
    history = pd.DataFrame(snapshots)
    if {"time", "equity"}.issubset(history.columns):
        history["time"] = pd.to_datetime(
            history["time"], utc=True, errors="coerce"
        )
        history["equity"] = pd.to_numeric(history["equity"], errors="coerce")
        history = history.dropna(subset=["time", "equity"]).sort_values("time")

        if not history.empty:
            history["day"] = history["time"].dt.strftime("%Y-%m-%d")
            history["month"] = history["time"].dt.strftime("%Y-%m")

            st.line_chart(history.set_index("time")["equity"])

            # Each period is compared with the END of the previous period
            # (the first one with the starting equity).
            daily = period_returns(history, initial_value, "day")
            monthly = period_returns(history, initial_value, "month")

            tab_day, tab_month = st.tabs(["Daily", "Monthly"])
            with tab_day:
                st.dataframe(
                    daily.iloc[::-1].round(2),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"Winning days: {(daily['P/L'] > 0).sum()} · "
                    f"Losing days: {(daily['P/L'] < 0).sum()}"
                )
            with tab_month:
                st.dataframe(
                    monthly.iloc[::-1].round(2),
                    use_container_width=True,
                    hide_index=True,
                )
else:
    st.info(
        "Performance snapshots will appear after the Worker is updated "
        "to save daily portfolio values. Historical daily/monthly equity "
        "cannot be reconstructed accurately from trade records alone."
    )

# ---------- TRADE HISTORY ----------
st.divider()
st.subheader("🧾 Trade history")

if trades:
    trade_df = pd.DataFrame(trades)
    if "time" in trade_df.columns:
        trade_df = trade_df.sort_values("time", ascending=False)
    st.dataframe(trade_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades recorded yet.")

# ---------- PAPER TRADING ----------
st.divider()
st.subheader("⚙️ Paper-trading controls")
st.caption(
    "The GitHub Actions Worker runs independently. "
    "This button requests a manual virtual trading cycle."
)

if st.button("▶️ Run paper-trading cycle", type="primary",
             use_container_width=True):
    # Daily candles give the signal; fills use the freshest live price.
    live = all_quotes()
    fill_prices = {**prices, **{t: q["price"] for t, q in live.items()}}
    changes, actions = apply_cycle(state, market, fill_prices)
    try:
        save_state(changes)
        st.success(
            "Virtual trading cycle saved. "
            + (" · ".join(actions) if actions else "No trades needed.")
        )
        st.rerun()
    except Exception as exc:
        st.error("Could not save the cycle.")
        st.code(f"{type(exc).__name__}: {str(exc)[:300]}")

# ---------- SAVE / LOCK ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reload saved portfolio", use_container_width=True):
        refresh_saved_state()
with col2:
    if st.button("🔒 Lock JARVIS", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.caption(
    "Prices may be delayed. Historical daily prices are not live quotes. "
    "All trades shown here are simulated; no real money is used."
)
