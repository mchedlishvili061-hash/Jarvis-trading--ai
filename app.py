
import streamlit as st
import pandas as pd
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

LIVE_SECONDS = 2

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.portfolio-card {
    background: linear-gradient(135deg, #202329, #141519);
    border: 1px solid #383b43;
    border-radius: 20px;
    padding: 25px 22px;
    margin: 8px 0 18px 0;
}
.portfolio-label {
    color: #a7aab4;
    font-size: 13px;
    letter-spacing: 1.5px;
}
.portfolio-value {
    color: white;
    font-size: clamp(38px, 8vw, 58px);
    font-weight: 750;
    line-height: 1.3;
}
.portfolio-pnl {
    font-size: 18px;
    font-weight: 650;
    margin-top: 5px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    min-height: 48px;
    border-radius: 12px;
    font-weight: 600;
}
div[data-testid="stMetric"] {
    background: #1c1e23;
    border: 1px solid #34363d;
    border-radius: 14px;
    padding: 12px;
}
div[data-testid="stTabs"] button {
    min-height: 44px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 JARVIS V5")
st.caption("Paper trading · Simulated money only")

# =========================================================
# SECRETS AND LOGIN
# =========================================================

required = [
    "JARVIS_APP_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
]

missing = [
    key for key in required
    if key not in st.secrets
    or not str(st.secrets[key]).strip()
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

# =========================================================
# DATABASE
# =========================================================

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
        raise RuntimeError(
            "Saved portfolio missing. Refusing to create a new portfolio."
        )

    saved = result.data[0].get("data")

    if not isinstance(saved, dict):
        raise RuntimeError("Saved portfolio data is invalid.")

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


def save_state(new_state):
    # Preserve fields written by the background Worker.
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

    merged["_db_updated_at"] = utc_now()
    st.session_state.jarvis_state = merged


def fmt_time(value):
    if not value:
        return "Not recorded"

    try:
        return pd.to_datetime(value, utc=True).strftime(
            "%d %b %Y, %H:%M UTC"
        )
    except Exception:
        return str(value)


# =========================================================
# MARKET DATA
# =========================================================

STOCK_TICKERS = tuple(
    ticker for ticker in ASSETS.values()
    if ticker not in CRYPTO_TICKERS
)


@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(ticker):
    return get_data(ticker)


@st.cache_data(ttl=10, show_spinner=False)
def cached_live_quotes(tickers):
    return get_live_quotes(tickers)


@st.cache_data(ttl=1, show_spinner=False)
def cached_crypto_quotes(tickers):
    return get_crypto_quotes(tickers)


@st.cache_data(ttl=30, show_spinner=False)
def cached_candles(ticker):
    return get_crypto_candles(ticker)


def all_quotes():
    quotes = dict(cached_live_quotes(STOCK_TICKERS))
    crypto = cached_crypto_quotes(CRYPTO_TICKERS)
    quotes.update(crypto)

    missing = tuple(
        ticker for ticker in CRYPTO_TICKERS
        if ticker not in crypto
    )

    if missing:
        quotes.update(cached_live_quotes(missing))

    return quotes


# Load daily data on full-app runs, not on each Live fragment refresh.
market = {}
prices = {}
market_errors = []

with st.spinner("Loading daily market data..."):
    for name, ticker in ASSETS.items():
        try:
            df = get_market_data(ticker)

            if df is None or df.empty or "Close" not in df.columns:
                market_errors.append(f"{name}: no market data")
                continue

            valid = df.dropna(subset=["Close"])

            if valid.empty:
                market_errors.append(f"{name}: no valid prices")
                continue

            market[ticker] = df
            prices[ticker] = float(valid["Close"].iloc[-1])

        except Exception as exc:
            market_errors.append(f"{name}: {type(exc).__name__}")


# =========================================================
# FOUR TABS
# =========================================================

live_tab, charts_tab, history_tab, controls_tab = st.tabs([
    "🏠 Live",
    "📈 Charts",
    "🧾 History",
    "⚙️ Controls",
])


# =========================================================
# LIVE TAB
# =========================================================

with live_tab:
    live_on = st.toggle(
        "⚡ Live updates · every 2 seconds",
        value=True,
        key="live_updates_enabled",
    )

    @st.fragment(
        run_every=LIVE_SECONDS if live_on else None
    )
    def live_dashboard():
        state = st.session_state.jarvis_state

        try:
            quotes = all_quotes()
        except Exception:
            st.warning("Live prices unavailable. Retry shortly.")
            return

        live_prices = {
            ticker: quote["price"]
            for ticker, quote in quotes.items()
            if isinstance(quote, dict)
            and quote.get("price") is not None
        }

        cash = float(state.get("cash", STARTING_CASH))
        positions = state.get("positions", {})
        trades = state.get("trades", [])

        equity = portfolio_value(cash, positions, live_prices)
        initial = float(state.get("starting_equity", STARTING_CASH))
        pnl = equity - initial
        percentage = pnl / initial * 100 if initial else 0

        color = "#4ade80" if pnl >= 0 else "#f87171"

        if pnl > 0:
            message = f"Winning +${pnl:,.2f}"
        elif pnl < 0:
            message = f"At a loss −${abs(pnl):,.2f}"
        else:
            message = "Break-even $0.00"

        st.markdown(
            f"""
            <div class="portfolio-card">
                <div class="portfolio-label">PORTFOLIO VALUE</div>
                <div class="portfolio-value">${equity:,.2f}</div>
                <div class="portfolio-pnl" style="color:{color}">
                    {message} ({percentage:+.2f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("Available cash", f"${cash:,.2f}")
        col2.metric("Open positions", len(positions))

        closed = [
            trade for trade in trades
            if trade.get("action") == "SELL"
            and trade.get("pnl") is not None
        ]

        realized = sum(float(trade["pnl"]) for trade in closed)
        unrealized = 0.0

        for ticker, pos in positions.items():
            try:
                if ticker in live_prices:
                    value = (
                        float(pos["units"])
                        * position_price(ticker, pos, live_prices)
                    )
                    unrealized += value - float(pos["cost"])
            except (KeyError, TypeError, ValueError):
                pass

        c1, c2 = st.columns(2)
        c1.metric("Unrealized P/L", f"${unrealized:+,.2f}")
        c2.metric("Realized P/L", f"${realized:+,.2f}")

        st.subheader("Open positions")

        if positions:
            for ticker, pos in positions.items():
                try:
                    if ticker not in live_prices:
                        with st.container(border=True):
                            st.markdown(f"**{ticker}**")
                            st.caption("Live price unavailable")
                        continue

                    value = (
                        float(pos["units"])
                        * position_price(ticker, pos, live_prices)
                    )
                    position_pnl = value - float(pos["cost"])
                    symbol = "🟢" if position_pnl >= 0 else "🔴"
                    name = next(
                        (n for n, t in ASSETS.items() if t == ticker),
                        ticker,
                    )

                    with st.container(border=True):
                        left, right = st.columns([3, 2])

                        with left:
                            st.markdown(f"**{name}**")
                            st.caption(
                                f"{ticker} · Value ${value:,.2f}"
                            )

                        with right:
                            st.markdown(
                                f"### {symbol} {position_pnl:+,.2f}"
                            )
                            st.caption("Unrealized P/L · USD")

                except (KeyError, TypeError, ValueError):
                    st.caption(f"{ticker}: position data unavailable")
        else:
            st.info("No open positions yet.")

        st.subheader("Market movers")

        try:
            movers = movers_table(quotes)

            if movers is not None and not movers.empty:
                st.dataframe(
                    movers,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("Market movers unavailable.")
        except Exception:
            st.caption("Could not calculate market movers.")

        st.caption(
            "Paper trading only · No real orders are placed."
        )

        st.caption(
            "Crypto may update more frequently. Stock/forex quotes "
            "may be delayed or show the last market close."
        )

    live_dashboard()


# =========================================================
# CHARTS TAB — NO AUTOMATIC REFRESH
# =========================================================

with charts_tab:
    st.subheader("Live crypto charts")

    for ticker in CRYPTO_TICKERS:
        name = next(
            (n for n, t in ASSETS.items() if t == ticker),
            ticker,
        )

        with st.expander(f"{name} · {ticker}"):
            try:
                candles = cached_candles(ticker)

                if candles is None or candles.empty:
                    st.info("Chart unavailable right now.")
                else:
                    st.line_chart(candles["Close"])
            except Exception:
                st.warning("Could not load crypto chart.")

    st.divider()
    st.subheader("Daily charts & trends")

    for name, ticker in ASSETS.items():
        if ticker not in market:
            continue

        df = market[ticker].dropna(subset=["Close"])

        if df.empty:
            continue

        last = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else last
        price = float(last["Close"])
        previous_price = float(previous["Close"])

        change = (
            (price / previous_price - 1) * 100
            if previous_price else 0
        )

        trend = "Bullish" if is_bullish(df) else "Bearish / unclear"

        with st.expander(f"{name} · {ticker}"):
            c1, c2 = st.columns(2)

            c1.metric(
                "Latest daily price",
                f"{price:,.4f}",
                f"{change:+.2f}%",
            )

            rsi = last.get("RSI")
            c2.metric(
                "RSI",
                f"{rsi:.1f}" if pd.notna(rsi) else "—",
            )

            st.caption(f"EMA trend: {trend}")

            columns = [
                col for col in ("Close", "EMA20", "EMA50")
                if col in df.columns
            ]

            if columns:
                st.line_chart(df[columns].tail(90))

    for error in market_errors:
        st.caption(error)

    st.caption(
        "Daily charts use historical market data and are not "
        "guaranteed to be real-time."
    )


# =========================================================
# HISTORY TAB
# =========================================================

with history_tab:
    state = st.session_state.jarvis_state

    st.subheader("Performance history")

    initial_value = float(
        state.get("starting_equity", STARTING_CASH)
    )
    snapshots = state.get("equity_snapshots", [])

    if snapshots:
        history = pd.DataFrame(snapshots)

        if {"time", "equity"}.issubset(history.columns):
            history["time"] = pd.to_datetime(
                history["time"], utc=True, errors="coerce"
            )
            history["equity"] = pd.to_numeric(
                history["equity"], errors="coerce"
            )

            history = history.dropna(
                subset=["time", "equity"]
            ).sort_values("time")

            if not history.empty:
                history["day"] = history["time"].dt.strftime("%Y-%m-%d")
                history["month"] = history["time"].dt.strftime("%Y-%m")

                st.line_chart(history.set_index("time")["equity"])

                daily = period_returns(history, initial_value, "day")
                monthly = period_returns(history, initial_value, "month")

                daily_tab, monthly_tab = st.tabs(["Daily", "Monthly"])

                with daily_tab:
                    if not daily.empty:
                        st.dataframe(
                            daily.iloc[::-1].round(2),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.caption(
                            f"Winning days: {(daily['P/L'] > 0).sum()} · "
                            f"Losing days: {(daily['P/L'] < 0).sum()}"
                        )
                    else:
                        st.info("No daily results yet.")

                with monthly_tab:
                    if not monthly.empty:
                        st.dataframe(
                            monthly.iloc[::-1].round(2),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No monthly results yet.")
            else:
                st.info("No valid equity snapshots.")
        else:
            st.info("Saved snapshots lack time/equity fields.")
    else:
        st.info(
            "Performance history will appear when your Worker saves "
            "equity snapshots. Historical daily results cannot be "
            "reconstructed reliably from trade records alone."
        )

    st.divider()
    st.subheader("Trade list")

    trades = state.get("trades", [])

    if trades:
        trade_df = pd.DataFrame(trades)

        if "time" in trade_df.columns:
            trade_df = trade_df.sort_values(
                "time", ascending=False
            )

        st.dataframe(
            trade_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No trades recorded yet.")

    st.caption(
        "Last database update: " + fmt_time(state.get("_db_updated_at"))
    )


# =========================================================
# CONTROLS TAB
# =========================================================

with controls_tab:
    st.subheader("Trading controls")

    st.caption(
        "The background Worker runs independently. "
        "Manual cycles use simulated trades."
    )

    if st.button(
        "▶️ Run paper-trading cycle",
        type="primary",
        use_container_width=True,
        key="run_paper_cycle",
    ):
        try:
            quotes = all_quotes()

            fill_prices = dict(prices)
            fill_prices.update({
                ticker: quote["price"]
                for ticker, quote in quotes.items()
                if isinstance(quote, dict)
                and quote.get("price") is not None
            })

            current_state = st.session_state.jarvis_state

            changes, actions = apply_cycle(
                current_state,
                market,
                fill_prices,
            )

            save_state(changes)

            st.success(
                "Virtual cycle saved. "
                + (
                    " · ".join(actions)
                    if actions else "No trades needed."
                )
            )
            st.rerun()

        except Exception as exc:
            st.error("Could not complete or save the cycle.")
            st.code(f"{type(exc).__name__}: {str(exc)[:300]}")

    st.divider()

    if st.button(
        "🔄 Reload saved portfolio",
        use_container_width=True,
        key="reload_portfolio",
    ):
        try:
            st.cache_data.clear()
            st.session_state.jarvis_state = load_state()
            st.rerun()
        except Exception as exc:
            st.error("Reload failed.")
            st.caption(str(exc)[:300])

    if st.button(
        "🔒 Lock JARVIS",
        use_container_width=True,
        key="lock_jarvis",
    ):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()

    st.caption(
        "Locking the dashboard does not stop the background Worker."
    )
    st.caption(
        "Paper trading only · No real money is used."
    )
