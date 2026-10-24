
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client

st.set_page_config(
    page_title="JARVIS Trading AI",
    page_icon="🤖",
    layout="wide",
)

ASSETS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "NVIDIA": "NVDA",
    "S&P 500 ETF": "SPY",
    "EUR/USD": "EURUSD=X",
}

STARTING_CASH = 100.0
ALLOCATION = 0.20
FEE_RATE = 0.001
USER_KEY = "jarvis-main"


# ---------- TITLE ----------
st.title("🤖 JARVIS TRADING AI")
st.caption("Market analysis • Virtual paper trading only")
st.warning("Simulation only. No real money or real orders are used.")


# ---------- SECRETS ----------
required_secrets = [
    "JARVIS_APP_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
]

missing = [
    key for key in required_secrets
    if key not in st.secrets or not str(st.secrets[key]).strip()
]

if missing:
    st.error("Missing or empty Streamlit Secrets:")
    for key in missing:
        st.code(key)
    st.info(
        "Open Streamlit Cloud → your app → Settings → Secrets. "
        "Add the missing value, save, then reboot the app."
    )
    st.stop()

APP_PASSWORD = str(st.secrets["JARVIS_APP_PASSWORD"])
SUPABASE_URL = str(st.secrets["SUPABASE_URL"]).strip().rstrip("/")
SUPABASE_SECRET_KEY = str(st.secrets["SUPABASE_SECRET_KEY"]).strip()

if not SUPABASE_URL.startswith("https://"):
    st.error("SUPABASE_URL must be your actual HTTPS Supabase Project URL.")
    st.stop()


# ---------- LOGIN ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.form("jarvis_login"):
        password = st.text_input(
            "JARVIS password",
            type="password",
        )
        submitted = st.form_submit_button(
            "Unlock JARVIS",
            type="primary",
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
    supabase = get_supabase(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY,
    )
except Exception:
    st.error(
        "Could not initialize Supabase. Check the Project URL "
        "and Secret Key in Streamlit Secrets."
    )
    st.stop()


def empty_state():
    return {
        "cash": STARTING_CASH,
        "positions": {},
        "trades": [],
    }


def save_state(state):
    payload = {
        "user_key": USER_KEY,
        "data": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    result = (
        supabase.table("paper_trading")
        .upsert(payload, on_conflict="user_key")
        .execute()
    )

    return result


def load_state():
    result = (
        supabase.table("paper_trading")
        .select("data")
        .eq("user_key", USER_KEY)
        .limit(1)
        .execute()
    )

    if result.data:
        saved = result.data[0].get("data")

        if isinstance(saved, dict):
            return {
                "cash": float(saved.get("cash", STARTING_CASH)),
                "positions": saved.get("positions", {}),
                "trades": saved.get("trades", []),
            }

    # No saved row yet: create the initial virtual portfolio.
    state = empty_state()
    save_state(state)
    return state


if "db_loaded" not in st.session_state:
    try:
        state = load_state()
        st.session_state.cash = state["cash"]
        st.session_state.positions = state["positions"]
        st.session_state.trades = state["trades"]
        st.session_state.db_loaded = True

    except Exception as exc:
        st.error("Could not load or save the JARVIS portfolio.")
        st.caption(
            "Check that public.paper_trading contains user_key (text), "
            "data (jsonb), and updated_at (timestamptz), and that "
            "user_key has a unique index."
        )
        st.code(f"{type(exc).__name__}: {str(exc)[:500]}")
        st.stop()


# ---------- MARKET DATA ----------
@st.cache_data(ttl=300)
def get_market_data(ticker):
    raw = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    })

    df["EMA20"] = df["Close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50, adjust=False
    ).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))
    df["Return30"] = df["Close"].pct_change(30) * 100

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def portfolio_value(prices):
    value = float(st.session_state.cash)

    for ticker, pos in st.session_state.positions.items():
        value += (
            pos["units"]
            * prices.get(ticker, pos["entry"])
        )

    return value


def utc_now():
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


# ---------- MARKET OVERVIEW ----------
st.subheader("📊 Market overview")

if st.button("🔄 Refresh market data"):
    st.cache_data.clear()
    st.rerun()

market = {}
prices = {}
errors = []

for name, ticker in ASSETS.items():
    try:
        df = get_market_data(ticker)

        if df.empty or len(df) < 2:
            errors.append(f"{name}: market data unavailable")
            continue

        market[ticker] = df
        last = df.iloc[-1]
        previous = df.iloc[-2]
        price = float(last["Close"])
        prices[ticker] = price

        change = (
            price / float(previous["Close"]) - 1
        ) * 100

        trend = (
            "Bullish"
            if last["EMA20"] > last["EMA50"]
            else "Bearish"
        )

        with st.expander(f"{name} ({ticker})", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{price:,.4f}", f"{change:+.2f}%")
            c2.metric("RSI", f"{float(last['RSI']):.1f}")
            c3.metric(
                "30-day return",
                f"{float(last['Return30']):+.2f}%",
            )
            c4.metric("EMA trend", trend)
            st.line_chart(df[["Close", "EMA20", "EMA50"]])

    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}")

for error in errors:
    st.warning(error)


# ---------- PAPER TRADING ----------
st.divider()
st.subheader("💰 Virtual paper trading")
st.caption(
    "Daily EMA20/EMA50 strategy. One position per asset. "
    "Signals are not guaranteed forecasts."
)

if st.button("▶️ Run paper-trading cycle", type="primary"):
    for name, ticker in ASSETS.items():
        if ticker not in market:
            continue

        df = market[ticker]
        last = df.iloc[-1]
        price = prices[ticker]

        bullish = float(last["EMA20"]) > float(last["EMA50"])
        pos = st.session_state.positions.get(ticker)

        if bullish and pos is None:
            equity = portfolio_value(prices)
            budget = min(
                st.session_state.cash,
                equity * ALLOCATION,
            )

            if budget > 1:
                fee = budget * FEE_RATE
                units = (budget - fee) / price

                st.session_state.cash -= budget
                st.session_state.positions[ticker] = {
                    "name": name,
                    "entry": price,
                    "units": units,
                    "cost": budget,
                    "opened": utc_now(),
                }

                st.session_state.trades.append({
                    "time": utc_now(),
                    "asset": name,
                    "action": "BUY",
                    "price": price,
                    "pnl": None,
                })

        elif not bullish and pos is not None:
            proceeds = pos["units"] * price * (1 - FEE_RATE)
            pnl = proceeds - pos["cost"]
            st.session_state.cash += proceeds

            st.session_state.trades.append({
                "time": utc_now(),
                "asset": name,
                "action": "SELL",
                "price": price,
                "pnl": round(pnl, 2),
            })

            del st.session_state.positions[ticker]

    try:
        save_state({
            "cash": st.session_state.cash,
            "positions": st.session_state.positions,
            "trades": st.session_state.trades,
        })
        st.success("Paper-trading cycle saved to Supabase.")

    except Exception as exc:
        st.error("Could not save the portfolio to Supabase.")
        st.code(f"{type(exc).__name__}: {str(exc)[:500]}")

    st.rerun()


# ---------- PORTFOLIO SUMMARY ----------
positions = st.session_state.positions

open_value = sum(
    p["units"] * prices.get(ticker, p["entry"])
    for ticker, p in positions.items()
)

equity = st.session_state.cash + open_value
pnl_total = equity - STARTING_CASH

closed = [
    t for t in st.session_state.trades
    if t["action"] == "SELL" and t["pnl"] is not None
]

wins = sum(1 for t in closed if t["pnl"] > 0)
losses = sum(1 for t in closed if t["pnl"] <= 0)
win_rate = wins / len(closed) * 100 if closed else 0

a, b, c, d = st.columns(4)
a.metric("Portfolio value", f"${equity:.2f}", f"{pnl_total:+.2f}")
b.metric("Available cash", f"${st.session_state.cash:.2f}")
c.metric(
    "Total return",
    f"{pnl_total / STARTING_CASH * 100:+.2f}%",
)
d.metric(
    "Closed-trade win rate",
    f"{win_rate:.1f}%",
    f"{wins} wins / {losses} losses",
)


# ---------- OPEN POSITIONS ----------
st.subheader("📌 Open positions")

if positions:
    rows = []

    for ticker, pos in positions.items():
        price = prices.get(ticker, pos["entry"])
        value = pos["units"] * price
        pnl = value - pos["cost"]

        rows.append({
            "Asset": pos["name"],
            "Entry price": round(pos["entry"], 4),
            "Latest price": round(price, 4),
            "Position value": round(value, 2),
            "Unrealized P/L": round(pnl, 2),
            "P/L %": round(pnl / pos["cost"] * 100, 2),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No open positions. Run a paper-trading cycle to check signals.")


# ---------- TRADE HISTORY ----------
st.subheader("🧾 Trade history")

if st.session_state.trades:
    st.dataframe(
        pd.DataFrame(st.session_state.trades).iloc[::-1],
        use_container_width=True,
    )
else:
    st.info("No virtual trades yet.")


# ---------- MANUAL SAVE ----------
if st.button("💾 Save portfolio now"):
    try:
        save_state({
            "cash": st.session_state.cash,
            "positions": st.session_state.positions,
            "trades": st.session_state.trades,
        })
        st.success("Portfolio saved.")
    except Exception as exc:
        st.error("Save failed.")
        st.code(f"{type(exc).__name__}: {str(exc)[:500]}")


# ---------- LOGOUT ----------
if st.button("🔒 Lock JARVIS"):
    st.session_state.authenticated = False
    st.rerun()

st.caption(
    "Virtual trading only. Historical daily data may be delayed. "
    "Supabase stores the paper portfolio between sessions; "
    "this app does not place real trades."
)
