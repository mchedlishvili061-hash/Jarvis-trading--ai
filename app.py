
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client

st.set_page_config(
    page_title="JARVIS Trading AI",
    page_icon="🤖",
    layout="wide"
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


# ---------- LOGIN ----------
st.title("🤖 JARVIS TRADING AI")
st.caption("Market analysis • Virtual paper trading only")
st.warning("Simulation only. No real money or real orders are used.")

try:
    APP_PASSWORD = st.secrets["JARVIS_APP_PASSWORD"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_SECRET_KEY"]
except Exception:
    st.error(
        "Missing Streamlit Secrets. Check JARVIS_APP_PASSWORD, "
        "SUPABASE_URL and SUPABASE_SECRET_KEY."
    )
    st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter JARVIS password", type="password")

    if st.button("Unlock JARVIS", type="primary"):
        if password and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


# ---------- SUPABASE ----------
@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


try:
    supabase = get_supabase()
except Exception:
    st.error("Could not connect to the database. Check your Supabase Secrets.")
    st.stop()


def empty_state():
    return {
        "cash": STARTING_CASH,
        "positions": {},
        "trades": [],
    }


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

    state = empty_state()
    save_state(state)
    return state


def save_state(state):
    payload = {
        "user_key": USER_KEY,
        "data": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    supabase.table("paper_trading").upsert(
        payload,
        on_conflict="user_key"
    ).execute()


if "db_loaded" not in st.session_state:
    try:
        saved_state = load_state()
        st.session_state.cash = saved_state["cash"]
        st.session_state.positions = saved_state["positions"]
        st.session_state.trades = saved_state["trades"]
        st.session_state.db_loaded = True
    except Exception:
        st.error(
            "Could not load/save paper-trading data. "
            "Check that paper_trading has user_key, data and updated_at columns, "
            "and that user_key has a unique index."
        )
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

    return df.dropna()


def get_price(data):
    return float(data["Close"].iloc[-1])


def portfolio_value(prices):
    value = float(st.session_state.cash)

    for ticker, pos in st.session_state.positions.items():
        value += pos["units"] * prices.get(ticker, pos["entry"])

    return value


# ---------- OVERVIEW ----------
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
            errors.append(f"{name}: no data available")
            continue

        market[ticker] = df
        last = df.iloc[-1]
        previous = df.iloc[-2]

        price = get_price(df)
        prices[ticker] = price

        change = (price / float(previous["Close"]) - 1) * 100
        trend = (
            "Bullish"
            if last["EMA20"] > last["EMA50"]
            else "Bearish"
        )

        with st.expander(f"{name} ({ticker})", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{price:,.4f}", f"{change:+.2f}%")
            c2.metric("RSI", f"{float(last['RSI']):.1f}")
            c3.metric("30-day return", f"{float(last['Return30']):+.2f}%")
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
    "Strategy: buy when EMA20 is above EMA50; "
    "sell when it is below. One position per asset."
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
                equity * ALLOCATION
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
                    "opened": datetime.now(
                        timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC"),
                }

                st.session_state.trades.append({
                    "time": datetime.now(
                        timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC"),
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
                "time": datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC"),
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
        st.success("Paper-trading cycle completed and saved.")
    except Exception:
        st.error("Cycle ran, but saving to Supabase failed.")

    st.rerun()


# ---------- PORTFOLIO ----------
positions = st.session_state.positions

open_value = sum(
    p["units"] * prices.get(ticker, p["entry"])
    for ticker, p in positions.items()
)

equity = st.session_state.cash + open_value
closed = [
    t for t in st.session_state.trades
    if t["action"] == "SELL" and t["pnl"] is not None
]

wins = sum(1 for t in closed if t["pnl"] > 0)
losses = sum(1 for t in closed if t["pnl"] <= 0)
win_rate = wins / len(closed) * 100 if closed else 0
pnl_total = equity - STARTING_CASH

a, b, c, d = st.columns(4)
a.metric("Portfolio value", f"${equity:.2f}", f"{pnl_total:+.2f}")
b.metric("Available cash", f"${st.session_state.cash:.2f}")
c.metric("Total return", f"{pnl_total / STARTING_CASH * 100:+.2f}%")
d.metric(
    "Closed-trade win rate",
    f"{win_rate:.1f}%",
    f"{wins} wins / {losses} losses"
)

st.subheader("📌 Open positions")

if positions:
    rows = []

    for ticker, p in positions.items():
        price = prices.get(ticker, p["entry"])
        value = p["units"] * price
        pnl = value - p["cost"]

        rows.append({
            "Asset": p["name"],
            "Entry price": round(p["entry"], 4),
            "Latest price": round(price, 4),
            "Position value": round(value, 2),
            "Unrealized P/L": round(pnl, 2),
            "P/L %": round(pnl / p["cost"] * 100, 2),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No open positions. Run a paper-trading cycle to check signals.")

st.subheader("🧾 Trade history")

if st.session_state.trades:
    st.dataframe(
        pd.DataFrame(st.session_state.trades).iloc[::-1],
        use_container_width=True,
    )
else:
    st.info("No virtual trades yet.")

st.caption(
    "Signals use daily historical candles and are not guaranteed forecasts. "
    "Paper-trading state is stored in Supabase."
)

if st.button("🔒 Lock JARVIS"):
    st.session_state.authenticated = False
    st.rerun()
