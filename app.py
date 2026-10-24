
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="JARVIS Trading AI", page_icon="🤖", layout="wide")

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

st.title("🤖 JARVIS TRADING AI")
st.caption("Market analysis • Virtual paper trading only")
st.warning("Simulation only. No real money or real orders are used.")

@st.cache_data(ttl=300)
def get_market_data(ticker):
    raw = yf.download(
        ticker, period="6mo", interval="1d",
        progress=False, auto_adjust=True
    )
    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Return30"] = df["Close"].pct_change(30) * 100
    return df.dropna()

# Temporary session storage: resets if the app session/server resets.
if "cash" not in st.session_state:
    st.session_state.cash = STARTING_CASH
    st.session_state.positions = {}
    st.session_state.trades = []

def get_price(data):
    return float(data["Close"].iloc[-1])

def portfolio_value(prices):
    value = float(st.session_state.cash)
    for ticker, pos in st.session_state.positions.items():
        if ticker in prices:
            value += pos["units"] * prices[ticker]
    return value

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
        if df.empty:
            errors.append(f"{name}: no data available")
            continue

        market[ticker] = df
        last = df.iloc[-1]
        previous = df.iloc[-2]
        price = get_price(df)
        prices[ticker] = price
        change = (price / float(previous["Close"]) - 1) * 100
        trend = "Bullish" if last["EMA20"] > last["EMA50"] else "Bearish"

        with st.expander(f"{name} ({ticker})", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{price:,.4f}", f"{change:+.2f}%")
            c2.metric("RSI", f"{float(last['RSI']):.1f}")
            c3.metric("30-day return", f"{float(last['Return30']):+.2f}%")
            c4.metric("EMA trend", trend)
            st.line_chart(df[["Close", "EMA20", "EMA50"]])

    except Exception as e:
        errors.append(f"{name}: {type(e).__name__}: {e}")

for error in errors:
    st.warning(error)

st.divider()
st.subheader("💰 Virtual paper trading")
st.caption("Strategy: buy when EMA20 is above EMA50; sell when it is below. One position per asset.")

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
                spend = budget - fee
                units = spend / price
                st.session_state.cash -= budget
                st.session_state.positions[ticker] = {
                    "name": name,
                    "entry": price,
                    "units": units,
                    "cost": budget,
                    "opened": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.trades.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "asset": name,
                "action": "SELL",
                "price": price,
                "pnl": round(pnl, 2),
            })
            del st.session_state.positions[ticker]

    st.success("Paper-trading cycle completed.")
    st.rerun()

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
d.metric("Closed-trade win rate", f"{win_rate:.1f}%", f"{wins} wins / {losses} losses")

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
        use_container_width=True
    )
else:
    st.info("No virtual trades yet.")

st.caption(
    "Signals use daily historical candles and are not guaranteed forecasts. "
    "This demo stores state in the current session; it may reset when the session or server restarts."
)
