
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="JARVIS Trading AI", page_icon="🤖", layout="wide")

st.title("🤖 JARVIS TRADING AI")
st.caption("Market monitor | Paper trading")
st.info("Analysis only — no real trades are executed.")

assets = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "NVIDIA": "NVDA",
    "S&P 500 ETF": "SPY",
    "EUR/USD": "EURUSD=X"
}

@st.cache_data(ttl=300)
def get_market_data(ticker):
    data = yf.download(
        ticker, period="6mo", interval="1d",
        progress=False, auto_adjust=True
    )

    if data.empty:
        return pd.DataFrame()

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = pd.DataFrame(index=data.index)
    result["Close"] = pd.to_numeric(close, errors="coerce")
    result["EMA20"] = result["Close"].ewm(span=20, adjust=False).mean()
    result["EMA50"] = result["Close"].ewm(span=50, adjust=False).mean()

    delta = result["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)

    result["RSI"] = 100 - (100 / (1 + rs))
    result["Return30"] = result["Close"].pct_change(30) * 100

    return result.dropna()

def number(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.iloc[0]
    return float(value)

st.subheader("📊 Market overview")
st.write("Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if st.button("🔄 Refresh market data"):
    st.cache_data.clear()
    st.rerun()

for name, ticker in assets.items():
    with st.expander(f"{name} ({ticker})", expanded=True):
        try:
            data = get_market_data(ticker)

            if data.empty or len(data) < 2:
                st.warning(f"{name}: No market data available right now.")
                continue

            last = data.iloc[-1]
            previous = data.iloc[-2]

            price = number(last["Close"])
            previous_price = number(previous["Close"])
            change = (price / previous_price - 1) * 100
            rsi = number(last["RSI"])
            ema20 = number(last["EMA20"])
            ema50 = number(last["EMA50"])
            return30 = number(last["Return30"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{price:,.4f}", f"{change:+.2f}%")
            c2.metric("RSI", f"{rsi:.1f}")
            c3.metric("30-day return", f"{return30:+.2f}%")
            c4.metric("Trend", "Bullish" if ema20 > ema50 else "Bearish")

            if ema20 > ema50 and rsi < 70:
                st.success("Potential bullish setup — review")
            elif ema20 < ema50 and rsi > 30:
                st.warning("Potential bearish trend — review")
            else:
                st.info("Mixed signal — wait for confirmation")

            st.line_chart(data[["Close", "EMA20", "EMA50"]])

        except Exception as e:
            st.error(f"{name}: {type(e).__name__}: {e}")

st.divider()
st.subheader("💰 Paper trading")
st.metric("Virtual starting balance", "$100.00")
st.write("Automatic virtual trading and win/loss statistics are not enabled in this version.")
st.caption("Educational use only. Signals are not financial advice.")
