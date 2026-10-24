
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="JARVIS Trading AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 JARVIS TRADING AI")
st.caption("Market monitor | Paper trading | Virtual balance")

st.info("Historical market analysis and market monitoring. No real trades are executed.")

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
        ticker,
        period="6mo",
        interval="1d",
        progress=False,
        auto_adjust=True
    )
    if data.empty:
        return data

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    data = data.copy()
    data["Close"] = close
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["Return30"] = close.pct_change(30) * 100
    return data.dropna()

st.subheader("📊 Market overview")
st.write("Last refresh:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if st.button("🔄 Refresh market data"):
    st.cache_data.clear()

for name, ticker in assets.items():
    with st.expander(f"{name} — {ticker}", expanded=True):
        try:
            data = get_market_data(ticker)

            if data.empty or len(data) < 2:
                st.warning("Market data unavailable. Try again later.")
                continue

            last = data.iloc[-1]
            previous = data.iloc[-2]
            price = float(last["Close"])
            change = (price / float(previous["Close"]) - 1) * 100
            rsi = float(last["RSI"])
            ema20 = float(last["EMA20"])
            ema50 = float(last["EMA50"])
            return30 = float(last["Return30"])

            a, b, c, d = st.columns(4)
            a.metric("Latest price", f"{price:,.4f}", f"{change:+.2f}%")
            b.metric("RSI (14)", f"{rsi:.1f}")
            c.metric("30-day return", f"{return30:+.2f}%")
            d.metric("EMA trend", "Bullish" if ema20 > ema50 else "Bearish")

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
st.metric("Starting virtual balance", "$100.00")
st.write(
    "Virtual trading and performance tracking will be added in the next stage. "
    "This version only analyzes markets and does not buy or sell."
)

st.caption("JARVIS AI • For educational use. Signals are not financial advice.")
