"""Shared data, strategy and reporting helpers for JARVIS.

Used by both app.py (Streamlit dashboard) and worker.py (daily GitHub
Actions job) so the trading rules live in exactly one place.

Paper trading only - no real orders are ever placed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ---------- MARKET DATA ----------

def get_data(ticker):
    """Download 6 months of daily data and add indicators.

    Returns None when no usable data came back.
    """
    raw = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return None

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    }).dropna()
    if df.empty:
        return None

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return30"] = df["Close"].pct_change(30) * 100
    return df.replace([np.inf, -np.inf], np.nan)


# ---------- SIGNAL ----------

def completed_candles(df):
    """Drop today's (UTC) candle: it is still moving and not final."""
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return df[index < today]


def is_bullish(df):
    """EMA20 > EMA50 on the last COMPLETED daily candle.

    Returns True / False, or None if there is not enough data.
    """
    done = completed_candles(df).dropna(subset=["EMA20", "EMA50"])
    if done.empty:
        return None
    last = done.iloc[-1]
    return bool(last["EMA20"] > last["EMA50"])


# ---------- PORTFOLIO ----------

def position_price(ticker, pos, prices):
    """Fresh price if we have one, else last known price, else entry."""
    if ticker in prices:
        return float(prices[ticker])
    return float(pos.get("last_price", pos["entry"]))


def portfolio_value(cash, positions, prices):
    total = float(cash)
    for ticker, pos in positions.items():
        total += float(pos["units"]) * position_price(ticker, pos, prices)
    return total


def stale_tickers(positions, prices):
    """Open positions that have no fresh price right now."""
    return [t for t in positions if t not in prices]


def apply_cycle(state, data, prices):
    """Run one virtual trading cycle.

    Signals come from the last completed candle; fills use the latest price.
    Returns (changes, actions) where `changes` holds the new cash,
    positions and trades. The input state is
