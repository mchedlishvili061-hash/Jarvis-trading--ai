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
    positions and trades. The input state is not modified.
    """
    cash = float(state.get("cash", STARTING_CASH))
    positions = {t: dict(p) for t, p in state.get("positions", {}).items()}
    trades = list(state.get("trades", []))
    actions = []

    for name, ticker in ASSETS.items():
        df = data.get(ticker)
        price = prices.get(ticker)
        if df is None or price is None:
            continue

        bullish = is_bullish(df)
        if bullish is None:
            continue

        pos = positions.get(ticker)
        if pos is not None:
            pos["last_price"] = price

        if bullish and pos is None:
            equity = portfolio_value(cash, positions, prices)
            budget = min(cash, equity * ALLOCATION)
            if budget > 1:
                units = budget * (1 - FEE_RATE) / price
                cash -= budget
                positions[ticker] = {
                    "name": name,
                    "entry": price,
                    "units": units,
                    "cost": budget,
                    "opened": utc_now(),
                    "last_price": price,
                }
                trades.append({
                    "time": utc_now(),
                    "asset": name,
                    "action": "BUY",
                    "price": price,
                    "pnl": None,
                })
                actions.append(f"Virtual BUY {name} at {price:.4f}")

        elif not bullish and pos is not None:
            proceeds = float(pos["units"]) * price * (1 - FEE_RATE)
            pnl = proceeds - float(pos["cost"])
            cash += proceeds
            trades.append({
                "time": utc_now(),
                "asset": name,
                "action": "SELL",
                "price": price,
                "pnl": round(pnl, 2),
            })
            del positions[ticker]
            actions.append(
                f"Virtual SELL {name} at {price:.4f}; P/L {pnl:.2f}"
            )

    return {"cash": cash, "positions": positions, "trades": trades}, actions


# ---------- REPORTING ----------

def period_returns(history, initial_value, key):
    """Profit/loss per period from end-of-period equity.

    `history` needs an `equity` column plus the period column named by
    `key` (for example "day" or "month"). Each period is compared with the
    END of the previous period; the first period is compared with
    `initial_value`.
    """
    history = history.sort_values("time")
    ends = history.groupby(key, sort=True)["equity"].last()
    starts = ends.shift(1)
    starts.iloc[0] = float(initial_value)

    out = pd.DataFrame({
        key: ends.index,
        "start": starts.values,
        "end": ends.values,
    })
    out["P/L"] = out["end"] - out["start"]
    out["P/L %"] = np.where(
        out["start"] != 0,
        out["P/L"] / out["start"] * 100,
        0,
    )
    return out


# ---------- LIVE QUOTES ----------

def _fetch_quote(ticker):
    """Latest price and previous close for one ticker, or None."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = float(info["last_price"])
        prev = float(info["previous_close"])
        if price > 0:
            return {"price": price, "prev_close": prev if prev > 0 else price}
    except Exception:
        pass

    # Fallback: last two daily closes.
    try:
        closes = (
            yf.Ticker(ticker)
            .history(period="5d", interval="1d")["Close"]
            .dropna()
        )
        if len(closes) >= 1:
            price = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) > 1 else price
            return {"price": price, "prev_close": prev}
    except Exception:
        pass
    return None


def get_live_quotes(tickers):
    """Fetch quotes for several tickers in parallel.

    Returns {ticker: {"price", "prev_close", "change_pct"}}. Tickers whose
    quote could not be fetched are left out.
    """
    tickers = list(tickers)
    if not tickers:
        return {}

    with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
        results = list(pool.map(_fetch_quote, tickers))

    quotes = {}
    for ticker, quote in zip(tickers, results):
        if quote is None:
            continue
        prev = quote["prev_close"]
        quote["change_pct"] = (quote["price"] / prev - 1) * 100 if prev else 0.0
        quotes[ticker] = quote
    return quotes


# ---------- LIVE TABLES ----------

def move_label(pct):
    if pct > 0:
        return f"🟢 ▲ {pct:+.2f}%"
    if pct < 0:
        return f"🔴 ▼ {pct:+.2f}%"
    return f"⚪ {pct:+.2f}%"


def movers_table(quotes):
    """Assets sorted from biggest gainer to biggest loser."""
    rows = []
    for name, ticker in ASSETS.items():
        quote = quotes.get(ticker)
        if quote is None:
            continue
        rows.append({
            "Asset": name,
            "Price": round(quote["price"], 4),
            "Change": quote["change_pct"],
        })
    if not rows:
        return pd.DataFrame(columns=["Asset", "Price", "Move"])

    df = pd.DataFrame(rows).sort_values("Change", ascending=False)
    df["Move"] = df["Change"].map(move_label)
    return df[["Asset", "Price", "Move"]]


def positions_table(positions, prices):
    """Open positions with live P/L, best first."""
    rows = []
    for ticker, pos in positions.items():
        current = position_price(ticker, pos, prices)
        value = float(pos["units"]) * current
        cost = float(pos["cost"])
        pnl = value - cost
        pct = pnl / cost * 100 if cost else 0.0
        rows.append({
            "Asset": pos.get("name", ticker),
            "Entry": round(float(pos["entry"]), 4),
            "Live": round(current, 4),
            "Value": round(value, 2),
            "P/L $": round(pnl, 2),
            "P/L": move_label(pct),
            "_pnl": pnl,
        })
    if not rows:
        return pd.DataFrame(
            columns=["Asset", "Entry", "Live", "Value", "P/L $", "P/L"]
        )

    df = pd.DataFrame(rows).sort_values("_pnl", ascending=False)
    return df.drop(columns="_pnl")
