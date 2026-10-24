
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from supabase import create_client

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
MAX_SNAPSHOTS = 400


def now():
    return datetime.now(timezone.utc).isoformat()


def today_utc():
    return datetime.now(timezone.utc).date().isoformat()


def get_data(ticker):
    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return None

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    }).dropna()

    result["EMA20"] = result["Close"].ewm(
        span=20, adjust=False
    ).mean()
    result["EMA50"] = result["Close"].ewm(
        span=50, adjust=False
    ).mean()

    result = result.dropna()
    return result if len(result) >= 2 else None


def portfolio_value(state, prices):
    value = float(state["cash"])

    for ticker, pos in state["positions"].items():
        price = prices.get(ticker, float(pos["entry"]))
        value += float(pos["units"]) * price

    return value


def main():
    url = os.environ["SUPABASE_URL"].strip().rstrip("/")
    key = os.environ["SUPABASE_SECRET_KEY"].strip()
    db = create_client(url, key)

    response = (
        db.table("paper_trading")
        .select("data")
        .eq("user_key", USER_KEY)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Saved portfolio not found. Refusing to create a new portfolio."
        )

    state = response.data[0].get("data")

    if not isinstance(state, dict):
        raise RuntimeError("Saved portfolio data is invalid.")

    state = dict(state)
    state.setdefault("cash", STARTING_CASH)
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    state.setdefault("equity_snapshots", [])

    today = today_utc()

    if state.get("worker_last_day") == today:
        print("Already processed today; skipping duplicate cycle.")
        return

    prices = {}
    data = {}

    for name, ticker in ASSETS.items():
        try:
            df = get_data(ticker)
            if df is not None:
                data[ticker] = df
                prices[ticker] = float(df["Close"].iloc[-1])
        except Exception as exc:
            print(f"{name}: data error: {type(exc).__name__}: {exc}")

    if not data:
        raise RuntimeError(
            "No market data retrieved; portfolio was not changed."
        )

    # Execute the existing virtual EMA strategy.
    for name, ticker in ASSETS.items():
        if ticker not in data:
            continue

        df = data[ticker]
        price = prices[ticker]
        last = df.iloc[-1]
        bullish = float(last["EMA20"]) > float(last["EMA50"])
        pos = state["positions"].get(ticker)

        if bullish and pos is None:
            equity = portfolio_value(state, prices)
            budget = min(float(state["cash"]), equity * ALLOCATION)

            if budget > 1:
                units = budget * (1 - FEE_RATE) / price
                state["cash"] = float(state["cash"]) - budget
                state["positions"][ticker] = {
                    "name": name,
                    "entry": price,
                    "units": units,
                    "cost": budget,
                    "opened": now(),
                }
                state["trades"].append({
                    "time": now(),
                    "asset": name,
                    "action": "BUY",
                    "price": price,
                    "pnl": None,
                })
                print(f"Virtual BUY {name} at {price:.4f}")

        elif not bullish and pos is not None:
            proceeds = (
                float(pos["units"]) * price * (1 - FEE_RATE)
            )
            pnl = proceeds - float(pos["cost"])
            state["cash"] = float(state["cash"]) + proceeds

            state["trades"].append({
                "time": now(),
                "asset": name,
                "action": "SELL",
                "price": price,
                "pnl": round(pnl, 2),
            })

            del state["positions"][ticker]
            print(f"Virtual SELL {name} at {price:.4f}; P/L {pnl:.2f}")

    # Save one daily portfolio snapshot for performance reporting.
    equity = portfolio_value(state, prices)
    snapshots = state.get("equity_snapshots", [])
    if not isinstance(snapshots, list):
        snapshots = []

    snapshots = [
        item for item in snapshots
        if isinstance(item, dict) and item.get("day") != today
    ]
    snapshots.append({
        "day": today,
        "time": now(),
        "equity": round(equity, 4),
        "cash": round(float(state["cash"]), 4),
    })

    state["equity_snapshots"] = snapshots[-MAX_SNAPSHOTS:]
    state["worker_last_day"] = today

    db.table("paper_trading").upsert(
        {
            "user_key": USER_KEY,
            "data": state,
            "updated_at": now(),
        },
        on_conflict="user_key",
    ).execute()

    print(f"Portfolio saved. Equity: ${equity:.2f}")
    print(f"Daily snapshot saved for {today}.")


if __name__ == "__main__":
    main()
