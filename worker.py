
import os
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
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


def now():
    return datetime.now(timezone.utc).isoformat()


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

    result = pd.DataFrame({"Close": close}).dropna()
    result["EMA20"] = result["Close"].ewm(
        span=20, adjust=False
    ).mean()
    result["EMA50"] = result["Close"].ewm(
        span=50, adjust=False
    ).mean()

    return result.dropna()


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

    if response.data:
        state = response.data[0]["data"]
    else:
        state = {
            "cash": STARTING_CASH,
            "positions": {},
            "trades": [],
        }

    state.setdefault("cash", STARTING_CASH)
    state.setdefault("positions", {})
    state.setdefault("trades", [])

    # Avoid repeating the same daily trading cycle.
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("worker_last_day") == today:
        print("Already processed today; no duplicate cycle.")
        return

    prices = {}
    data = {}

    for name, ticker in ASSETS.items():
        try:
            df = get_data(ticker)
            if df is not None and len(df) >= 2:
                data[ticker] = df
                prices[ticker] = float(df["Close"].iloc[-1])
        except Exception as exc:
            print(f"{name}: data error: {exc}")

    if not data:
        raise RuntimeError("No market data retrieved; portfolio not changed.")

    for name, ticker in ASSETS.items():
        if ticker not in data:
            continue

        df = data[ticker]
        price = prices[ticker]
        bullish = float(df["EMA20"].iloc[-1]) > float(
            df["EMA50"].iloc[-1]
        )
        pos = state["positions"].get(ticker)

        if bullish and pos is None:
            equity = float(state["cash"])
            for symbol, p in state["positions"].items():
                equity += p["units"] * prices.get(symbol, p["entry"])

            budget = min(float(state["cash"]), equity * ALLOCATION)
            if budget > 1:
                units = budget * (1 - FEE_RATE) / price
                state["cash"] -= budget
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
                print(f"Virtual BUY {name} at {price}")

        elif not bullish and pos is not None:
            proceeds = pos["units"] * price * (1 - FEE_RATE)
            pnl = proceeds - pos["cost"]
            state["cash"] += proceeds
            state["trades"].append({
                "time": now(),
                "asset": name,
                "action": "SELL",
                "price": price,
                "pnl": round(pnl, 2),
            })
            del state["positions"][ticker]
            print(f"Virtual SELL {name} at {price}; P/L {pnl:.2f}")

    state["worker_last_day"] = today

    db.table("paper_trading").upsert(
        {
            "user_key": USER_KEY,
            "data": state,
            "updated_at": now(),
        },
        on_conflict="user_key",
    ).execute()

    print("Paper portfolio saved successfully.")


if __name__ == "__main__":
    main()

