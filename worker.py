import os
from datetime import datetime, timezone

from supabase import create_client

from strategy import (
    ASSETS,
    STARTING_CASH,
    USER_KEY,
    apply_cycle,
    get_data,
    portfolio_value,
    stale_tickers,
    utc_now,
)

MAX_SNAPSHOTS = 400


def today_utc():
    return datetime.now(timezone.utc).date().isoformat()


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

    # Shared virtual EMA strategy (signal = last completed daily candle).
    changes, actions = apply_cycle(state, data, prices)
    state.update(changes)
    for line in actions:
        print(line)

    # Daily snapshot. Positions with no fresh price today are valued at
    # their last known price and flagged as stale instead of silently
    # being valued at their entry price.
    equity = portfolio_value(state["cash"], state["positions"], prices)
    stale = stale_tickers(state["positions"], prices)
    if stale:
        print(f"Warning: no fresh price for {', '.join(stale)}; "
              "using last known price in today's snapshot.")

    snapshots = state.get("equity_snapshots", [])
    if not isinstance(snapshots, list):
        snapshots = []

    snapshots = [
        item for item in snapshots
        if isinstance(item, dict) and item.get("day") != today
    ]
    snapshot = {
        "day": today,
        "time": utc_now(),
        "equity": round(equity, 4),
        "cash": round(float(state["cash"]), 4),
    }
    if stale:
        snapshot["stale"] = stale
    snapshots.append(snapshot)

    state["equity_snapshots"] = snapshots[-MAX_SNAPSHOTS:]
    state["worker_last_day"] = today

    db.table("paper_trading").upsert(
        {
            "user_key": USER_KEY,
            "data": state,
            "updated_at": utc_now(),
        },
        on_conflict="user_key",
    ).execute()

    print(f"Portfolio saved. Equity: ${equity:.2f}")
    print(f"Daily snapshot saved for {today}.")


if __name__ == "__main__":
    main()
