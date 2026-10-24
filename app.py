
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
    initial_sidebar_state="collapsed",
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fmt_time(value):
    if not value:
        return "Not recorded"
    try:
        return pd.to_datetime(value, utc=True).strftime(
            "%d %b %Y, %H:%M UTC"
        )
    except Exception:
        return str(value)


st.title("🤖 JARVIS")
st.caption("Trading dashboard · Virtual money only")
st.warning("Paper trading only. No real orders are placed.")

# ---------- SECRETS ----------
required = [
    "JARVIS_APP_PASSWORD",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
]
missing = [
    key for key in required
    if key not in st.secrets or not str(st.secrets[key]).strip()
]
if missing:
    st.error("Missing Streamlit Secrets: " + ", ".join(missing))
    st.stop()

APP_PASSWORD = str(st.secrets["JARVIS_APP_PASSWORD"])
SUPABASE_URL = str(st.secrets["SUPABASE_URL"]).strip().rstrip("/")
SUPABASE_SECRET_KEY = str(st.secrets["SUPABASE_SECRET_KEY"]).strip()

if not SUPABASE_URL.startswith("https://"):
    st.error("Check SUPABASE_URL in Streamlit Secrets.")
    st.stop()


# ---------- LOGIN ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔐 Unlock JARVIS")
    with st.form("jarvis_login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Unlock",
            type="primary",
            use_container_width=True,
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
    db = get_supabase(SUPABASE_URL, SUPABASE_SECRET_KEY)
except Exception as exc:
    st.error("Could not connect to Supabase.")
    st.caption(str(exc)[:300])
    st.stop()


def load_state():
    result = (
        db.table("paper_trading")
        .select("data,updated_at")
        .eq("user_key", USER_KEY)
        .limit(1)
        .execute()
    )
    if not result.data:
        st.error(
            "No saved portfolio was found. To avoid accidentally "
            "creating a fresh portfolio, this app will not initialize one."
        )
        st.stop()

    saved = result.data[0].get("data")
    if not isinstance(saved, dict):
        st.error("Saved portfolio data is not valid JSON.")
        st.stop()

    state = dict(saved)
    state.setdefault("cash", STARTING_CASH)
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    state.setdefault("equity_snapshots", [])
    state["_db_updated_at"] = result.data[0].get("updated_at")
    return state


if "jarvis_state" not in st.session_state:
    try:
        st.session_state.jarvis_state = load_state()
    except Exception as exc:
        st.error("Could not load the saved portfolio.")
        st.code(f"{type(exc).__name__}: {str(exc)[:300]}")
        st.stop()


def refresh_saved_state():
    try:
        st.session_state.jarvis_state = load_state()
        st.session_state.last_refresh = utc_now()
        st.rerun()
    except Exception as exc:
        st.error("Refresh failed.")
        st.caption(str(exc)[:300])


state = st.session_state.jarvis_state


def save_state(new_state):
    # Read the newest saved JSON first so unknown fields and worker
    # metadata are not discarded by the app.
    result = (
        db.table("paper_trading")
        .select("data")
        .eq("user_key", USER_KEY)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise RuntimeError("Saved portfolio row is missing.")

    latest = result.data[0].get("data")
    if not isinstance(latest, dict):
        raise RuntimeError("Saved portfolio data is invalid.")

    # Merge the app's portfolio fields into the saved state.
    merged = dict(latest)
    for key in ("cash", "positions", "trades"):
        if key in new_state:
            merged[key] = new_state[key]

    db.table("paper_trading").upsert(
        {
            "user_key": USER_KEY,
            "data": merged,
            "updated_at": utc_now(),
        },
        on_conflict="user_key",
    ).execute()

    st.session_state.jarvis_state = merged


# ---------- MARKET DATA ----------
@st.cache_data(ttl=300, show_spinner=False)
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
    }).dropna()

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

    return df.replace([np.inf, -np.inf], np.nan)


def get_portfolio_value(cash, positions, prices):
    total = float(cash)
    for ticker, pos in positions.items():
        total += float(pos["units"]) * prices.get(
            ticker, float(pos["entry"])
        )
    return total


st.divider()
head_left, head_right = st.columns([2, 1])
with head_left:
    st.subheader("📊 Market watch")
with head_right:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state.jarvis_state = load_state()
        st.session_state.last_refresh = utc_now()
        st.rerun()

market = {}
prices = {}
errors = []

with st.spinner("Loading market data..."):
    for name, ticker in ASSETS.items():
        try:
            df = get_market_data(ticker)
            usable = df.dropna(subset=["Close"])
            if usable.empty:
                errors.append(f"{name}: no data")
                continue
            market[ticker] = df
            prices[ticker] = float(usable["Close"].iloc[-1])
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")

for name, ticker in ASSETS.items():
    if ticker not in market:
        continue

    df = market[ticker]
    valid = df.dropna(subset=["Close"])
    last = valid.iloc[-1]
    prev = valid.iloc[-2] if len(valid) > 1 else last
    price = prices[ticker]
    change = (
        (price / float(prev["Close"]) - 1) * 100
        if float(prev["Close"]) else 0
    )
    trend = (
        "Bullish" if pd.notna(last.get("EMA20"))
        and pd.notna(last.get("EMA50"))
        and last["EMA20"] > last["EMA50"]
        else "Bearish / unclear"
    )

    with st.container(border=True):
        st.markdown(f"**{name}** · `{ticker}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest available price", f"{price:,.4f}", f"{change:+.2f}%")
        rsi = last.get("RSI")
        c2.metric("RSI", f"{rsi:.1f}" if pd.notna(rsi) else "—")
        c3.metric("EMA trend", trend)

        with st.expander("View chart"):
            st.line_chart(valid[["Close", "EMA20", "EMA50"]].tail(90))
            st.caption(
                "Daily historical data from Yahoo Finance; "
                "not guaranteed to be real-time."
            )

for error in errors:
    st.warning(error)


# ---------- PORTFOLIO ----------
st.divider()
st.subheader("💰 Your portfolio")

cash = float(state.get("cash", STARTING_CASH))
positions = state.get("positions", {})
trades = state.get("trades", [])
equity = get_portfolio_value(cash, positions, prices)
unrealized = 0.0

for ticker, pos in positions.items():
    current = prices.get(ticker, float(pos["entry"]))
    unrealized += (
        float(pos["units"]) * current - float(pos["cost"])
    )

closed = [
    trade for trade in trades
    if trade.get("action") == "SELL"
    and trade.get("pnl") is not None
]
realized = sum(float(t["pnl"]) for t in closed)
wins = sum(1 for t in closed if float(t["pnl"]) > 0)
losses = sum(1 for t in closed if float(t["pnl"]) <= 0)
win_rate = wins / len(closed) * 100 if closed else 0

initial_value = float(state.get("starting_equity", STARTING_CASH))
total_pnl = equity - initial_value
total_return = (
    total_pnl / initial_value * 100 if initial_value else 0
)

c1, c2 = st.columns(2)
c1.metric("Portfolio value", f"${equity:,.2f}", f"{total_pnl:+.2f} USD")
c2.metric("Total return", f"{total_return:+.2f}%")

c3, c4 = st.columns(2)
c3.metric("Available cash", f"${cash:,.2f}")
c4.metric("Unrealized P/L", f"${unrealized:+,.2f}")

c5, c6 = st.columns(2)
c5.metric("Realized P/L", f"${realized:+,.2f}")
c6.metric("Closed trade win rate", f"{win_rate:.1f}%",
          f"{wins} wins / {losses} losses")

st.caption(
    "Last database update: "
    + fmt_time(state.get("_db_updated_at"))
)


# ---------- OPEN POSITIONS ----------
st.subheader("📌 Open positions")

if positions:
    rows = []
    for ticker, pos in positions.items():
        current = prices.get(ticker, float(pos["entry"]))
        value = float(pos["units"]) * current
        cost = float(pos["cost"])
        pnl = value - cost
        rows.append({
            "Asset": pos.get("name", ticker),
            "Entry": round(float(pos["entry"]), 4),
            "Latest": round(current, 4),
            "Position value": round(value, 2),
            "Unrealized P/L": round(pnl, 2),
            "P/L %": round(pnl / cost * 100, 2) if cost else 0,
            "Opened": fmt_time(pos.get("opened")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)
else:
    st.info("No open positions right now.")


# ---------- DAILY / MONTHLY REPORTS ----------
st.divider()
st.subheader("📈 Performance history")

snapshots = state.get("equity_snapshots", [])
if snapshots:
    history = pd.DataFrame(snapshots)
    if {"time", "equity"}.issubset(history.columns):
        history["time"] = pd.to_datetime(history["time"], utc=True,
                                         errors="coerce")
        history["equity"] = pd.to_numeric(history["equity"],
                                          errors="coerce")
        history = history.dropna(subset=["time", "equity"]).sort_values("time")

        if not history.empty:
            history["day"] = history["time"].dt.strftime("%Y-%m-%d")
            history["month"] = history["time"].dt.strftime("%Y-%m")
            st.line_chart(history.set_index("time")["equity"])

            daily = history.groupby("day", as_index=False).agg(
                start=("equity", "first"),
                end=("equity", "last"),
            )
            daily["P/L"] = daily["end"] - daily["start"]
            daily["P/L %"] = np.where(
                daily["start"] != 0,
                daily["P/L"] / daily["start"] * 100,
                0,
            )

            monthly = history.groupby("month", as_index=False).agg(
                start=("equity", "first"),
                end=("equity", "last"),
            )
            monthly["P/L"] = monthly["end"] - monthly["start"]
            monthly["P/L %"] = np.where(
                monthly["start"] != 0,
                monthly["P/L"] / monthly["start"] * 100,
                0,
            )

            tab_day, tab_month = st.tabs(["Daily", "Monthly"])
            with tab_day:
                st.dataframe(
                    daily.iloc[::-1].round(2),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"Winning days: {(daily['P/L'] > 0).sum()} · "
                    f"Losing days: {(daily['P/L'] < 0).sum()}"
                )
            with tab_month:
                st.dataframe(
                    monthly.iloc[::-1].round(2),
                    use_container_width=True,
                    hide_index=True,
                )
else:
    st.info(
        "Performance snapshots will appear after the Worker is updated "
        "to save daily portfolio values. Historical daily/monthly equity "
        "cannot be reconstructed accurately from trade records alone."
    )


# ---------- TRADE HISTORY ----------
st.divider()
st.subheader("🧾 Trade history")

if trades:
    trade_df = pd.DataFrame(trades)
    if "time" in trade_df.columns:
        trade_df = trade_df.sort_values("time", ascending=False)
    st.dataframe(trade_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades recorded yet.")


# ---------- PAPER TRADING ----------
st.divider()
st.subheader("⚙️ Paper-trading controls")
st.caption(
    "The GitHub Actions Worker runs independently. "
    "This button requests a manual virtual trading cycle."
)

if st.button("▶️ Run paper-trading cycle", type="primary",
             use_container_width=True):
    new_state = dict(state)
    new_positions = dict(positions)
    new_trades = list(trades)
    new_cash = cash

    for name, ticker in ASSETS.items():
        if ticker not in market:
            continue

        df = market[ticker].dropna(subset=["EMA20", "EMA50", "Close"])
        if df.empty:
            continue

        last = df.iloc[-1]
        price = prices[ticker]
        bullish = float(last["EMA20"]) > float(last["EMA50"])
        pos = new_positions.get(ticker)

        if bullish and pos is None:
            current_equity = get_portfolio_value(
                new_cash, new_positions, prices
            )
            budget = min(new_cash, current_equity * ALLOCATION)

            if budget > 1:
                units = budget * (1 - FEE_RATE) / price
                new_cash -= budget
                new_positions[ticker] = {
                    "name": name,
                    "entry": price,
                    "units": units,
                    "cost": budget,
                    "opened": utc_now(),
                }
                new_trades.append({
                    "time": utc_now(),
                    "asset": name,
                    "action": "BUY",
                    "price": price,
                    "pnl": None,
                })

        elif not bullish and pos is not None:
            proceeds = float(pos["units"]) * price * (1 - FEE_RATE)
            pnl = proceeds - float(pos["cost"])
            new_cash += proceeds
            new_trades.append({
                "time": utc_now(),
                "asset": name,
                "action": "SELL",
                "price": price,
                "pnl": round(pnl, 2),
            })
            del new_positions[ticker]

    new_state.update({
        "cash": new_cash,
        "positions": new_positions,
        "trades": new_trades,
    })

    try:
        save_state(new_state)
        st.success("Virtual trading cycle saved.")
        st.rerun()
    except Exception as exc:
        st.error("Could not save the cycle.")
        st.code(f"{type(exc).__name__}: {str(exc)[:300]}")


# ---------- SAVE / LOCK ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reload saved portfolio", use_container_width=True):
        refresh_saved_state()
with col2:
    if st.button("🔒 Lock JARVIS", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.caption(
    "Prices may be delayed. Historical daily prices are not live quotes. "
    "All trades shown here are simulated; no real money is used."
)
