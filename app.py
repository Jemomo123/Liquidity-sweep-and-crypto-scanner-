"""
=============================================================================
CRYPTO FUTURES SCANNER DASHBOARD
=============================================================================
A live, auto-refreshing crypto futures scanner with two engines:
  - Engine A: Multi-pair futures scanner (3m / 5m)
  - Engine B: BTC Liquidity + Sweep Engine (1H)

DATA SOURCES (all public, no API keys required):
  - OKX, Gate.io, MEXC for multi-pair scanner
  - Binance for BTC Liquidity Engine

=============================================================================
STREAMLIT CLOUD DEPLOYMENT INSTRUCTIONS
=============================================================================

1. REPOSITORY SETUP:
   - Create a new GitHub repository
   - Add app.py and requirements.txt to the root of the repo
   - Commit and push both files

2. STREAMLIT CLOUD:
   - Go to https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub account and select the repository
   - Set Main file path: app.py
   - Click "Deploy!"

3. ENVIRONMENT:
   - No secrets or API keys required (all public endpoints)
   - Python 3.9+ recommended
   - requirements.txt is auto-installed by Streamlit Cloud

4. PERFORMANCE NOTES:
   - App auto-refreshes every 60 seconds via streamlit-autorefresh
   - Cache TTL is set to 60 seconds (@st.cache_data(ttl=60))
   - "Scan Now" button clears cache and forces immediate refresh
   - On Streamlit Cloud free tier, expect ~3-6s load time per refresh

5. TROUBLESHOOTING:
   - If exchanges return errors, the app falls back gracefully
   - Rate limits: app uses public REST endpoints with no auth
   - If data is stale, click "Scan Now" to force refresh

=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from typing import Optional

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Futures Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light theme CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header {
        font-size: 1.6rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.85rem; color: #6c757d; margin-bottom: 1.2rem;
    }
    .regime-box {
        border-radius: 10px; padding: 16px 20px; margin-bottom: 8px;
        border: 1px solid #dee2e6;
    }
    .regime-trending { background: #d4edda; border-color: #c3e6cb; }
    .regime-ranging  { background: #fff3cd; border-color: #ffeeba; }
    .signal-high   { color: #155724; font-weight: 700; }
    .signal-medium { color: #856404; font-weight: 700; }
    .signal-low    { color: #721c24; font-weight: 600; }
    .signal-grey   { color: #868e96; }
    .stDataFrame { font-size: 0.82rem; }
    div[data-testid="metric-container"] {
        background: white; border-radius: 8px;
        border: 1px solid #dee2e6; padding: 8px 12px;
    }
    .section-title {
        font-size: 1rem; font-weight: 600;
        color: #343a40; margin: 0.8rem 0 0.4rem 0;
        border-bottom: 2px solid #dee2e6; padding-bottom: 4px;
    }
    .liq-panel {
        background: white; border-radius: 10px;
        padding: 16px; border: 1px solid #dee2e6;
    }
    .narrative-text {
        font-size: 0.9rem; color: #343a40; line-height: 1.6;
    }
    .tag-bullish { background:#d4edda; color:#155724; border-radius:4px; padding:2px 8px; font-size:0.78rem; }
    .tag-bearish { background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 8px; font-size:0.78rem; }
    .tag-neutral { background:#e2e3e5; color:#383d41; border-radius:4px; padding:2px 8px; font-size:0.78rem; }
    .stButton>button {
        background:#1a1a2e; color:white; border-radius:8px;
        border:none; padding:8px 20px; font-weight:600;
    }
    .stButton>button:hover { background:#16213e; }
    table.scanner-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
    table.scanner-table th {
        background:#f1f3f5; color:#343a40;
        padding:8px 10px; text-align:left; border-bottom:2px solid #dee2e6;
    }
    table.scanner-table td {
        padding:7px 10px; border-bottom:1px solid #f1f3f5; vertical-align:top;
    }
    table.scanner-table tr:hover td { background:#f8f9fa; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COMPRESSION_THRESHOLD = 0.004   # 0.4%
EXPANSION_BODY_RATIO  = 1.5     # 150% of avg body
EXPANSION_WICK_RATIO  = 0.60    # 60% of range
TREND_SMA_SEP         = 0.012   # 1.2% for reversion
REGIME_SMA_SEP        = 0.01    # 1.0% for trending label
FIREWALL_DIST         = 0.01    # 1% swing obstacle
LIQUIDITY_HOLE        = 0.015   # 1.5% for room
MAX_SIGNAL_AGE        = 2       # candles before greying out

HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/1.0"}
TIMEOUT  = 8

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(url: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def klines_to_df(raw: list, source: str = "binance") -> pd.DataFrame:
    """Convert raw kline lists to OHLCV DataFrame."""
    if not raw:
        return pd.DataFrame()
    try:
        if source == "binance":
            cols = ["ts","open","high","low","close","vol",
                    "close_ts","qvol","trades","tbbv","tbqv","ignore"]
            df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
        elif source == "okx":
            cols = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]
            df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
        elif source == "gate":
            df = pd.DataFrame(raw)
            df = df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"vol"})
        elif source == "mexc":
            cols = ["ts","open","high","low","close","vol","close_ts",
                    "qvol","trades","tbbv","tbqv","ignore"]
            df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
        else:
            return pd.DataFrame()

        for c in ["open","high","low","close","vol"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            if df["ts"].max() > 1e12:
                df["ts"] = df["ts"] / 1000
        df = df.dropna(subset=["open","high","low","close"]).sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# ─── OKX ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def okx_top_pairs(n: int = 40) -> list:
    url = "https://www.okx.com/api/v5/public/instruments"
    data = safe_get(url, {"instType": "SWAP"})
    if not data or "data" not in data:
        return []
    pairs = [d["instId"] for d in data["data"] if d.get("instId","").endswith("-USDT-SWAP")]
    # Get 24h volume
    ticker_url = "https://www.okx.com/api/v5/market/tickers"
    t_data = safe_get(ticker_url, {"instType": "SWAP"})
    if not t_data or "data" not in t_data:
        return pairs[:n]
    vols = {d["instId"]: float(d.get("volCcy24h", 0)) for d in t_data["data"]}
    pairs_sorted = sorted(pairs, key=lambda x: vols.get(x, 0), reverse=True)
    # Ensure BTC included
    btc = "BTC-USDT-SWAP"
    if btc not in pairs_sorted[:n]:
        pairs_sorted = [btc] + [p for p in pairs_sorted if p != btc]
    return pairs_sorted[:n]


@st.cache_data(ttl=60)
def okx_klines(inst_id: str, bar: str, limit: int = 120) -> pd.DataFrame:
    url = "https://www.okx.com/api/v5/market/candles"
    data = safe_get(url, {"instId": inst_id, "bar": bar, "limit": limit})
    if not data or "data" not in data:
        return pd.DataFrame()
    raw = data["data"]
    raw.reverse()  # OKX returns newest first
    return klines_to_df(raw, "okx")


# ─── Gate.io ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def gate_top_pairs(n: int = 40) -> list:
    url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
    data = safe_get(url)
    if not isinstance(data, list):
        return []
    pairs = [(d["name"], float(d.get("volume_24h_base", 0))) for d in data if "USDT" in d.get("name","")]
    pairs_sorted = [p[0] for p in sorted(pairs, key=lambda x: x[1], reverse=True)]
    btc = "BTC_USDT"
    if btc not in pairs_sorted[:n]:
        pairs_sorted = [btc] + [p for p in pairs_sorted if p != btc]
    return pairs_sorted[:n]


@st.cache_data(ttl=60)
def gate_klines(contract: str, interval: str, limit: int = 120) -> pd.DataFrame:
    url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
    data = safe_get(url, {"contract": contract, "interval": interval, "limit": limit})
    if not isinstance(data, list):
        return pd.DataFrame()
    df = klines_to_df(data, "gate")
    return df


# ─── MEXC ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def mexc_top_pairs(n: int = 40) -> list:
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    data = safe_get(url)
    if not data or "data" not in data:
        return []
    pairs = [(d["symbol"], float(d.get("amount24", 0)))
             for d in data["data"] if "_USDT" in d.get("symbol","")]
    pairs_sorted = [p[0] for p in sorted(pairs, key=lambda x: x[1], reverse=True)]
    btc = "BTC_USDT"
    if btc not in pairs_sorted[:n]:
        pairs_sorted = [btc] + [p for p in pairs_sorted if p != btc]
    return pairs_sorted[:n]


@st.cache_data(ttl=60)
def mexc_klines(symbol: str, interval: str, limit: int = 120) -> pd.DataFrame:
    url = "https://contract.mexc.com/api/v1/contract/kline"
    data = safe_get(url, {"symbol": symbol, "interval": interval, "limit": limit})
    if not data or "data" not in data:
        return pd.DataFrame()
    d = data["data"]
    try:
        rows = []
        times = d.get("time", [])
        opens  = d.get("open",  [])
        highs  = d.get("high",  [])
        lows   = d.get("low",   [])
        closes = d.get("close", [])
        vols   = d.get("vol",   [])
        for i in range(len(times)):
            rows.append({
                "ts":    times[i],
                "open":  float(opens[i])  if i < len(opens)  else np.nan,
                "high":  float(highs[i])  if i < len(highs)  else np.nan,
                "low":   float(lows[i])   if i < len(lows)   else np.nan,
                "close": float(closes[i]) if i < len(closes) else np.nan,
                "vol":   float(vols[i])   if i < len(vols)   else np.nan,
            })
        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# ─── Binance (BTC Liquidity Engine) ─────────────────────────────────────────

@st.cache_data(ttl=60)
def binance_klines(symbol: str, interval: str, limit: int = 60) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    data = safe_get(url, {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(data, list):
        return pd.DataFrame()
    return klines_to_df(data, "binance")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    df = df.copy()
    df["sma20"]  = calc_sma(df["close"], 20)
    df["sma100"] = calc_sma(df["close"], 100)
    df["rsi14"]  = calc_rsi(df["close"], 14)
    df["body"]   = (df["close"] - df["open"]).abs()
    df["range"]  = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(10, min_periods=5).mean()
    return df


def get_swing_levels(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """Return recent swing highs and lows from last `lookback` candles."""
    if df.empty or len(df) < 5:
        return [], []
    sub = df.tail(lookback)
    highs, lows = [], []
    closes = sub["close"].values
    highs_arr = sub["high"].values
    lows_arr  = sub["low"].values
    for i in range(2, len(sub) - 2):
        if highs_arr[i] == max(highs_arr[i-2:i+3]):
            highs.append(highs_arr[i])
        if lows_arr[i] == min(lows_arr[i-2:i+3]):
            lows.append(lows_arr[i])
    return sorted(highs, reverse=True), sorted(lows)


# ─────────────────────────────────────────────────────────────────────────────
# CORE EDGE ENGINE — Signal Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_compression(df: pd.DataFrame) -> bool:
    """Price and SMA20/100 all within 0.4% of each other."""
    if df.empty or "sma20" not in df.columns or "sma100" not in df.columns:
        return False
    last = df.iloc[-1]
    price = last["close"]
    s20   = last.get("sma20",  np.nan)
    s100  = last.get("sma100", np.nan)
    if pd.isna(s20) or pd.isna(s100):
        return False
    spread = max(price, s20, s100) / min(price, s20, s100) - 1
    return spread <= COMPRESSION_THRESHOLD


def detect_expansion(df: pd.DataFrame) -> Optional[str]:
    """
    Returns 'long', 'short', or None.
    Expansion: first candle leaving compression, confirmed by elephant/tail candle.
    """
    if df.empty or len(df) < 3 or "sma20" not in df.columns:
        return None
    # Check if previous candle was in compression, current breaks out
    prev2 = df.iloc[-3]
    prev  = df.iloc[-2]
    curr  = df.iloc[-1]

    def in_comp(row):
        p, s20, s100 = row["close"], row.get("sma20", np.nan), row.get("sma100", np.nan)
        if pd.isna(s20) or pd.isna(s100): return False
        spread = max(p, s20, s100) / min(p, s20, s100) - 1
        return spread <= COMPRESSION_THRESHOLD

    was_comp = in_comp(prev2) or in_comp(prev)
    if not was_comp:
        return None

    # Check elephant or tail candle on current or previous
    def is_elephant(row):
        avg_b = row.get("avg_body", np.nan)
        if pd.isna(avg_b) or avg_b == 0: return False, None
        body  = row.get("body", 0)
        rng   = row.get("range", 0)
        wick  = rng - body
        if body >= EXPANSION_BODY_RATIO * avg_b:
            return True, "Long" if row["close"] > row["open"] else "Short"
        if rng > 0 and wick >= EXPANSION_WICK_RATIO * rng:
            upper_wick = row["high"] - max(row["open"], row["close"])
            lower_wick = min(row["open"], row["close"]) - row["low"]
            if lower_wick > upper_wick:
                return True, "Long"
            else:
                return True, "Short"
        return False, None

    el_curr, dir_curr = is_elephant(curr)
    el_prev, dir_prev = is_elephant(prev)

    if el_curr and dir_curr:
        return dir_curr.lower()
    if el_prev and dir_prev:
        return dir_prev.lower()
    return None


def detect_pullback(df: pd.DataFrame) -> Optional[str]:
    """First pullback to SMA20 with confirmation."""
    if df.empty or len(df) < 5 or "sma20" not in df.columns:
        return None
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    s20_last = last.get("sma20",  np.nan)
    s20_prev = prev.get("sma20",  np.nan)
    if pd.isna(s20_last) or pd.isna(s20_prev):
        return None

    # Price touched SMA20 and bounced
    low_touched = prev["low"] <= s20_prev * 1.005 and prev["close"] > s20_prev
    high_touched = prev["high"] >= s20_prev * 0.995 and prev["close"] < s20_prev

    # Confirmation candle
    avg_b = last.get("avg_body", np.nan)
    body  = last.get("body", 0)
    is_conf = (not pd.isna(avg_b) and avg_b > 0 and body >= avg_b * 0.8)

    if low_touched and is_conf and last["close"] > s20_last:
        return "long"
    if high_touched and is_conf and last["close"] < s20_last:
        return "short"
    return None


def detect_reversion(df: pd.DataFrame) -> Optional[str]:
    """
    Reversion to SMA100 allowed if:
    - SMA20-SMA100 separation >= 1.2%
    - No SMA crossings in last 15 candles
    """
    if df.empty or len(df) < 20 or "sma20" not in df.columns or "sma100" not in df.columns:
        return None
    last = df.iloc[-1]
    s20   = last.get("sma20",  np.nan)
    s100  = last.get("sma100", np.nan)
    if pd.isna(s20) or pd.isna(s100) or s100 == 0:
        return None

    sep = abs(s20 - s100) / s100
    if sep < TREND_SMA_SEP:
        return None

    # Check no crossings in last 15 candles
    sub = df.tail(15)
    above = (sub["sma20"] > sub["sma100"]).values
    crossings = sum(1 for i in range(1, len(above)) if above[i] != above[i-1])
    if crossings > 0:
        return None

    # Price approaching SMA100
    price = last["close"]
    dist_to_100 = abs(price - s100) / s100
    if dist_to_100 > 0.006:  # within 0.6%
        return None

    direction = "long" if s20 > s100 else "short"
    return direction


def check_firewalls(df: pd.DataFrame, direction: str) -> dict:
    """Check swing level obstacles within 1%."""
    price = df.iloc[-1]["close"] if not df.empty else None
    if price is None:
        return {"obstacle": "Unknown", "reason": "No data"}

    swing_highs, swing_lows = get_swing_levels(df, lookback=30)

    if direction == "long":
        nearby = [h for h in swing_highs if 0 < (h - price) / price <= FIREWALL_DIST]
        if nearby:
            dist_pct = (nearby[0] - price) / price * 100
            return {"obstacle": "Resistance nearby", "reason": f"Swing high {dist_pct:.1f}% above"}
        return {"obstacle": "None", "reason": "No swing within 1%"}
    else:
        nearby = [l for l in swing_lows if 0 < (price - l) / price <= FIREWALL_DIST]
        if nearby:
            dist_pct = (price - nearby[0]) / price * 100
            return {"obstacle": "Support nearby", "reason": f"Swing low {dist_pct:.1f}% below"}
        return {"obstacle": "None", "reason": "No swing within 1%"}


def check_liquidity_hole(df: pd.DataFrame, direction: str) -> dict:
    """Next swing >= 1.5% away means room exists."""
    price = df.iloc[-1]["close"] if not df.empty else None
    if price is None:
        return {"room": "Unknown", "reason": "No data"}

    swing_highs, swing_lows = get_swing_levels(df, lookback=40)

    if direction == "long":
        ahead = [h for h in swing_highs if h > price]
        if ahead:
            dist = (ahead[0] - price) / price
            if dist >= 0.025:
                return {"room": "Large", "reason": f"Next swing {dist*100:.1f}% away"}
            elif dist >= LIQUIDITY_HOLE:
                return {"room": "Moderate", "reason": f"Next swing {dist*100:.1f}% away"}
            else:
                return {"room": "Limited", "reason": f"Next swing {dist*100:.1f}% away"}
        return {"room": "Large", "reason": "No major swing overhead"}
    else:
        ahead = [l for l in swing_lows if l < price]
        if ahead:
            dist = (price - ahead[0]) / price
            if dist >= 0.025:
                return {"room": "Large", "reason": f"Next swing {dist*100:.1f}% away"}
            elif dist >= LIQUIDITY_HOLE:
                return {"room": "Moderate", "reason": f"Next swing {dist*100:.1f}% away"}
            else:
                return {"room": "Limited", "reason": f"Next swing {dist*100:.1f}% away"}
        return {"room": "Large", "reason": "No major swing below"}


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(signal_age: int, setup_type: str, bias_15m: str,
                 direction: str, room: str, obstacle: str, rsi: float,
                 candle_quality: str) -> tuple:
    """Returns (score, tier, conviction_parts)."""
    score = 0
    parts = []

    # Freshness (25 pts)
    if signal_age == 0:
        score += 25; parts.append("New breakout")
    elif signal_age == 1:
        score += 15; parts.append("1 candle ago")
    elif signal_age == 2:
        score += 5;  parts.append("2 candles ago")

    # Candle confirmation (20 pts)
    if candle_quality == "elephant":
        score += 20; parts.append("Strong elephant candle")
    elif candle_quality == "tail":
        score += 15; parts.append("Rejection tail")
    elif candle_quality == "ok":
        score += 10; parts.append("Confirmed candle")

    # 15m Bias alignment (15 pts)
    if (direction == "long" and bias_15m == "bullish") or \
       (direction == "short" and bias_15m == "bearish"):
        score += 15; parts.append("15m aligned")
    elif bias_15m == "neutral":
        score += 5; parts.append("15m neutral")
    else:
        parts.append("15m against")

    # Liquidity hole (15 pts)
    if room == "Large":
        score += 15; parts.append("Large room")
    elif room == "Moderate":
        score += 8;  parts.append("Moderate room")
    else:
        parts.append("Limited room")

    # Firewall (15 pts)
    if obstacle == "None":
        score += 15; parts.append("No resistance")
    else:
        score += 0;  parts.append("Obstacle present")

    # RSI (10 pts)
    if not pd.isna(rsi):
        if direction == "long" and rsi < 40:
            score += 10; parts.append("RSI in fuel zone")
        elif direction == "short" and rsi > 60:
            score += 10; parts.append("RSI in fuel zone")
        elif direction == "long" and 40 <= rsi <= 55:
            score += 6;  parts.append("RSI neutral")
        elif direction == "short" and 45 <= rsi <= 60:
            score += 6;  parts.append("RSI neutral")
        elif direction == "long" and rsi > 70:
            score += 0;  parts.append("RSI overextended")
        elif direction == "short" and rsi < 30:
            score += 0;  parts.append("RSI overextended")
        else:
            score += 3;  parts.append("RSI moderate")

    if score >= 75:
        tier = "HIGH"
    elif score >= 55:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return score, tier, parts


# ─────────────────────────────────────────────────────────────────────────────
# 15m BIAS
# ─────────────────────────────────────────────────────────────────────────────

def get_bias_15m(df_15m: pd.DataFrame) -> str:
    if df_15m.empty or "sma20" not in df_15m.columns or "sma100" not in df_15m.columns:
        return "neutral"
    last = df_15m.iloc[-1]
    s20  = last.get("sma20",  np.nan)
    s100 = last.get("sma100", np.nan)
    if pd.isna(s20) or pd.isna(s100):
        return "neutral"
    if s20 > s100 * 1.002:
        return "bullish"
    if s20 < s100 * 0.998:
        return "bearish"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE QUALITY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_candle(df: pd.DataFrame, idx: int = -1) -> str:
    if df.empty or abs(idx) > len(df):
        return "weak"
    row = df.iloc[idx]
    avg_b = row.get("avg_body", np.nan)
    body  = row.get("body", 0)
    rng   = row.get("range", 0)
    if pd.isna(avg_b) or avg_b == 0 or rng == 0:
        return "weak"
    if body >= EXPANSION_BODY_RATIO * avg_b:
        return "elephant"
    wick = rng - body
    if wick >= EXPANSION_WICK_RATIO * rng:
        return "tail"
    if body >= avg_b * 0.8:
        return "ok"
    return "weak"


# ─────────────────────────────────────────────────────────────────────────────
# BTC REGIME ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_btc_regime() -> dict:
    """Returns regime dict for 15m, 1H, 4H using OKX data."""
    result = {}
    tf_map = {"15m": "15m", "1H": "1H", "4H": "4H"}
    for label, tf in tf_map.items():
        try:
            df = okx_klines("BTC-USDT-SWAP", tf, limit=120)
            if df.empty or len(df) < 20:
                result[label] = {"regime": "Unknown", "sep": 0, "price": None}
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            s20  = last.get("sma20",  np.nan)
            s100 = last.get("sma100", np.nan)
            if pd.isna(s20) or pd.isna(s100) or s100 == 0:
                result[label] = {"regime": "Unknown", "sep": 0, "price": last["close"]}
                continue
            sep = (s20 - s100) / s100
            trending = abs(sep) >= REGIME_SMA_SEP
            result[label] = {
                "regime": "Trending" if trending else "Ranging",
                "direction": "Up" if sep > 0 else "Down",
                "sep_pct": round(abs(sep) * 100, 2),
                "price": round(last["close"], 2),
                "sma20": round(s20, 2),
                "sma100": round(s100, 2),
            }
        except Exception:
            result[label] = {"regime": "Unknown", "sep": 0, "price": None}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BTC LIQUIDITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_btc_1h_data() -> pd.DataFrame:
    df = binance_klines("BTCUSDT", "1h", 60)
    if df.empty:
        return pd.DataFrame()
    return add_indicators(df)


def get_bias(df: pd.DataFrame) -> str:
    return get_bias_15m(df)  # same logic works for 1H


def get_liquidity_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"highs": [], "lows": []}
    highs, lows = get_swing_levels(df, lookback=40)
    price = df.iloc[-1]["close"]
    nearby_highs = [h for h in highs if h > price][:3]
    nearby_lows  = [l for l in lows  if l < price][:3]
    return {
        "highs": [round(h, 2) for h in nearby_highs],
        "lows":  [round(l, 2) for l in nearby_lows],
        "price": round(price, 2),
    }


def get_sweep(df: pd.DataFrame) -> Optional[dict]:
    """Detect recent liquidity sweep (wick beyond prior swing)."""
    if df.empty or len(df) < 10:
        return None
    price = df.iloc[-1]["close"]
    highs, lows = get_swing_levels(df.iloc[:-5], lookback=30)

    last3 = df.tail(5)
    for _, row in last3.iterrows():
        if highs and row["high"] > highs[0] and row["close"] < highs[0]:
            return {
                "type": "Bearish sweep",
                "level": round(highs[0], 2),
                "candle_close": round(row["close"], 2),
            }
        if lows and row["low"] < lows[0] and row["close"] > lows[0]:
            return {
                "type": "Bullish sweep",
                "level": round(lows[0], 2),
                "candle_close": round(row["close"], 2),
            }
    return None


def get_impulse(df: pd.DataFrame) -> Optional[dict]:
    """Detect recent impulse move (large candle > 1.5x avg body)."""
    if df.empty or len(df) < 5:
        return None
    last5 = df.tail(5)
    for i in range(len(last5) - 1, -1, -1):
        row = last5.iloc[i]
        avg_b = row.get("avg_body", np.nan)
        body  = row.get("body",     0)
        if not pd.isna(avg_b) and avg_b > 0 and body >= EXPANSION_BODY_RATIO * avg_b:
            direction = "bullish" if row["close"] > row["open"] else "bearish"
            pct = body / row["close"] * 100
            return {
                "direction": direction,
                "magnitude_pct": round(pct, 2),
                "candle_idx": i,
            }
    return None


def get_pullback_zone(df: pd.DataFrame, impulse: Optional[dict]) -> Optional[dict]:
    """Expected pullback zone around SMA20."""
    if df.empty or "sma20" not in df.columns:
        return None
    last  = df.iloc[-1]
    s20   = last.get("sma20", np.nan)
    price = last["close"]
    if pd.isna(s20):
        return None
    lower = round(s20 * 0.995, 2)
    upper = round(s20 * 1.005, 2)
    in_zone = lower <= price <= upper
    return {
        "lower": lower,
        "upper": upper,
        "sma20": round(s20, 2),
        "in_zone": in_zone,
    }


def get_price_path(df: pd.DataFrame, bias: str, sweep: Optional[dict],
                   impulse: Optional[dict]) -> str:
    """Generate narrative price path description."""
    if df.empty:
        return "Insufficient data to determine price path."
    last  = df.iloc[-1]
    price = round(last["close"], 2)

    parts = []
    if bias == "bullish":
        parts.append(f"BTC is in a bullish structure at ${price:,}.")
    elif bias == "bearish":
        parts.append(f"BTC is in a bearish structure at ${price:,}.")
    else:
        parts.append(f"BTC is ranging near ${price:,}.")

    if sweep:
        parts.append(f"{sweep['type']} detected at ${sweep['level']:,} — price rejected and closed at ${sweep['candle_close']:,}.")

    if impulse:
        parts.append(f"Recent {impulse['direction']} impulse of {impulse['magnitude_pct']}% suggests momentum.")
    else:
        parts.append("No strong impulse in the last 5 candles.")

    liq = get_liquidity_levels(df)
    if liq["highs"]:
        parts.append(f"Liquidity resting above at ${liq['highs'][0]:,}.")
    if liq["lows"]:
        parts.append(f"Support/liquidity below at ${liq['lows'][0]:,}.")

    return " ".join(parts)


@st.cache_data(ttl=60)
def build_btc_liquidity_panel() -> dict:
    df = get_btc_1h_data()
    if df.empty:
        return {"error": "Unable to fetch BTC 1H data"}

    bias    = get_bias(df)
    levels  = get_liquidity_levels(df)
    sweep   = get_sweep(df)
    impulse = get_impulse(df)
    zone    = get_pullback_zone(df, impulse)
    path    = get_price_path(df, bias, sweep, impulse)

    # RSI
    rsi_val = df.iloc[-1].get("rsi14", np.nan)

    return {
        "bias":      bias,
        "levels":    levels,
        "sweep":     sweep,
        "impulse":   impulse,
        "zone":      zone,
        "narrative": path,
        "price":     levels.get("price"),
        "rsi":       round(rsi_val, 1) if not pd.isna(rsi_val) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PAIR SCANNER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def normalize_pair(raw: str, source: str) -> str:
    """Normalize pair name for display."""
    raw = raw.upper()
    if source == "okx":
        return raw.replace("-SWAP", "").replace("-", "/")
    if source == "gate":
        return raw.replace("_", "/")
    if source == "mexc":
        return raw.replace("_", "/")
    return raw


def get_tf_klines(pair: str, source: str, tf: str, limit: int = 120) -> pd.DataFrame:
    """Route kline fetch to the right exchange function."""
    try:
        if source == "okx":
            tf_map = {"3m": "3m", "5m": "5m", "15m": "15m"}
            return okx_klines(pair, tf_map.get(tf, tf), limit)
        if source == "gate":
            tf_map = {"3m": "3m", "5m": "5m", "15m": "15m"}
            return gate_klines(pair, tf_map.get(tf, tf), limit)
        if source == "mexc":
            tf_map = {"3m": "Min3", "5m": "Min5", "15m": "Min15"}
            return mexc_klines(pair, tf_map.get(tf, tf), limit)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def scan_pair(pair: str, source: str, tf: str) -> Optional[dict]:
    """
    Scan a single pair on a given timeframe.
    Returns signal dict or None.
    """
    try:
        df_exec = get_tf_klines(pair, source, tf, limit=130)
        df_15m  = get_tf_klines(pair, source, "15m", limit=130)

        if df_exec.empty or len(df_exec) < 25:
            return None

        df_exec = add_indicators(df_exec)
        df_15m  = add_indicators(df_15m) if not df_15m.empty and len(df_15m) >= 20 else df_15m

        last  = df_exec.iloc[-1]
        price = last["close"]
        rsi   = last.get("rsi14", np.nan)

        # Mutual exclusion: expansion OR reversion, not both
        exp_dir = detect_expansion(df_exec)
        rev_dir = detect_reversion(df_exec)
        pb_dir  = detect_pullback(df_exec)

        setup = None
        direction = None

        if exp_dir:
            setup = "Expansion"
            direction = exp_dir
        elif pb_dir and not exp_dir:
            setup = "Pullback"
            direction = pb_dir
        elif rev_dir and not exp_dir:
            setup = "Reversion"
            direction = rev_dir
        else:
            return None

        # 15m bias — get early so we can filter contradictory signals
        bias_15m = get_bias_15m(df_15m) if not df_15m.empty else "neutral"

        # FILTER: Drop signals that go hard against 15m trend
        # Short with Bullish 15m, or Long with Bearish 15m → skip
        if direction == "long" and bias_15m == "bearish":
            return None
        if direction == "short" and bias_15m == "bullish":
            return None

        # Signal age — check if signal just appeared or was there before
        signal_age = 0
        if len(df_exec) > 3:
            was_there_1_ago = False
            was_there_2_ago = False
            if setup == "Expansion":
                was_there_1_ago = detect_expansion(df_exec.iloc[:-1]) is not None
                was_there_2_ago = detect_expansion(df_exec.iloc[:-2]) is not None
            elif setup == "Pullback":
                was_there_1_ago = detect_pullback(df_exec.iloc[:-1]) is not None
                was_there_2_ago = detect_pullback(df_exec.iloc[:-2]) is not None

            if was_there_2_ago:
                signal_age = 2
            elif was_there_1_ago:
                signal_age = 1
            else:
                signal_age = 0

        # Drop signals older than 2 candles
        if signal_age > MAX_SIGNAL_AGE:
            return None

        # Room and obstacle
        room_info     = check_liquidity_hole(df_exec, direction)
        obstacle_info = check_firewalls(df_exec, direction)

        # FILTER: Drop signals where next swing is less than 0.5% away — too crowded
        room_reason = room_info.get("reason", "")
        if room_info["room"] == "Limited":
            # Extract distance from reason string e.g. "Next swing 0.1% away"
            try:
                dist_str = room_reason.split("Next swing ")[1].split("%")[0]
                dist_val = float(dist_str)
                if dist_val < 0.5:
                    return None  # Too close to next swing, no room at all
            except Exception:
                pass

        # Candle quality
        cq = classify_candle(df_exec, -1)
        if cq == "weak":
            cq = classify_candle(df_exec, -2)

        # FILTER: Drop weak candle quality signals entirely
        if cq == "weak":
            return None

        score, tier, conviction_parts = score_signal(
            signal_age, setup, bias_15m, direction,
            room_info["room"], obstacle_info["obstacle"],
            rsi, cq
        )

        # FILTER: Drop very low quality signals (score < 30)
        if score < 30:
            return None

        display_name = normalize_pair(pair, source)

        return {
            "pair":         display_name,
            "raw_pair":     pair,
            "source":       source,
            "tf":           tf,
            "setup":        setup,
            "direction":    direction.capitalize(),
            "score":        score,
            "tier":         tier,
            "conviction":   f"{tier} ({' • '.join(conviction_parts)})",
            "bias_15m":     bias_15m.capitalize(),
            "rsi":          round(rsi, 1) if not pd.isna(rsi) else "—",
            "room":         room_info["room"],
            "room_reason":  room_info["reason"],
            "obstacle":     obstacle_info["obstacle"],
            "obs_reason":   obstacle_info["reason"],
            "freshness":    "New" if signal_age == 0 else f"{signal_age} candle{'s' if signal_age > 1 else ''} ago",
            "signal_age":   signal_age,
            "price":        round(price, 4),
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def run_scanner(tf: str) -> pd.DataFrame:
    """
    Run the full multi-pair scanner for a given timeframe.
    Pulls top pairs from OKX, Gate.io, MEXC.
    """
    results = []

    # Gather pairs
    pair_sources = []

    okx_pairs = okx_top_pairs(35)
    for p in okx_pairs[:30]:
        pair_sources.append((p, "okx"))

    gate_pairs = gate_top_pairs(25)
    for p in gate_pairs[:20]:
        # Avoid duplicates
        norm = normalize_pair(p, "gate")
        existing = {normalize_pair(x[0], x[1]) for x in pair_sources}
        if norm not in existing:
            pair_sources.append((p, "gate"))

    mexc_pairs = mexc_top_pairs(20)
    for p in mexc_pairs[:15]:
        norm = normalize_pair(p, "mexc")
        existing = {normalize_pair(x[0], x[1]) for x in pair_sources}
        if norm not in existing:
            pair_sources.append((p, "mexc"))

    # Limit total pairs
    pair_sources = pair_sources[:50]

    progress = st.empty()
    total = len(pair_sources)

    for i, (pair, source) in enumerate(pair_sources):
        progress.caption(f"Scanning {i+1}/{total}: {normalize_pair(pair, source)}")
        sig = scan_pair(pair, source, tf)
        if sig:
            results.append(sig)
        # Small delay to avoid rate limits
        time.sleep(0.05)

    progress.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BTC REGIME BOX RENDERER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_btc_regime_cached() -> dict:
    return get_btc_regime()


def render_regime_box(regime: dict):
    st.markdown('<p class="section-title">📊 BTC Market Regime</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    tf_labels = {"15m": "15-Minute", "1H": "1-Hour", "4H": "4-Hour"}
    for i, (tf, label) in enumerate(tf_labels.items()):
        with cols[i]:
            r = regime.get(tf, {})
            reg    = r.get("regime",    "Unknown")
            dirn   = r.get("direction", "—")
            sep    = r.get("sep_pct",   0)
            price  = r.get("price",     "—")
            s20    = r.get("sma20",     "—")
            s100   = r.get("sma100",    "—")

            css_class = "regime-trending" if reg == "Trending" else "regime-ranging"
            icon = "🟢" if reg == "Trending" and dirn == "Up" else \
                   "🔴" if reg == "Trending" and dirn == "Down" else "🟡"

            st.markdown(f"""
            <div class="regime-box {css_class}">
                <div style="font-weight:700; font-size:0.9rem; color:#343a40;">{label}</div>
                <div style="font-size:1.3rem; font-weight:800; margin:4px 0;">{icon} {reg}</div>
                <div style="font-size:0.78rem; color:#495057;">
                    Direction: <b>{dirn}</b> &nbsp;|&nbsp; SMA gap: <b>{sep}%</b>
                </div>
                <div style="font-size:0.75rem; color:#6c757d; margin-top:4px;">
                    Price: {price} &nbsp; SMA20: {s20} &nbsp; SMA100: {s100}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BTC LIQUIDITY PANEL RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_liquidity_panel(panel: dict):
    st.markdown('<p class="section-title">💧 BTC Liquidity Engine (1H)</p>', unsafe_allow_html=True)

    if "error" in panel:
        st.warning(f"Liquidity data unavailable: {panel['error']}")
        return

    bias   = panel.get("bias",   "neutral")
    price  = panel.get("price",  "—")
    rsi    = panel.get("rsi",    "—")
    sweep  = panel.get("sweep")
    imp    = panel.get("impulse")
    zone   = panel.get("zone")
    levels = panel.get("levels", {})
    narr   = panel.get("narrative", "")

    bias_tag = f'<span class="tag-bullish">Bullish</span>' if bias == "bullish" else \
               f'<span class="tag-bearish">Bearish</span>' if bias == "bearish" else \
               f'<span class="tag-neutral">Neutral</span>'

    sweep_html = "None detected"
    if sweep:
        stype = sweep["type"]
        slvl  = sweep["level"]
        sclose= sweep["candle_close"]
        sweep_html = f'<span style="color:{"#155724" if "Bullish" in stype else "#721c24"}; font-weight:600;">{stype}</span> at ${slvl:,} → closed ${sclose:,}'

    impulse_html = "No recent impulse"
    if imp:
        direction_label = "Bullish" if imp["direction"] == "bullish" else "Bearish"
        impulse_html = f'<span style="color:{"#155724" if direction_label=="Bullish" else "#721c24"}; font-weight:600;">{direction_label}</span> impulse {imp["magnitude_pct"]}%'

    zone_html = "Awaiting pullback"
    if zone:
        in_z = zone["in_zone"]
        zone_html = f'${zone["lower"]:,} — ${zone["upper"]:,} (SMA20: ${zone["sma20"]:,})'
        if in_z:
            zone_html += ' <span class="tag-bullish">In Zone ✓</span>'

    highs_str = "  ".join([f"${h:,}" for h in levels.get("highs", [])]) or "—"
    lows_str  = "  ".join([f"${l:,}" for l in levels.get("lows",  [])]) or "—"

    st.markdown(f"""
    <div class="liq-panel">
        <div class="narrative-text" style="margin-bottom:12px; padding:10px; background:#f8f9fa; border-radius:6px; border-left:3px solid #1a1a2e;">
            {narr}
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
            <div>
                <div style="font-size:0.75rem; color:#6c757d; text-transform:uppercase; font-weight:600;">Bias</div>
                <div style="margin-top:4px;">{bias_tag} &nbsp; ${price:,} &nbsp; RSI: {rsi}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#6c757d; text-transform:uppercase; font-weight:600;">Liquidity Sweep</div>
                <div style="margin-top:4px; font-size:0.85rem;">{sweep_html}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#6c757d; text-transform:uppercase; font-weight:600;">Impulse</div>
                <div style="margin-top:4px; font-size:0.85rem;">{impulse_html}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#6c757d; text-transform:uppercase; font-weight:600;">Pullback Zone</div>
                <div style="margin-top:4px; font-size:0.85rem;">{zone_html}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#6c757d; text-transform:uppercase; font-weight:600;">Resistance Levels</div>
                <div style="margin-top:4px; font-size:0.85rem; color:#721c24;">{highs_str}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#6c757d; text-transform:uppercase; font-weight:600;">Support Levels</div>
                <div style="margin-top:4px; font-size:0.85rem; color:#155724;">{lows_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER TABLE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def tier_html(tier: str, text: str) -> str:
    cls = f"signal-{'high' if tier=='HIGH' else 'medium' if tier=='MEDIUM' else 'low'}"
    return f'<span class="{cls}">{text}</span>'


def render_scanner_table(df: pd.DataFrame, direction_filter: str, show_all: bool):
    if df.empty:
        st.info("No signals found. Try refreshing or check exchange connectivity.")
        return

    # Filter by direction
    filtered = df.copy()
    if direction_filter == "Longs":
        filtered = filtered[filtered["direction"] == "Long"]
    elif direction_filter == "Shorts":
        filtered = filtered[filtered["direction"] == "Short"]

    filtered = filtered.sort_values("score", ascending=False)
    display_df = filtered if show_all else filtered.head(10)

    if display_df.empty:
        st.info(f"No {direction_filter.lower()} signals found.")
        return

    # Build clean display dataframe
    rows = []
    for _, row in display_df.iterrows():
        tier      = row.get("tier", "LOW")
        direction = row.get("direction", "—")
        dir_arrow = "↑ Long" if direction == "Long" else "↓ Short"

        conviction_text   = row.get("conviction", "—")
        conviction_detail = conviction_text.split("(")[1].rstrip(")") if "(" in conviction_text else "—"

        rsi_val = row.get("rsi", "—")
        rsi_str = str(rsi_val)
        if isinstance(rsi_val, (int, float)):
            if rsi_val > 70:
                rsi_str = f"{rsi_val} ⚠ High"
            elif rsi_val < 30:
                rsi_str = f"{rsi_val} ⚠ Low"

        room_label  = row.get("room", "—")
        room_reason = row.get("room_reason", "")
        obs_label   = row.get("obstacle", "—")
        obs_reason  = row.get("obs_reason", "")

        tier_icon = "🟢" if tier == "HIGH" else "🟡" if tier == "MEDIUM" else "🔴"

        rows.append({
            "Pair":           f"{row.get('pair','—')} ({row.get('source','').upper()})",
            "Setup":          row.get("setup", "—"),
            "Direction":      dir_arrow,
            "Conviction":     f"{tier_icon} {tier}",
            "Reason":         conviction_detail,
            "15m Bias":       row.get("bias_15m", "—"),
            "RSI":            rsi_str,
            "Room":           f"{room_label} — {room_reason}",
            "Obstacle":       f"{obs_label} — {obs_reason}" if obs_reason else obs_label,
            "Freshness":      row.get("freshness", "—"),
            "Score":          row.get("score", 0),
        })

    display = pd.DataFrame(rows)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pair":       st.column_config.TextColumn("Pair", width="medium"),
            "Setup":      st.column_config.TextColumn("Setup", width="small"),
            "Direction":  st.column_config.TextColumn("Direction", width="small"),
            "Conviction": st.column_config.TextColumn("Conviction", width="small"),
            "Reason":     st.column_config.TextColumn("Why", width="large"),
            "15m Bias":   st.column_config.TextColumn("15m Bias", width="small"),
            "RSI":        st.column_config.TextColumn("RSI", width="small"),
            "Room":       st.column_config.TextColumn("Room to Move", width="medium"),
            "Obstacle":   st.column_config.TextColumn("Obstacle", width="medium"),
            "Freshness":  st.column_config.TextColumn("Freshness", width="small"),
            "Score":      st.column_config.ProgressColumn("Score", min_value=0, max_value=100, width="small"),
        }
    )
    st.caption(f"Showing {len(display_df)} of {len(filtered)} signals · Sorted by score")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Auto-refresh
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=60000, key="scanner_refresh")

    # Header
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.markdown(f'<p class="main-header">📡 Crypto Futures Scanner</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Live scanner · Auto-refreshes every 60s · Last update: {now_utc}</p>',
                unsafe_allow_html=True)

    # Top controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
    with ctrl_col1:
        if st.button("🔄 Scan Now"):
            st.cache_data.clear()
            st.rerun()
    with ctrl_col2:
        st.caption("⚡ Engines: A (3m/5m) + B (BTC 1H)")

    st.divider()

    # ── BTC Regime Box ────────────────────────────────────────────────────
    with st.spinner("Loading BTC regime data..."):
        regime = get_btc_regime_cached()
    render_regime_box(regime)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── BTC Liquidity Panel ───────────────────────────────────────────────
    with st.spinner("Loading BTC liquidity engine..."):
        liq_panel = build_btc_liquidity_panel()
    render_liquidity_panel(liq_panel)

    st.divider()

    # ── Engine A: Scanner Tabs ────────────────────────────────────────────
    st.markdown('<p class="section-title">🔍 Engine A — Multi-Pair Futures Scanner</p>',
                unsafe_allow_html=True)

    tab_3m, tab_5m = st.tabs(["📊 3m Radar", "📊 5m Radar"])

    for tab, tf in [(tab_3m, "3m"), (tab_5m, "5m")]:
        with tab:
            col_filter, col_toggle, _ = st.columns([2, 2, 4])
            with col_filter:
                direction_filter = st.radio(
                    "Direction",
                    ["All", "Longs", "Shorts"],
                    horizontal=True,
                    key=f"dir_{tf}",
                )
            with col_toggle:
                show_all = st.toggle("Show all signals", key=f"all_{tf}")

            with st.spinner(f"Running {tf} scanner across 30+ pairs..."):
                scanner_df = run_scanner(tf)

            # Summary stats
            if not scanner_df.empty:
                high_cnt = len(scanner_df[scanner_df["tier"] == "HIGH"])
                med_cnt  = len(scanner_df[scanner_df["tier"] == "MEDIUM"])
                low_cnt  = len(scanner_df[scanner_df["tier"] == "LOW"])
                long_cnt = len(scanner_df[scanner_df["direction"] == "Long"])
                short_cnt= len(scanner_df[scanner_df["direction"] == "Short"])

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Signals", len(scanner_df))
                m2.metric("🟢 HIGH",   high_cnt)
                m3.metric("🟡 MEDIUM", med_cnt)
                m4.metric("🔴 LOW",    low_cnt)
                m5.metric("↑ Longs / ↓ Shorts", f"{long_cnt} / {short_cnt}")

            render_scanner_table(scanner_df, direction_filter, show_all)

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; color:#6c757d; text-align:center; padding:8px 0;">
        Data sources: OKX · Gate.io · MEXC · Binance (BTC only) · Public REST APIs · No keys required
        &nbsp;|&nbsp; Engine A scans 3m &amp; 5m · Engine B analyses BTC 1H structure
        &nbsp;|&nbsp; All signals are informational only — not financial advice
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
