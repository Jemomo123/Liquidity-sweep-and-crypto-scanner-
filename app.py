"""
=====================================================================
BTC EXPANSION EDGE SCANNER
Streamlit Cloud — Single File app.py
-------------------------------------
Deploy: Push app.py + requirements.txt to GitHub root
        → share.streamlit.io → New App → select repo → Deploy
No API keys required. All public endpoints.
=====================================================================
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
    page_title="Edge Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .stApp { background: #f5f5f0; }

  .card { background:#fff; border-radius:12px; border:1px solid #e4e4e0; padding:16px 18px; margin-bottom:14px; }
  .card-title { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:10px; }

  .state-sqz  { background:#fff7e6; border:1.5px solid #f5a623; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
  .state-cross { background:#e8f4ff; border:1.5px solid #2196f3; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
  .state-label { font-size:0.85rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
  .state-meta  { font-size:0.72rem; color:#666; margin-top:3px; }

  .sig-reversal  { background:#fff0f0; border-left:4px solid #e53935; border-radius:0 10px 10px 0; padding:14px 16px; margin-bottom:10px; }
  .sig-expansion { background:#f0fff4; border-left:4px solid #1db954; border-radius:0 10px 10px 0; padding:14px 16px; margin-bottom:10px; }
  .sig-pullback  { background:#f0f4ff; border-left:4px solid #2196f3; border-radius:0 10px 10px 0; padding:14px 16px; margin-bottom:10px; }

  .sig-type  { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:4px; }
  .sig-title { font-size:1rem; font-weight:700; color:#111; margin-bottom:4px; }
  .sig-long  { color:#1db954; }
  .sig-short { color:#e53935; }
  .sig-body  { font-size:0.78rem; color:#555; margin-top:6px; line-height:1.6; }

  .badge-high   { background:#1db954; color:#fff; border-radius:5px; padding:2px 9px; font-size:0.7rem; font-weight:700; }
  .badge-medium { background:#f5a623; color:#fff; border-radius:5px; padding:2px 9px; font-size:0.7rem; font-weight:700; }
  .badge-low    { background:#e53935; color:#fff; border-radius:5px; padding:2px 9px; font-size:0.7rem; font-weight:700; }

  .tag { display:inline-block; border-radius:4px; padding:2px 8px; font-size:0.67rem; font-weight:600; margin-right:4px; }
  .tag-bull { background:#e8f8ef; color:#1db954; }
  .tag-bear { background:#fde8e8; color:#e53935; }
  .tag-neut { background:#ececec; color:#666; }
  .tag-sqz  { background:#fff7e6; color:#f5a623; }
  .tag-cross { background:#e8f4ff; color:#2196f3; }

  .exch-badge      { background:#1db954; color:#000; font-weight:700; font-size:0.68rem; padding:2px 9px; border-radius:20px; font-family:'IBM Plex Mono',monospace; }
  .exch-badge-fail { background:#e53935; color:#fff; font-weight:700; font-size:0.68rem; padding:2px 9px; border-radius:20px; }

  .log-row { font-family:'IBM Plex Mono',monospace; font-size:0.67rem; color:#555; padding:4px 0; border-bottom:1px solid #f0f0ec; }
  .log-row:last-child { border-bottom:none; }
  .log-sig  { color:#1db954; font-weight:600; }
  .log-comp { color:#f5a623; font-weight:600; }

  .regime-up   { background:#e8f8ef; border:1.5px solid #1db954; border-radius:8px; padding:10px 14px; }
  .regime-down { background:#fde8e8; border:1.5px solid #e53935; border-radius:8px; padding:10px 14px; }
  .regime-range{ background:#fff7e6; border:1.5px solid #f5a623; border-radius:8px; padding:10px 14px; }
  .regime-label{ font-weight:700; font-size:0.82rem; }
  .regime-meta { font-size:0.7rem; color:#666; margin-top:3px; font-family:'IBM Plex Mono',monospace; }

  .stButton > button { background:#111; color:#fff; border:none; border-radius:8px; padding:9px 22px; font-weight:600; font-size:0.82rem; }
  .stButton > button:hover { background:#333; }
  div[data-testid="metric-container"] { background:white; border-radius:8px; border:1px solid #e4e4e0; padding:8px 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COMPRESSION_THRESHOLD = 0.002   # 0.2% = 0.20% cluster spread threshold (Nearness Engine)
COMPRESSION_MIN_CANDLES = 3      # Must persist ≥3 consecutive candles
EXPANSION_BODY_RATIO  = 1.5     # elephant: body ≥ 150% avg body
EXPANSION_WICK_RATIO  = 0.60    # tail: wick ≥ 60% of range
TREND_SMA_SEP         = 0.012   # 1.2% separation for reversal setups
REGIME_SMA_SEP        = 0.010   # 1.0% for regime trending
FIREWALL_DIST         = 0.010   # 1.0% obstacle detection
LIQUIDITY_HOLE_LARGE  = 0.025   # 2.5% = large room
LIQUIDITY_HOLE_MOD    = 0.015   # 1.5% = moderate room
MAX_SIGNAL_AGE        = 2
TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 EdgeScanner/3.0"}

# ─────────────────────────────────────────────────────────────────────────────
# SCAN LOG
# ─────────────────────────────────────────────────────────────────────────────

if "scan_log" not in st.session_state:
    st.session_state.scan_log = []


def add_log(exchange: str, pair: str, comp_state: str, signal: str, conviction: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    st.session_state.scan_log.insert(0, {
        "time": ts, "exchange": exchange, "pair": pair,
        "comp": comp_state, "signal": signal, "conviction": conviction,
    })
    st.session_state.scan_log = st.session_state.scan_log[:20]


# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — MULTI-EXCHANGE DATA FETCHER WITH FAILOVER
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(url: str, params: dict = None) -> Optional[object]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _norm(raw, source: str) -> pd.DataFrame:
    """Normalize any exchange kline format to standard OHLCV dataframe."""
    if not raw:
        return pd.DataFrame()
    try:
        if source in ("binance", "mexc"):
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol",
                                             "ct","qv","n","tbbv","tbqv","ig"][:len(raw[0])])
        elif source == "okx":
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol",
                                             "volCcy","volCcyQuote","confirm"][:len(raw[0])])
            df = df.iloc[::-1].reset_index(drop=True)
        elif source == "gate":
            df = pd.DataFrame(raw)
            df = df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"vol"})
        elif source == "bingx":
            df = pd.DataFrame(raw)
            df = df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"vol"})
        elif source == "weex":
            df = pd.DataFrame(raw)
            df = df.rename(columns={"time":"ts","open":"open","high":"high","low":"low","close":"close","vol":"vol"})
        else:
            return pd.DataFrame()

        for c in ["open","high","low","close","vol"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            if df["ts"].dropna().max() > 1e12:
                df["ts"] = df["ts"] / 1000

        df = df.dropna(subset=["open","high","low","close"]).sort_values("ts").reset_index(drop=True)
        cols = [c for c in ["ts","open","high","low","close","vol"] if c in df.columns]
        return df[cols]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_btc_data_with_failover(interval: str = "1h", limit: int = 60) -> tuple:
    """
    Try Binance → OKX → MEXC → Gate.io → BingX sequentially.
    Returns (DataFrame, active_exchange_name).
    Interval is in standard format: 1m, 3m, 5m, 15m, 1h, 4h.
    """
    iv_binance = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","1h":"1h","4h":"4h"}
    iv_okx     = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","1h":"1H","4h":"4H"}
    iv_mexc    = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","1h":"60m","4h":"4h"}
    iv_gate    = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","1h":"1h","4h":"4h"}
    iv_bingx   = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","1h":"1h","4h":"4h"}

    attempts = [
        ("Binance", lambda: _norm(
            safe_get("https://api.binance.com/api/v3/klines",
                     {"symbol":"BTCUSDT","interval":iv_binance.get(interval,interval),"limit":limit}),
            "binance")),
        ("OKX", lambda: _norm(
            (safe_get("https://www.okx.com/api/v5/market/candles",
                      {"instId":"BTC-USDT-SWAP","bar":iv_okx.get(interval,interval),"limit":limit}) or {}).get("data"),
            "okx")),
        ("MEXC", lambda: _norm(
            safe_get("https://api.mexc.com/api/v3/klines",
                     {"symbol":"BTCUSDT","interval":iv_mexc.get(interval,interval),"limit":limit}),
            "mexc")),
        ("Gate.io", lambda: _norm(
            safe_get("https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                     {"contract":"BTC_USDT","interval":iv_gate.get(interval,interval),"limit":limit}),
            "gate")),
        ("BingX", lambda: _norm(
            (safe_get("https://open-api.bingx.com/openApi/swap/v2/quote/klines",
                      {"symbol":"BTC-USDT","interval":iv_bingx.get(interval,interval),"limit":limit}) or {}).get("data"),
            "bingx")),
    ]

    for name, fetch_fn in attempts:
        try:
            df = fetch_fn()
            if df is not None and not df.empty and len(df) >= 10:
                return df, name
        except Exception:
            continue
    return pd.DataFrame(), "None"


# ── Per-pair kline fetchers ──────────────────────────────────────────────────

@st.cache_data(ttl=60)
def okx_klines(inst_id: str, bar: str, limit: int = 130) -> pd.DataFrame:
    d = safe_get("https://www.okx.com/api/v5/market/candles",
                 {"instId":inst_id,"bar":bar,"limit":limit})
    return _norm((d or {}).get("data"), "okx")


@st.cache_data(ttl=60)
def gate_klines(contract: str, interval: str, limit: int = 130) -> pd.DataFrame:
    d = safe_get("https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                 {"contract":contract,"interval":interval,"limit":limit})
    return _norm(d if isinstance(d, list) else None, "gate")


@st.cache_data(ttl=60)
def mexc_klines(symbol: str, interval: str, limit: int = 130) -> pd.DataFrame:
    d = safe_get("https://api.mexc.com/api/v3/klines",
                 {"symbol":symbol,"interval":interval,"limit":limit})
    return _norm(d if isinstance(d, list) else None, "mexc")


@st.cache_data(ttl=60)
def okx_top_pairs(n: int = 35) -> list:
    tickers = safe_get("https://www.okx.com/api/v5/market/tickers", {"instType":"SWAP"})
    if not tickers or "data" not in tickers:
        return ["BTC-USDT-SWAP"]
    pairs = sorted(
        [(d["instId"], float(d.get("volCcy24h", 0)))
         for d in tickers["data"] if d.get("instId","").endswith("-USDT-SWAP")],
        key=lambda x: x[1], reverse=True
    )
    result = [p[0] for p in pairs]
    if "BTC-USDT-SWAP" not in result[:n]:
        result = ["BTC-USDT-SWAP"] + [p for p in result if p != "BTC-USDT-SWAP"]
    return result[:n]


@st.cache_data(ttl=60)
def gate_top_pairs(n: int = 25) -> list:
    d = safe_get("https://api.gateio.ws/api/v4/futures/usdt/contracts")
    if not isinstance(d, list):
        return []
    pairs = sorted(
        [(x["name"], float(x.get("volume_24h_base", 0)))
         for x in d if "USDT" in x.get("name","")],
        key=lambda x: x[1], reverse=True
    )
    result = [p[0] for p in pairs]
    if "BTC_USDT" not in result[:n]:
        result = ["BTC_USDT"] + [p for p in result if p != "BTC_USDT"]
    return result[:n]


@st.cache_data(ttl=60)
def mexc_top_pairs(n: int = 20) -> list:
    d = safe_get("https://api.mexc.com/api/v3/ticker/24hr")
    if not isinstance(d, list):
        return []
    pairs = sorted(
        [(x["symbol"], float(x.get("quoteVolume", 0)))
         for x in d if str(x.get("symbol","")).endswith("USDT")],
        key=lambda x: x[1], reverse=True
    )
    result = [p[0] for p in pairs]
    if "BTCUSDT" not in result[:n]:
        result = ["BTCUSDT"] + [p for p in result if p != "BTCUSDT"]
    return result[:n]


def get_tf_klines(pair: str, source: str, tf: str, limit: int = 130) -> pd.DataFrame:
    try:
        if source == "okx":
            return okx_klines(pair, tf, limit)
        if source == "gate":
            return gate_klines(pair, tf, limit)
        if source == "mexc":
            return mexc_klines(pair, tf, limit)
    except Exception:
        pass
    return pd.DataFrame()


def pair_display(raw: str, source: str) -> str:
    raw = raw.upper()
    if source == "okx":
        return raw.replace("-SWAP","").replace("-","/")
    return raw.replace("_","/")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    df = df.copy()
    df["sma20"]    = df["close"].rolling(20, min_periods=15).mean()
    df["sma100"]   = df["close"].rolling(100, min_periods=20).mean()
    delta          = df["close"].diff()
    gain           = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss           = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rs             = gain / loss.replace(0, np.nan)
    df["rsi14"]    = 100 - (100 / (1 + rs))
    df["body"]     = (df["close"] - df["open"]).abs()
    df["range"]    = df["high"] - df["low"]
    df["avg_body"] = df["body"].rolling(10, min_periods=5).mean()
    return df


def get_swing_levels(df: pd.DataFrame, lookback: int = 40) -> tuple:
    if df.empty or len(df) < 5:
        return [], []
    sub = df.tail(lookback)
    ha, la = sub["high"].values, sub["low"].values
    highs, lows = [], []
    for i in range(2, len(sub) - 2):
        if ha[i] == max(ha[i-2:i+3]):
            highs.append(ha[i])
        if la[i] == min(la[i-2:i+3]):
            lows.append(la[i])
    return sorted(highs, reverse=True), sorted(lows)


# ─────────────────────────────────────────────────────────────────────────────
# PART 0.5 — NEARNESS ENGINE (FOUNDATION OF EDGE)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_spread_pct(row) -> float:
    """
    CLUSTER SPREAD % = ((max - min) / close) * 100
    Measures total width of the [close, sma20, sma100] cluster.
    """
    p    = row.get("close", np.nan)
    s20  = row.get("sma20",  np.nan)
    s100 = row.get("sma100", np.nan)
    if any(pd.isna(x) for x in [p, s20, s100]) or p == 0:
        return 999.0
    cluster = [p, s20, s100]
    return ((max(cluster) - min(cluster)) / p) * 100


def nearness_engine(df: pd.DataFrame) -> dict:
    """
    NEARNESS ENGINE — Precision compression detector.

    Rules:
    1. Compute CLUSTER_SPREAD_PCT = ((max - min) / close) * 100
       for [close, sma20, sma100]
    2. Compression = spread <= 0.20%
    3. compression_active = True only if last 3 candles are ALL in compression
    4. Returns spread of current candle and active flag.
    """
    if df.empty or "sma20" not in df.columns or "sma100" not in df.columns or len(df) < COMPRESSION_MIN_CANDLES:
        return {
            "compression_active": False,
            "spread_pct": None,
            "candles_in_comp": 0,
            "status": "INACTIVE",
        }

    # Check last N candles all meet threshold
    tail = df.tail(COMPRESSION_MIN_CANDLES)
    spreads = [cluster_spread_pct(tail.iloc[i]) for i in range(len(tail))]
    all_in  = all(s <= COMPRESSION_THRESHOLD * 100 for s in spreads)
    current_spread = spreads[-1]

    # Count consecutive candles currently in compression (from end backwards)
    candles_in = 0
    for s in reversed(spreads):
        if s <= COMPRESSION_THRESHOLD * 100:
            candles_in += 1
        else:
            break

    return {
        "compression_active": all_in,
        "spread_pct":         round(current_spread, 3),
        "candles_in_comp":    candles_in,
        "status":             "ACTIVE" if all_in else "INACTIVE",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — COMPRESSION ENGINE (depends on Nearness Engine)
# ─────────────────────────────────────────────────────────────────────────────

def _in_compression(row) -> bool:
    """Single-candle check using cluster spread formula."""
    return cluster_spread_pct(row) <= (COMPRESSION_THRESHOLD * 100)


def detect_compression_state(df: pd.DataFrame) -> dict:
    """
    Uses Nearness Engine for precision.
    Returns: { state, spread_pct, compression_active, candles_in_comp, detail }
    state: 'SQZ' | 'CROSSOVER' | 'NONE'
    """
    if df.empty or "sma20" not in df.columns or len(df) < 3:
        return {
            "state": "NONE", "spread_pct": None,
            "compression_active": False, "candles_in_comp": 0, "detail": ""
        }

    nearness = nearness_engine(df)

    if not nearness["compression_active"]:
        return {
            "state": "NONE",
            "spread_pct": nearness["spread_pct"],
            "compression_active": False,
            "candles_in_comp": nearness["candles_in_comp"],
            "detail": f"Spread:{nearness['spread_pct']}% — INACTIVE",
        }

    # Compression is active — determine SQZ vs CROSSOVER
    last = df.iloc[-1]
    prev = df.iloc[-2]
    s20, s100   = last.get("sma20", np.nan), last.get("sma100", np.nan)
    ps20, ps100 = prev.get("sma20", np.nan), prev.get("sma100", np.nan)

    is_cross = False
    if not any(pd.isna(x) for x in [ps20, ps100, s20, s100]):
        is_cross = (ps20 > ps100) != (s20 > s100)

    p = last["close"]
    state  = "CROSSOVER" if is_cross else "SQZ"
    detail = (f"P:{p:.4f} SMA20:{s20:.4f} SMA100:{s100:.4f} "
              f"Spread:{nearness['spread_pct']}% "
              f"({nearness['candles_in_comp']} candles)")

    return {
        "state":              state,
        "spread_pct":         nearness["spread_pct"],
        "compression_active": True,
        "candles_in_comp":    nearness["candles_in_comp"],
        "detail":             detail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — EXPANSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _classify_candle(row) -> tuple:
    """Returns (candle_type, direction) or (None, None)."""
    avg_b = row.get("avg_body", np.nan)
    body  = row.get("body", 0)
    rng   = row.get("range", 0)
    if pd.isna(avg_b) or avg_b == 0 or rng == 0:
        return None, None
    # Elephant — large body
    if body >= EXPANSION_BODY_RATIO * avg_b:
        return "elephant", ("long" if row["close"] > row["open"] else "short")
    # Tail — large wick
    wick = rng - body
    if wick >= EXPANSION_WICK_RATIO * rng:
        upper = row["high"] - max(row["open"], row["close"])
        lower = min(row["open"], row["close"]) - row["low"]
        return "tail", ("long" if lower > upper else "short")
    return None, None


def detect_expansion(df: pd.DataFrame) -> Optional[dict]:
    """
    Forward-only expansion detection from compression.

    Rules (from prompt):
    1. Find most recent compression candle in last 6 candles
    2. First candle AFTER compression = breakout candle (monitored)
    3. ENTRY = 1st or 2nd candle after breakout that is elephant or tail
    4. If neither qualifies → no signal

    Returns: { direction, candle_type, signal_age } or None
    """
    if df.empty or len(df) < 5 or "sma20" not in df.columns:
        return None

    sub = df.tail(8).reset_index(drop=True)

    # Find most recent candle that was part of a compression state
    # Use single-candle spread check (Nearness Engine row-level)
    comp_idx = None
    for i in range(len(sub) - 1, -1, -1):
        if _in_compression(sub.iloc[i]):
            comp_idx = i
            break

    # Also require that compression was sustained (≥3 candles) ending at comp_idx
    # Check using full df nearness engine — only proceed if compression was active recently
    full_nearness = nearness_engine(df)
    # Allow expansion detection if compression was active in last scan
    # (nearness may now be INACTIVE because expansion just started — that's correct)
    # We look back to see if it WAS active within last 5 candles
    was_active = False
    for lookback in range(3, min(8, len(df))):
        if nearness_engine(df.iloc[:-lookback] if lookback > 0 else df).get("compression_active"):
            was_active = True
            break
    if not was_active and not full_nearness["compression_active"]:
        return None

    if comp_idx is None:
        return None

    # Candles after compression
    post = sub.iloc[comp_idx + 1:].reset_index(drop=True)
    if len(post) == 0:
        return None

    # Check 1st and 2nd candle after compression
    for i in range(min(2, len(post))):
        ctype, direction = _classify_candle(post.iloc[i])
        if ctype and direction:
            return {
                "direction":   direction,
                "candle_type": ctype,
                "signal_age":  i,  # 0 = current candle, 1 = one candle ago
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — LIQUIDITY ENGINE (1H)
# ─────────────────────────────────────────────────────────────────────────────

def detect_3bar_pivots(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 3:
        return [], []
    ha, la = df["high"].values, df["low"].values
    bsl, ssl = [], []
    for i in range(1, len(df) - 1):
        if ha[i] > ha[i-1] and ha[i] > ha[i+1]:
            bsl.append({"price": ha[i], "idx": i})
        if la[i] < la[i-1] and la[i] < la[i+1]:
            ssl.append({"price": la[i], "idx": i})
    return bsl, ssl


def get_liquidity_sweep(df: pd.DataFrame) -> Optional[dict]:
    """
    BEAR SWEEP: high breaks BSL → closes below
    BULL SWEEP: low breaks SSL → closes above
    """
    if df.empty or len(df) < 6:
        return None

    bsl, ssl = detect_3bar_pivots(df.iloc[:-1])
    last5 = df.tail(5)

    for idx in range(len(last5) - 1, -1, -1):
        row = last5.iloc[idx]
        if bsl:
            top = max(bsl, key=lambda x: x["price"])
            if row["high"] > top["price"] and row["close"] < top["price"]:
                wick_pct = (row["high"] - top["price"]) / top["price"] * 100
                return {
                    "type": "BEAR SWEEP", "level": round(top["price"], 2),
                    "wick_pct": round(wick_pct, 3),
                    "bars_ago": len(last5) - 1 - idx,
                    "confirmed": True,
                }
        if ssl:
            bot = min(ssl, key=lambda x: x["price"])
            if row["low"] < bot["price"] and row["close"] > bot["price"]:
                wick_pct = (bot["price"] - row["low"]) / bot["price"] * 100
                return {
                    "type": "BULL SWEEP", "level": round(bot["price"], 2),
                    "wick_pct": round(wick_pct, 3),
                    "bars_ago": len(last5) - 1 - idx,
                    "confirmed": True,
                }
    return None


def get_liquidity_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"bsl": [], "ssl": [], "price": None}
    bsl, ssl = detect_3bar_pivots(df)
    price    = df.iloc[-1]["close"]
    above    = sorted([x["price"] for x in bsl if x["price"] > price])[:3]
    below    = sorted([x["price"] for x in ssl if x["price"] < price], reverse=True)[:3]
    return {
        "bsl":   [round(p, 2) for p in above],
        "ssl":   [round(p, 2) for p in below],
        "price": round(price, 2),
    }


def get_1h_bias(df: pd.DataFrame) -> str:
    if df.empty or "sma20" not in df.columns:
        return "neutral"
    last = df.iloc[-1]
    s20, s100 = last.get("sma20", np.nan), last.get("sma100", np.nan)
    if pd.isna(s20) or pd.isna(s100):
        return "neutral"
    if s20 > s100 * 1.002:
        return "bullish"
    if s20 < s100 * 0.998:
        return "bearish"
    return "neutral"


def get_impulse_1h(df: pd.DataFrame) -> Optional[dict]:
    if df.empty or len(df) < 5:
        return None
    for i in range(len(df) - 1, max(len(df) - 7, -1), -1):
        row = df.iloc[i]
        avg_b = row.get("avg_body", np.nan)
        body  = row.get("body", 0)
        if not pd.isna(avg_b) and avg_b > 0 and body >= EXPANSION_BODY_RATIO * avg_b:
            return {
                "direction": "bullish" if row["close"] > row["open"] else "bearish",
                "pct":       round(body / row["close"] * 100, 2),
                "bars_ago":  len(df) - 1 - i,
            }
    return None


def get_pullback_zone(df: pd.DataFrame) -> Optional[dict]:
    if df.empty or "sma20" not in df.columns:
        return None
    last = df.iloc[-1]
    s20  = last.get("sma20", np.nan)
    if pd.isna(s20):
        return None
    price = last["close"]
    lower = round(s20 * 0.9962, 2)
    upper = round(s20 * 1.0038, 2)
    return {
        "lower": lower, "upper": upper,
        "sma20": round(s20, 2),
        "in_zone": lower <= price <= upper,
    }


@st.cache_data(ttl=60)
def build_liquidity_panel() -> dict:
    df_raw, active = get_btc_data_with_failover("1h", 100)
    if df_raw.empty:
        return {"error": "All exchange sources failed. Check network.", "active": "None"}
    df      = add_indicators(df_raw)
    sweep   = get_liquidity_sweep(df)
    levels  = get_liquidity_levels(df)
    bias    = get_1h_bias(df)
    impulse = get_impulse_1h(df)
    zone    = get_pullback_zone(df)
    rsi_val = df.iloc[-1].get("rsi14", np.nan)
    return {
        "active": active, "bias": bias, "sweep": sweep,
        "levels": levels, "impulse": impulse, "zone": zone,
        "price":  levels.get("price"),
        "rsi":    round(rsi_val, 1) if not pd.isna(rsi_val) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — POST-EXPANSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def assess_trend_health(df: pd.DataFrame, direction: str) -> dict:
    """5 metrics → HEALTHY / EXHAUSTED / MIXED."""
    if df.empty or len(df) < 10:
        return {"health": "UNKNOWN", "score": 0, "metrics": []}

    metrics = []
    last    = df.iloc[-1]
    recent  = df.tail(5)

    # 1. Structure continuation
    if direction == "long":
        metrics.append(("Structure", "healthy" if recent["high"].iloc[-1] > recent["high"].iloc[0] else "weak"))
    else:
        metrics.append(("Structure", "healthy" if recent["low"].iloc[-1] < recent["low"].iloc[0] else "weak"))

    # 2. SMA20 respect
    s20 = last.get("sma20", np.nan)
    if not pd.isna(s20):
        ok = last["close"] > s20 if direction == "long" else last["close"] < s20
        metrics.append(("SMA20 respect", "healthy" if ok else "weak"))
    else:
        metrics.append(("SMA20 respect", "weak"))

    # 3. Candle body quality
    avg_r = recent["body"].mean()
    avg_a = df["body"].tail(20).mean() if len(df) >= 20 else avg_r
    metrics.append(("Body quality", "healthy" if avg_r >= avg_a * 0.8 else "weak"))

    # 4. Momentum (RSI)
    rsi = last.get("rsi14", np.nan)
    if not pd.isna(rsi):
        ok = 45 <= rsi <= 70 if direction == "long" else 30 <= rsi <= 55
        metrics.append(("Momentum", "healthy" if ok else "weak"))
    else:
        metrics.append(("Momentum", "weak"))

    # 5. Opposite liquidity swept
    highs, lows = get_swing_levels(df, 20)
    opp_swept = False
    if direction == "long" and highs:
        opp_swept = recent["high"].max() >= highs[0] * 0.998
    elif direction == "short" and lows:
        opp_swept = recent["low"].min() <= lows[0] * 1.002
    metrics.append(("Opp. liq swept", "weak" if opp_swept else "healthy"))

    healthy = sum(1 for _, s in metrics if s == "healthy")
    weak    = sum(1 for _, s in metrics if s == "weak")
    health  = "HEALTHY" if healthy >= 3 else "EXHAUSTED" if weak >= 3 else "MIXED"
    return {"health": health, "score": healthy, "metrics": metrics}


def detect_pullback(df: pd.DataFrame) -> Optional[dict]:
    """Pullback to SMA20 with elephant/tail confirmation candle."""
    if df.empty or len(df) < 10 or "sma20" not in df.columns:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    s20p = prev.get("sma20", np.nan)
    s20l = last.get("sma20", np.nan)
    if pd.isna(s20p) or pd.isna(s20l):
        return None

    low_touch  = prev["low"]  <= s20p * 1.005 and prev["close"] > s20p
    high_touch = prev["high"] >= s20p * 0.995 and prev["close"] < s20p

    ctype, cdir = _classify_candle(last)
    if not ctype:
        ctype, cdir = _classify_candle(prev)

    if low_touch and ctype and last["close"] > s20l:
        return {"direction": "long", "candle_type": ctype}
    if high_touch and ctype and last["close"] < s20l:
        return {"direction": "short", "candle_type": ctype}
    return None


def detect_reversal(df: pd.DataFrame) -> Optional[dict]:
    """
    Late reversal after exhausted trend.
    Requires: SMA sep ≥1.2%, no crossing last 15 candles,
              price near SMA100, elephant/tail confirmation.
    """
    if df.empty or len(df) < 20 or "sma20" not in df.columns:
        return None
    last = df.iloc[-1]
    s20, s100 = last.get("sma20", np.nan), last.get("sma100", np.nan)
    if pd.isna(s20) or pd.isna(s100) or s100 == 0:
        return None

    sep = abs(s20 - s100) / s100
    if sep < TREND_SMA_SEP:
        return None

    # No crossing in last 15 candles
    sub   = df.tail(15)
    above = (sub["sma20"] > sub["sma100"]).values
    if any(above[i] != above[i-1] for i in range(1, len(above))):
        return None

    # Price near SMA100
    if abs(last["close"] - s100) / s100 > 0.006:
        return None

    ctype, cdir = _classify_candle(last)
    if not ctype:
        ctype, cdir = _classify_candle(df.iloc[-2])
    if not ctype:
        return None

    direction = "long" if s20 > s100 else "short"
    return {"direction": direction, "candle_type": ctype}


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — CONVICTION SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(signal_age: int, candle_type: str, bias_15m: str,
                 direction: str, room: str, obstacle: str, rsi: float) -> tuple:
    score, parts = 0, []

    # Freshness 25
    if signal_age == 0:
        score += 25; parts.append("Fresh compression")
    elif signal_age == 1:
        score += 12; parts.append("1 candle ago")

    # Candle 20
    if candle_type == "elephant":
        score += 20; parts.append("Strong elephant candle")
    elif candle_type == "tail":
        score += 15; parts.append("Rejection tail")

    # Bias 15
    if (direction == "long" and bias_15m == "bullish") or \
       (direction == "short" and bias_15m == "bearish"):
        score += 15; parts.append("Structure aligned")
    elif bias_15m == "neutral":
        score += 5

    # Room 15
    if room == "Large":
        score += 15; parts.append("Large room ahead")
    elif room == "Moderate":
        score += 8;  parts.append("Moderate room")
    else:
        parts.append("Limited room")

    # Obstacle 15
    if obstacle == "None":
        score += 15; parts.append("No obstacles")
    else:
        parts.append("Obstacle present")

    # RSI 10
    if not (isinstance(rsi, str)) and not pd.isna(rsi):
        if (direction == "long" and rsi < 40) or (direction == "short" and rsi > 60):
            score += 10; parts.append("RSI in fuel zone")
        elif 40 <= rsi <= 60:
            score += 5; parts.append("RSI neutral")

    tier = "HIGH" if score >= 75 else "MEDIUM" if score >= 55 else "LOW"
    return score, tier, parts


def check_room_obstacle(df: pd.DataFrame, direction: str) -> tuple:
    if df.empty:
        return "Unknown", "", "None", "No data"
    price  = df.iloc[-1]["close"]
    highs, lows = get_swing_levels(df, 40)

    if direction == "long":
        ahead = [h for h in highs if h > price]
        if ahead:
            d = (ahead[0] - price) / price
            room_d = f"Next swing {d*100:.1f}% away"
            room   = "Large" if d >= LIQUIDITY_HOLE_LARGE else ("Moderate" if d >= LIQUIDITY_HOLE_MOD else "Limited")
        else:
            room, room_d = "Large", "No major swing overhead"
        near = [h for h in highs if 0 < (h - price) / price <= FIREWALL_DIST]
        if near:
            d = (near[0] - price) / price * 100
            return room, room_d, "Resistance", f"Swing high {d:.1f}% above"
    else:
        ahead = [l for l in lows if l < price]
        if ahead:
            d = (price - ahead[0]) / price
            room_d = f"Next swing {d*100:.1f}% away"
            room   = "Large" if d >= LIQUIDITY_HOLE_LARGE else ("Moderate" if d >= LIQUIDITY_HOLE_MOD else "Limited")
        else:
            room, room_d = "Large", "No major swing below"
        near = [l for l in lows if 0 < (price - l) / price <= FIREWALL_DIST]
        if near:
            d = (price - near[0]) / price * 100
            return room, room_d, "Support", f"Swing low {d:.1f}% below"

    return room, room_d, "None", "No swing within 1%"


def get_15m_bias(pair: str, source: str) -> str:
    df = get_tf_klines(pair, source, "15m", 130)
    if df.empty or len(df) < 20:
        return "neutral"
    df = add_indicators(df)
    last = df.iloc[-1]
    s20, s100 = last.get("sma20", np.nan), last.get("sma100", np.nan)
    if pd.isna(s20) or pd.isna(s100):
        return "neutral"
    if s20 > s100 * 1.002:
        return "bullish"
    if s20 < s100 * 0.998:
        return "bearish"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# BTC REGIME
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_btc_regime() -> dict:
    result = {}
    for label, interval, limit in [("15m","15m",130),("1H","1h",60),("4H","4h",60)]:
        df, active = get_btc_data_with_failover(interval, limit)
        if df.empty or len(df) < 20:
            result[label] = {"regime":"Unknown","dir":"—","sep":0,"price":"—","active":active}
            continue
        df   = add_indicators(df)
        last = df.iloc[-1]
        s20, s100 = last.get("sma20", np.nan), last.get("sma100", np.nan)
        if pd.isna(s20) or pd.isna(s100) or s100 == 0:
            result[label] = {"regime":"Unknown","dir":"—","sep":0,"price":round(last["close"],2),"active":active}
            continue
        sep = (s20 - s100) / s100
        result[label] = {
            "regime": "Trending" if abs(sep) >= REGIME_SMA_SEP else "Ranging",
            "dir":    "Up" if sep > 0 else "Down",
            "sep":    round(abs(sep)*100, 2),
            "price":  round(last["close"], 2),
            "sma20":  round(s20, 2),
            "sma100": round(s100, 2),
            "active": active,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PAIR SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan_pair(pair: str, source: str, tf: str) -> Optional[dict]:
    try:
        df = get_tf_klines(pair, source, tf, 130)
        if df.empty or len(df) < 25:
            return None
        df = add_indicators(df)

        price = df.iloc[-1]["close"]
        rsi   = df.iloc[-1].get("rsi14", np.nan)

        # Always check compression state
        comp = detect_compression_state(df)

        signal_type = None
        direction   = None
        candle_type = None
        signal_age  = 0
        health_info = None

        # Priority 1: Reversal after exhaustion
        rev = detect_reversal(df)
        if rev:
            health = assess_trend_health(df, rev["direction"])
            if health["health"] == "EXHAUSTED":
                signal_type = "REVERSAL"
                direction   = rev["direction"]
                candle_type = rev["candle_type"]
                health_info = health

        # Priority 2: Fresh expansion from compression
        if not signal_type:
            exp = detect_expansion(df)
            if exp:
                bias_15m = get_15m_bias(pair, source)
                # Drop contradictory bias
                if exp["direction"] == "long"  and bias_15m == "bearish": pass
                elif exp["direction"] == "short" and bias_15m == "bullish": pass
                else:
                    signal_type = "EXPANSION"
                    direction   = exp["direction"]
                    candle_type = exp["candle_type"]
                    signal_age  = exp["signal_age"]

        # Priority 3: Pullback continuation
        if not signal_type:
            pb = detect_pullback(df)
            if pb:
                bias_15m = get_15m_bias(pair, source)
                if pb["direction"] == "long"  and bias_15m == "bearish": pass
                elif pb["direction"] == "short" and bias_15m == "bullish": pass
                else:
                    signal_type = "PULLBACK"
                    direction   = pb["direction"]
                    candle_type = pb["candle_type"]

        # Log compression even without signal
        if not signal_type:
            if comp["state"] != "NONE":
                add_log(source.upper(), pair_display(pair, source),
                        comp["state"], "—", "—")
            return None

        if signal_age > MAX_SIGNAL_AGE:
            return None

        bias_15m = get_15m_bias(pair, source)

        # Final contradictory filter
        if signal_type in ("EXPANSION","PULLBACK"):
            if direction == "long"  and bias_15m == "bearish": return None
            if direction == "short" and bias_15m == "bullish": return None

        room, room_d, obs, obs_d = check_room_obstacle(df, direction)

        # Filter: less than 0.5% room
        if room == "Limited":
            try:
                d = float(room_d.split("Next swing ")[1].split("%")[0])
                if d < 0.5:
                    return None
            except Exception:
                pass

        score, tier, parts = score_signal(signal_age, candle_type, bias_15m,
                                          direction, room, obs, rsi)
        if score < 30:
            return None

        disp      = pair_display(pair, source)
        freshness = "New" if signal_age == 0 else f"{signal_age} candle{'s' if signal_age > 1 else ''} ago"

        add_log(source.upper(), disp, comp["state"],
                f"{signal_type} {direction.upper()}", tier)

        return {
            "pair":          disp,
            "raw_pair":      pair,
            "source":        source,
            "tf":            tf,
            "signal_type":   signal_type,
            "direction":     direction.capitalize(),
            "candle_type":   candle_type or "—",
            "score":         score,
            "tier":          tier,
            "parts":         parts,
            "bias_15m":      bias_15m.capitalize(),
            "rsi":           round(rsi, 1) if not pd.isna(rsi) else "—",
            "room":          room,
            "room_d":        room_d,
            "obs":           obs,
            "obs_d":         obs_d,
            "freshness":     freshness,
            "signal_age":    signal_age,
            "comp_state":    comp["state"],
            "spread_pct":    comp.get("spread_pct"),
            "candles_in_comp": comp.get("candles_in_comp", 0),
            "price":         round(price, 4),
            "health":        health_info,
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def run_scanner(tf: str, exchanges: list = None) -> pd.DataFrame:
    if exchanges is None:
        exchanges = ["okx", "gate", "mexc"]

    pair_sources = []
    if "okx" in exchanges:
        for p in okx_top_pairs(28):
            pair_sources.append((p, "okx"))
    if "gate" in exchanges:
        for p in gate_top_pairs(18):
            nm = pair_display(p, "gate")
            if nm not in {pair_display(x[0], x[1]) for x in pair_sources}:
                pair_sources.append((p, "gate"))
    if "mexc" in exchanges:
        for p in mexc_top_pairs(12):
            nm = pair_display(p, "mexc")
            if nm not in {pair_display(x[0], x[1]) for x in pair_sources}:
                pair_sources.append((p, "mexc"))

    if not pair_sources:
        return pd.DataFrame()

    pair_sources = pair_sources[:50]
    results  = []
    progress = st.empty()

    for i, (pair, source) in enumerate(pair_sources):
        progress.caption(f"Scanning {i+1}/{len(pair_sources)}: {pair_display(pair, source)}")
        sig = scan_pair(pair, source, tf)
        if sig:
            results.append(sig)
        time.sleep(0.04)

    progress.empty()
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    pri = {"REVERSAL": 0, "EXPANSION": 1, "PULLBACK": 2}
    df["_p"] = df["signal_type"].map(pri).fillna(3)
    df = df.sort_values(["_p", "score"], ascending=[True, False]).drop("_p", axis=1)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def render_regime(regime: dict):
    st.markdown('<div class="card-title">BTC MARKET REGIME</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (tf, label) in enumerate([("15m","15-Min"),("1H","1-Hour"),("4H","4-Hour")]):
        with cols[i]:
            r = regime.get(tf, {})
            reg, dirn, sep, price = r.get("regime","Unknown"), r.get("dir","—"), r.get("sep",0), r.get("price","—")
            if reg == "Trending" and dirn == "Up":
                css, icon = "regime-up",   "▲"
            elif reg == "Trending" and dirn == "Down":
                css, icon = "regime-down", "▼"
            else:
                css, icon = "regime-range","◆"
            active = r.get("active","—")
            st.markdown(f"""
            <div class="{css}">
              <div class="regime-label">{icon} {label} — {reg}</div>
              <div class="regime-meta">Dir:{dirn} | Gap:{sep}% | ${price:,} | {active}</div>
            </div>""", unsafe_allow_html=True)


def render_liquidity(panel: dict):
    if "error" in panel:
        st.warning(f"⚠ Liquidity engine unavailable: {panel['error']}")
        return

    active  = panel.get("active","—")
    bias    = panel.get("bias","neutral")
    price   = panel.get("price","—")
    rsi     = panel.get("rsi","—")
    sweep   = panel.get("sweep")
    imp     = panel.get("impulse")
    zone    = panel.get("zone")
    levels  = panel.get("levels", {})

    bias_tag = (f'<span class="tag tag-bull">Bullish</span>' if bias == "bullish" else
                f'<span class="tag tag-bear">Bearish</span>' if bias == "bearish" else
                f'<span class="tag tag-neut">Neutral</span>')

    sweep_html = "None detected"
    if sweep:
        col = "#1db954" if "BULL" in sweep["type"] else "#e53935"
        sweep_html = (f'<span style="color:{col};font-weight:700;">{sweep["type"]}</span> '
                      f'@ ${sweep["level"]:,} | Wick:{sweep["wick_pct"]}% | '
                      f'Confirmed:{"✓" if sweep["confirmed"] else "✗"} | '
                      f'{sweep["bars_ago"]} bars ago')

    imp_html = "No recent impulse"
    if imp:
        col = "#1db954" if imp["direction"] == "bullish" else "#e53935"
        imp_html = (f'<span style="color:{col};font-weight:700;">'
                    f'{imp["direction"].capitalize()}</span> {imp["pct"]}% · {imp["bars_ago"]} bars ago')

    zone_html = "—"
    if zone:
        zone_html = f"${zone['lower']:,} – ${zone['upper']:,}"
        if zone["in_zone"]:
            zone_html += ' <span class="tag tag-bull">In Zone ✓</span>'

    bsl_str = " | ".join([f"${p:,}" for p in levels.get("bsl", [])]) or "—"
    ssl_str = " | ".join([f"${p:,}" for p in levels.get("ssl", [])]) or "—"

    st.markdown(f"""
    <div class="card">
      <div class="card-title">BTC LIQUIDITY ENGINE (1H) &nbsp;
        <span class="exch-badge">{active}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:0.82rem;">
        <div>
          <div style="font-size:0.67rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Bias & Price</div>
          <div>{bias_tag} &nbsp;<b>${price:,}</b> &nbsp;RSI:{rsi}</div>
        </div>
        <div>
          <div style="font-size:0.67rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Liquidity Sweep</div>
          <div>{sweep_html}</div>
        </div>
        <div>
          <div style="font-size:0.67rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:4px;">1H Impulse</div>
          <div>{imp_html}</div>
        </div>
        <div>
          <div style="font-size:0.67rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Pullback Zone (SMA20 ±0.38%)</div>
          <div>{zone_html}</div>
        </div>
        <div>
          <div style="font-size:0.67rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Buy-Side Liq (BSL)</div>
          <div style="color:#e53935;">{bsl_str}</div>
        </div>
        <div>
          <div style="font-size:0.67rem;color:#888;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Sell-Side Liq (SSL)</div>
          <div style="color:#1db954;">{ssl_str}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_signal_card(row: dict):
    sig   = row.get("signal_type","")
    dirn  = row.get("direction","")
    tier  = row.get("tier","LOW")
    parts = row.get("parts", [])
    comp  = row.get("comp_state","")
    health = row.get("health")

    css = {"REVERSAL":"sig-reversal","EXPANSION":"sig-expansion","PULLBACK":"sig-pullback"}.get(sig,"sig-expansion")
    dir_cls = "sig-long" if dirn == "Long" else "sig-short"
    arr     = "▲" if dirn == "Long" else "▼"

    badge = (f'<span class="badge-high">HIGH</span>'   if tier == "HIGH" else
             f'<span class="badge-medium">MEDIUM</span>' if tier == "MEDIUM" else
             f'<span class="badge-low">LOW</span>')

    score_bar = f'{row.get("score",0)}/100'

    comp_tag = ""
    if comp == "SQZ":
        comp_tag = '<span class="tag tag-sqz">SQZ</span>'
    elif comp == "CROSSOVER":
        comp_tag = '<span class="tag tag-cross">CROSSOVER</span>'

    bias = row.get("bias_15m","")
    bias_tag = (f'<span class="tag tag-bull">15m Bullish</span>' if "bullish" in bias.lower() else
                f'<span class="tag tag-bear">15m Bearish</span>' if "bearish" in bias.lower() else
                f'<span class="tag tag-neut">15m Neutral</span>')

    health_html = ""
    if health:
        hcol = "#1db954" if health["health"]=="HEALTHY" else "#e53935" if health["health"]=="EXHAUSTED" else "#f5a623"
        health_html = f'<span style="color:{hcol};font-weight:700;font-size:0.72rem;">Trend:{health["health"]} ({health["score"]}/5)</span> &nbsp;'

    obs_html = (f'✓ No obstacle' if row.get("obs","None") == "None"
                else f'⚠ {row.get("obs","")} — {row.get("obs_d","")}')

    reason = " • ".join(parts)

    # Nearness Engine debug display
    spread_pct   = row.get("spread_pct")
    candles_in   = row.get("candles_in_comp", 0)
    spread_str   = f"{spread_pct:.3f}%" if spread_pct is not None else "—"
    spread_col   = "#1db954" if (spread_pct is not None and spread_pct <= 0.20) else "#e53935"
    comp_status  = "ACTIVE" if row.get("comp_state") not in ("NONE","") else "INACTIVE"
    comp_s_col   = "#1db954" if comp_status == "ACTIVE" else "#888"
    nearness_html = (
        f'<span style="font-size:0.68rem;font-family:IBM Plex Mono,monospace;">' +
        f'CLUSTER SPREAD: <b style="color:{spread_col};">{spread_str}</b>' +
        f' &nbsp;|' +
        f' COMPRESSION: <b style="color:{comp_s_col};">{comp_status}</b>' +
        f' ({candles_in} candles)</span>'
    )

    st.markdown(f"""
    <div class="{css}">
      <div class="sig-type">{sig} &nbsp;{comp_tag}&nbsp;{badge}&nbsp;{score_bar}</div>
      <div class="sig-title">
        <span class="{dir_cls}">{arr} {dirn}</span> &nbsp;
        {row.get('pair','')}
        <span style="color:#888;font-weight:400;font-size:0.82rem;">
          &nbsp;{row.get('tf','')} · ${row.get('price','')} · RSI {row.get('rsi','')}
        </span>
      </div>
      <div class="sig-body">
        {nearness_html}
        <br>{health_html}{bias_tag}
        &nbsp; Candle: <b>{row.get('candle_type','—')}</b>
        &nbsp; Freshness: <b>{row.get('freshness','—')}</b>
        <br>Room: <b>{row.get('room','—')}</b> — {row.get('room_d','')}
        <br><span style="color:#888;">{obs_html}</span>
        <br><span style="color:#777;font-size:0.74rem;">Reason: {reason}</span>
      </div>
    </div>""", unsafe_allow_html=True)


def render_scan_log():
    if not st.session_state.scan_log:
        st.caption("No scans yet — run scanner above.")
        return
    rows = ""
    for e in st.session_state.scan_log:
        sig_cls  = "log-sig"  if e["signal"] not in ("—","") else ""
        comp_cls = "log-comp" if e["comp"]   not in ("NONE","—","") else ""
        rows += (f'<div class="log-row">'
                 f'{e["time"]} &nbsp; <b>{e["exchange"]}</b> &nbsp; {e["pair"]} &nbsp;'
                 f'<span class="{comp_cls}">{e["comp"]}</span> &nbsp;'
                 f'<span class="{sig_cls}">{e["signal"]}</span> &nbsp;'
                 f'{e["conviction"]}</div>')
    st.markdown(f'<div class="card" style="padding:12px 16px;">{rows}</div>', unsafe_allow_html=True)


def render_scanner(df: pd.DataFrame, dir_filter: str, show_all: bool):
    if df.empty:
        st.markdown("""
        <div class="card" style="text-align:center;padding:28px;color:#888;">
          No signals found right now. Markets may be in compression — that is the setup. Watch for the breakout.
        </div>""", unsafe_allow_html=True)
        return

    filtered = df.copy()
    if dir_filter == "Longs":
        filtered = filtered[filtered["direction"] == "Long"]
    elif dir_filter == "Shorts":
        filtered = filtered[filtered["direction"] == "Short"]

    display = filtered if show_all else filtered.head(10)

    if display.empty:
        st.info(f"No {dir_filter.lower()} signals at this moment.")
        return

    for _, row in display.iterrows():
        render_signal_card(row.to_dict())

    st.caption(f"Showing {len(display)} of {len(filtered)} · Priority: Reversal > Expansion > Pullback · Sorted by score")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=60000, key="edge_refresh")

    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC · %d %b %Y")

    # ── Header row ───────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns([3, 2, 3, 1])
    with h1:
        st.markdown(f"""
        <div style="padding:6px 0;">
          <span style="font-size:1.15rem;font-weight:800;color:#111;">📡 Edge Scanner</span>
          <span style="font-size:0.7rem;color:#888;font-family:'IBM Plex Mono',monospace;margin-left:12px;">{now_utc}</span>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.selectbox("BTC Source", ["Auto (failover)","Binance","OKX","Gate.io","MEXC","BingX"],
                     index=0, key="btc_src", label_visibility="collapsed")
    with h3:
        exchanges = st.multiselect("Scan Exchanges", ["OKX","Gate.io","MEXC"],
                                    default=["OKX","Gate.io","MEXC"],
                                    key="scan_ex", label_visibility="collapsed")
        if not exchanges:
            exchanges = ["OKX"]
    with h4:
        if st.button("⟳ Scan Now"):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # ── BTC Regime ───────────────────────────────────────────────────────────
    with st.spinner("Loading BTC regime..."):
        regime = get_btc_regime()
    render_regime(regime)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── BTC Liquidity Panel ──────────────────────────────────────────────────
    with st.spinner("Loading BTC liquidity engine..."):
        liq = build_liquidity_panel()
    render_liquidity(liq)

    st.divider()

    # ── Multi-pair scanner ───────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.9rem;font-weight:800;color:#111;margin-bottom:10px;">🔍 MULTI-PAIR SCANNER</div>', unsafe_allow_html=True)

    exch_keys = []
    for e in exchanges:
        k = e.lower().replace(".","").replace("io","")
        k = "gate" if "gat" in k else k
        exch_keys.append(k)

    tab_3m, tab_5m = st.tabs(["📊 3m Radar", "📊 5m Radar"])

    for tab, tf in [(tab_3m, "3m"), (tab_5m, "5m")]:
        with tab:
            fc, tc, _ = st.columns([2, 2, 4])
            with fc:
                dir_filter = st.radio("Direction", ["All","Longs","Shorts"],
                                       horizontal=True, key=f"dir_{tf}")
            with tc:
                show_all = st.toggle("Show all signals", key=f"all_{tf}")

            with st.spinner(f"Scanning {tf} across {len(pair_sources_preview(exch_keys))} pairs..."):
                scan_df = run_scanner(tf, exchanges=exch_keys)

            if not scan_df.empty:
                h  = len(scan_df[scan_df["tier"]=="HIGH"])
                m  = len(scan_df[scan_df["tier"]=="MEDIUM"])
                l  = len(scan_df[scan_df["tier"]=="LOW"])
                lo = len(scan_df[scan_df["direction"]=="Long"])
                sh = len(scan_df[scan_df["direction"]=="Short"])
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Signals",    len(scan_df))
                c2.metric("🟢 HIGH",    h)
                c3.metric("🟡 MEDIUM",  m)
                c4.metric("🔴 LOW",     l)
                c5.metric("▲ / ▼",      f"{lo} / {sh}")

            render_scanner(scan_df, dir_filter, show_all)

    # ── Scan log ─────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Scan Log — last 20 events", expanded=False):
        render_scan_log()

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.7rem;color:#bbb;text-align:center;padding:10px 0;">
      Sources: Binance · OKX · Gate.io · MEXC · BingX · All public REST APIs · No keys required
      &nbsp;|&nbsp; Signals are informational only — not financial advice
    </div>""", unsafe_allow_html=True)


def pair_sources_preview(exch_keys: list) -> list:
    """Return estimated pair count for spinner message."""
    est = 0
    if "okx"  in exch_keys: est += 28
    if "gate" in exch_keys: est += 18
    if "mexc" in exch_keys: est += 12
    return list(range(min(est, 50)))


if __name__ == "__main__":
    main()
