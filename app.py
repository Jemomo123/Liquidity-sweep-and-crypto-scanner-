Maybe the best app.py to start

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

  .sig-compression { background:#fffbf0; border-left:4px solid #f5a623; border-radius:0 10px 10px 0; padding:14px 16px; margin-bottom:10px; opacity:0.92; }
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
  .badge-watch  { background:#f5a623; color:#fff; border-radius:5px; padding:2px 9px; font-size:0.7rem; font-weight:700; }

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
COMPRESSION_THRESHOLD_4H = 0.004  # 0.4% = 0.40% for 4H timeframe (looser for bigger picture)
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
def okx_top_pairs(n: int = 100) -> list:
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
def gate_top_pairs(n: int = 60) -> list:
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
def mexc_top_pairs(n: int = 50) -> list:
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
    
    # Average volume for volume filter (20 periods)
    if "volume" in df.columns:
        df["avg_volume"] = df["volume"].rolling(20, min_periods=10).mean()
    else:
        df["avg_volume"] = np.nan
    
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

    Current candle only:
    1. Compute CLUSTER_SPREAD_PCT for current candle
    2. compression_active = True if current spread <= 0.20%
    3. Also count how many consecutive candles ending now are in compression
       (used only to distinguish SQZ from building state)
    """
    if df.empty or "sma20" not in df.columns or "sma100" not in df.columns or len(df) < 2:
        return {
            "compression_active": False,
            "spread_pct": None,
            "candles_in_comp": 0,
            "status": "INACTIVE",
        }

    current_spread = cluster_spread_pct(df.iloc[-1])
    in_comp_now    = current_spread <= (COMPRESSION_THRESHOLD * 100)

    # Count consecutive candles in compression ending at current (for SQZ vs building)
    candles_in = 0
    for i in range(len(df) - 1, max(len(df) - COMPRESSION_MIN_CANDLES - 2, -1), -1):
        if cluster_spread_pct(df.iloc[i]) <= (COMPRESSION_THRESHOLD * 100):
            candles_in += 1
        else:
            break

    return {
        "compression_active": in_comp_now,
        "spread_pct":         round(current_spread, 3),
        "candles_in_comp":    candles_in,
        "status":             "ACTIVE" if in_comp_now else "INACTIVE",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — COMPRESSION ENGINE (depends on Nearness Engine)
# ─────────────────────────────────────────────────────────────────────────────

def _in_compression(row) -> bool:
    """Single-candle check using cluster spread formula."""
    return cluster_spread_pct(row) <= (COMPRESSION_THRESHOLD * 100)


def detect_compression_state(df: pd.DataFrame, tf: str = "15m") -> dict:
    """
    Uses Nearness Engine for precision.

    CROSSOVER — report on 1 candle minimum (includes V-shapes and single touches).
    SQZ       — report only when 3+ consecutive candles are in compression.
    
    Args:
        df: DataFrame with OHLCV + indicators
        tf: Timeframe (determines threshold: "4H" uses 0.40%, others use 0.20%)

    Returns: { state, spread_pct, compression_active, candles_in_comp, quality, detail }
    state: 'SQZ' | 'CROSSOVER' | 'NONE'
    """
    if df.empty or "sma20" not in df.columns or len(df) < 2:
        return {
            "state": "NONE", "spread_pct": None,
            "compression_active": False, "candles_in_comp": 0, "quality": "GOOD", "detail": ""
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]
    s20,  s100  = last.get("sma20", np.nan), last.get("sma100", np.nan)
    ps20, ps100 = prev.get("sma20", np.nan), prev.get("sma100", np.nan)
    
    # Determine threshold based on timeframe
    if tf == "4H" or tf == "4h":
        threshold = COMPRESSION_THRESHOLD_4H  # 0.40% for 4H
    else:
        threshold = COMPRESSION_THRESHOLD  # 0.20% for others

    current_spread = cluster_spread_pct(last)
    in_comp_now    = current_spread <= (threshold * 100)

    if not in_comp_now:
        return {
            "state": "NONE",
            "spread_pct": round(current_spread, 3),
            "compression_active": False,
            "candles_in_comp": 0,
            "detail": f"Spread:{round(current_spread,3)}% — INACTIVE",
        }

    # Count consecutive candles in compression from current candle backwards
    candles_in = 0
 
