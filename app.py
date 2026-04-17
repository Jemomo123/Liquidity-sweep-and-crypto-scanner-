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
    for i in range(len(df) - 1, max(len(df) - 20, -1), -1):
        if cluster_spread_pct(df.iloc[i]) <= (threshold * 100):
            candles_in += 1
        else:
            break

    # Detect CROSSOVER:
    # ONLY valid when cluster spread is already within threshold (price+SMA20+SMA100 together)
    # Then check if SMA20 crossed SMA100 this candle OR they are within 0.05% of each other
    is_cross = False
    if in_comp_now and not any(pd.isna(x) for x in [ps20, ps100, s20, s100]):
        crossed_this_candle = (ps20 > ps100) != (s20 > s100)
        sma_gap = abs(s20 - s100) / last["close"] * 100 if last["close"] > 0 else 999
        already_together = sma_gap <= 0.05
        is_cross = crossed_this_candle or already_together

    if is_cross:
        # CROSSOVER — all three together AND SMA20/SMA100 crossing or touching
        state = "CROSSOVER"
        compression_active = True
    elif candles_in >= COMPRESSION_MIN_CANDLES:
        # SQZ — 3+ consecutive candles all within threshold
        state = "SQZ"
        compression_active = True
    else:
        # Not in compression — stay silent
        return {
            "state": "NONE",
            "spread_pct": round(current_spread, 3),
            "compression_active": False,
            "candles_in_comp": candles_in,
            "detail": f"Spread:{round(current_spread,3)}% — building ({candles_in}/3 candles)",
        }

    p = last["close"]
    detail = (f"P:{p:.4f} SMA20:{s20:.4f} SMA100:{s100:.4f} "
              f"Spread:{round(current_spread,3)}% ({candles_in} candles)")

    # Determine compression quality tier (adjusted for timeframe)
    if tf == "4H" or tf == "4h":
        # 4H quality tiers (looser)
        if current_spread <= 0.20:
            quality = "ELITE"
        elif current_spread <= 0.30:
            quality = "HIGH"
        else:
            quality = "GOOD"
    else:
        # Lower timeframes (3m, 5m, 15m)
        if current_spread <= 0.10:
            quality = "ELITE"
        elif current_spread <= 0.15:
            quality = "HIGH"
        else:
            quality = "GOOD"
    
    return {
        "state":              state,
        "spread_pct":         round(current_spread, 3),
        "compression_active": compression_active,
        "candles_in_comp":    candles_in,
        "quality":            quality,
        "detail":             detail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — EXPANSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _classify_candle(row) -> tuple:
    """
    Returns (candle_type, direction) or (None, None).
    
    Volume Filter: Candle must have volume ≥120% of average to be valid.
    This filters out weak breakouts with low participation.
    """
    avg_b = row.get("avg_body", np.nan)
    avg_v = row.get("avg_volume", np.nan)
    body  = row.get("body", 0)
    rng   = row.get("range", 0)
    vol   = row.get("volume", np.nan)
    
    if pd.isna(avg_b) or avg_b == 0 or rng == 0:
        return None, None
    
    # VOLUME FILTER: Must have above-average volume
    if not pd.isna(avg_v) and not pd.isna(vol) and avg_v > 0:
        if vol < avg_v * 1.2:  # Volume must be ≥120% of average
            return None, None  # Low volume = weak signal, skip
    
    # Elephant — large body with high volume
    if body >= EXPANSION_BODY_RATIO * avg_b:
        return "elephant", ("long" if row["close"] > row["open"] else "short")
    
    # Tail — large wick with high volume
    wick = rng - body
    if wick >= EXPANSION_WICK_RATIO * rng:
        upper = row["high"] - max(row["open"], row["close"])
        lower = min(row["open"], row["close"]) - row["low"]
        return "tail", ("long" if lower > upper else "short")
    
    return None, None




# ─────────────────────────────────────────────────────────────────────────────
# CHOP DETECTION - Detects when SMAs are flat and useless
# ─────────────────────────────────────────────────────────────────────────────

def detect_chop(df: pd.DataFrame) -> dict:
    """
    Detect if market is in chop/sideways mode where SMAs are useless.
    
    Chop conditions:
    1. SMA20 and SMA100 have minimal slope (< 0.15% movement over 10 candles)
    2. Price frequently crosses SMAs (> 4 crosses in last 15 candles)
    3. Low volatility (ATR declining)
    
    Returns:
        dict: {
            "is_chop": bool,
            "chop_score": int (0-100, higher = more choppy),
            "reason": str,
            "sma_slope": float,
            "crosses": int,
            "volatility_state": str
        }
    """
    if df.empty or len(df) < 20:
        return {
            "is_chop": False,
            "chop_score": 0,
            "reason": "Insufficient data",
            "sma_slope": None,
            "crosses": 0,
            "volatility_state": "Unknown"
        }
    
    chop_score = 0
    reasons = []
    
    # 1. Check SMA slope (are SMAs flat?)
    sma20_now = df.iloc[-1]["sma20"]
    sma20_10ago = df.iloc[-11]["sma20"] if len(df) >= 11 else df.iloc[0]["sma20"]
    sma100_now = df.iloc[-1]["sma100"]
    sma100_10ago = df.iloc[-11]["sma100"] if len(df) >= 11 else df.iloc[0]["sma100"]
    
    if not pd.isna(sma20_now) and not pd.isna(sma20_10ago) and sma20_10ago != 0:
        sma20_change = abs(sma20_now - sma20_10ago) / sma20_10ago * 100
    else:
        sma20_change = 0
    
    if not pd.isna(sma100_now) and not pd.isna(sma100_10ago) and sma100_10ago != 0:
        sma100_change = abs(sma100_now - sma100_10ago) / sma100_10ago * 100
    else:
        sma100_change = 0
    
    avg_sma_slope = (sma20_change + sma100_change) / 2
    
    # Flat SMAs = chop
    if avg_sma_slope < 0.10:  # < 0.10% movement in 10 candles
        chop_score += 40
        reasons.append(f"Flat SMAs ({avg_sma_slope:.2f}% movement)")
    elif avg_sma_slope < 0.20:
        chop_score += 25
        reasons.append(f"Weak SMA trend ({avg_sma_slope:.2f}%)")
    
    # 2. Count SMA crosses (is price whipsawing?)
    crosses = 0
    lookback = min(15, len(df) - 1)
    
    for i in range(len(df) - lookback, len(df) - 1):
        if pd.isna(df.iloc[i]["sma20"]) or pd.isna(df.iloc[i+1]["sma20"]):
            continue
        prev_above = df.iloc[i]["close"] > df.iloc[i]["sma20"]
        curr_above = df.iloc[i+1]["close"] > df.iloc[i+1]["sma20"]
        if prev_above != curr_above:
            crosses += 1
    
    # Many crosses = whipsaw
    if crosses >= 5:
        chop_score += 35
        reasons.append(f"Excessive whipsaw ({crosses} SMA crosses)")
    elif crosses >= 3:
        chop_score += 20
        reasons.append(f"Moderate whipsaw ({crosses} crosses)")
    
    # 3. Check volatility state (is range shrinking?)
    if len(df) >= 20:
        recent_ranges = df.tail(10)["range"]
        older_ranges = df.iloc[-20:-10]["range"]
        
        if len(recent_ranges) > 0 and len(older_ranges) > 0:
            recent_avg = recent_ranges.mean()
            older_avg = older_ranges.mean()
            
            if older_avg > 0:
                vol_change = (recent_avg - older_avg) / older_avg * 100
            else:
                vol_change = 0
            
            if vol_change < -20:  # Volatility dropped 20%
                chop_score += 25
                reasons.append(f"Volatility collapsing ({vol_change:.1f}%)")
                volatility_state = "Collapsing"
            elif vol_change < -10:
                chop_score += 15
                reasons.append(f"Volatility declining ({vol_change:.1f}%)")
                volatility_state = "Declining"
            elif vol_change > 20:
                volatility_state = "Expanding"
            else:
                volatility_state = "Stable"
        else:
            volatility_state = "Unknown"
    else:
        volatility_state = "Unknown"
    
    # Determine if choppy
    is_chop = chop_score >= 50  # Score >= 50 = chop mode
    
    if is_chop:
        reason = "CHOP MODE: " + " | ".join(reasons)
    else:
        reason = "Trending" if chop_score < 30 else "Borderline chop"
    
    return {
        "is_chop": is_chop,
        "chop_score": min(chop_score, 100),
        "reason": reason,
        "sma_slope": round(avg_sma_slope, 3),
        "crosses": crosses,
        "volatility_state": volatility_state,
        "reasons": reasons
    }



# ─────────────────────────────────────────────────────────────────────────────
# BREAKOUT CONFIRMATION - Detects if expansion is real or fake
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME ALIGNMENT - Detects when multiple timeframes compress together
# ─────────────────────────────────────────────────────────────────────────────

def detect_mtf_alignment(pair: str, source: str) -> dict:
    """
    Check if 3m, 5m, and 15m all show compression (SQZ or CROSSOVER).
    
    This is a VERY strong signal - when all 3 timeframes compress together,
    the breakout is usually powerful.
    
    NOTE: This function is called ONLY when scanning lower timeframes (3m, 5m, 15m).
    To avoid slowing down scans, we use minimal error handling and quick returns.
    
    Returns:
        dict: {
            "aligned": bool,
            "pattern": str (e.g., "3m:SQZ | 5m:CROSSOVER | 15m:SQZ"),
            "strength": str ("TRIPLE" | "DOUBLE" | "SINGLE" | "NONE"),
            "timeframes_compressed": list of str
        }
    """
    # Quick timeout wrapper - if this takes >2 seconds, return empty result
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("MTF alignment check timed out")
    
    try:
        # Set 2-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)
        
        # Fetch all 3 timeframes
        df_3m = get_tf_klines(pair, source, "3m", 130)
        df_5m = get_tf_klines(pair, source, "5m", 130)
        df_15m = get_tf_klines(pair, source, "15m", 130)
        
        # Cancel timeout
        signal.alarm(0)
        
        # Add indicators
        df_3m = add_indicators(df_3m) if not df_3m.empty else df_3m
        df_5m = add_indicators(df_5m) if not df_5m.empty else df_5m
        df_15m = add_indicators(df_15m) if not df_15m.empty else df_15m
        
        # Detect compression on each
        comp_3m = detect_compression_state(df_3m, "3m")
        comp_5m = detect_compression_state(df_5m, "5m")
        comp_15m = detect_compression_state(df_15m, "15m")
        
        # Build pattern string
        states = []
        compressed_tfs = []
        
        if comp_3m["state"] != "NONE":
            states.append(f"3m:{comp_3m['state']}")
            compressed_tfs.append("3m")
        
        if comp_5m["state"] != "NONE":
            states.append(f"5m:{comp_5m['state']}")
            compressed_tfs.append("5m")
        
        if comp_15m["state"] != "NONE":
            states.append(f"15m:{comp_15m['state']}")
            compressed_tfs.append("15m")
        
        # Determine strength
        count = len(compressed_tfs)
        if count == 3:
            aligned = True
            strength = "TRIPLE"
            pattern = " | ".join(states)
        elif count == 2:
            aligned = True
            strength = "DOUBLE"
            pattern = " | ".join(states)
        elif count == 1:
            aligned = False
            strength = "SINGLE"
            pattern = " | ".join(states) if states else "None compressed"
        else:
            aligned = False
            strength = "NONE"
            pattern = "None compressed"
        
        return {
            "aligned": aligned,
            "pattern": pattern,
            "strength": strength,
            "timeframes_compressed": compressed_tfs,
            "spreads": {
                "3m": comp_3m.get("spread_pct"),
                "5m": comp_5m.get("spread_pct"),
                "15m": comp_15m.get("spread_pct")
            }
        }
    
    except (TimeoutError, Exception) as e:
        # If MTF check fails or times out, return default (don't block the signal)
        return {
            "aligned": False,
            "pattern": "MTF check skipped" if isinstance(e, TimeoutError) else "Error",
            "strength": "NONE",
            "timeframes_compressed": [],
            "spreads": {}
        }



# ─────────────────────────────────────────────────────────────────────────────
# SESSION BREAKOUT DETECTOR - Extended SQZ before London/NY opens
# ─────────────────────────────────────────────────────────────────────────────

def detect_session_breakout_setup(df: pd.DataFrame, current_time_utc: datetime = None) -> dict:
    """
    Detect extended SQZ (sustained compression) in the 45 minutes before
    London open (08:00 UTC) or New York open (13:00 UTC).
    
    Setup:
    - Price squeezed between flat SMA20 and SMA100 for extended period
    - SMA20 and SMA100 are relatively flat (low slope)
    - Time window: 45 minutes before session open
    - This is a classic session breakout setup
    
    Returns:
        dict: {
            "is_session_setup": bool,
            "session": str ("London" | "New York" | "None"),
            "minutes_to_open": int,
            "sqz_duration": int (candles),
            "setup_quality": str ("ELITE" | "HIGH" | "GOOD" | "NONE"),
            "price_position": str (e.g., "Between SMAs"),
            "sma_flatness": str (e.g., "Very flat - 0.08%")
        }
    """
    from datetime import datetime, timezone, timedelta
    
    if df.empty or len(df) < 20:
        return {
            "is_session_setup": False,
            "session": "None",
            "minutes_to_open": 999,
            "sqz_duration": 0,
            "setup_quality": "NONE",
            "price_position": "—",
            "sma_flatness": "—"
        }
    
    # Get current time (use provided or system time)
    if current_time_utc is None:
        now_utc = datetime.now(timezone.utc)
    else:
        now_utc = current_time_utc
    
    current_hour = now_utc.hour
    current_minute = now_utc.minute
    
    # Calculate minutes to next session open
    london_open_hour = 8  # 08:00 UTC
    ny_open_hour = 13     # 13:00 UTC
    
    # Minutes until London open
    if current_hour < london_open_hour:
        mins_to_london = (london_open_hour - current_hour) * 60 - current_minute
    elif current_hour == london_open_hour:
        mins_to_london = -current_minute  # Negative if already open
    else:
        # Already past London open today, calculate for tomorrow
        mins_to_london = (24 - current_hour + london_open_hour) * 60 - current_minute
    
    # Minutes until NY open
    if current_hour < ny_open_hour:
        mins_to_ny = (ny_open_hour - current_hour) * 60 - current_minute
    elif current_hour == ny_open_hour:
        mins_to_ny = -current_minute
    else:
        # Already past NY open today, calculate for tomorrow
        mins_to_ny = (24 - current_hour + ny_open_hour) * 60 - current_minute
    
    # Check if we're in the 45-minute window before either session
    in_london_window = 0 < mins_to_london <= 45
    in_ny_window = 0 < mins_to_ny <= 45
    
    if not (in_london_window or in_ny_window):
        return {
            "is_session_setup": False,
            "session": "None",
            "minutes_to_open": min(mins_to_london, mins_to_ny) if mins_to_london > 0 or mins_to_ny > 0 else 999,
            "sqz_duration": 0,
            "setup_quality": "NONE",
            "price_position": "—",
            "sma_flatness": "—"
        }
    
    # Determine which session
    if in_london_window:
        session = "London"
        mins_to_open = mins_to_london
    else:
        session = "New York"
        mins_to_open = mins_to_ny
    
    # Now check the setup quality
    last = df.iloc[-1]
    price = last["close"]
    sma20 = last.get("sma20", np.nan)
    sma100 = last.get("sma100", np.nan)
    
    if pd.isna(sma20) or pd.isna(sma100):
        return {
            "is_session_setup": False,
            "session": session,
            "minutes_to_open": mins_to_open,
            "sqz_duration": 0,
            "setup_quality": "NONE",
            "price_position": "—",
            "sma_flatness": "—"
        }
    
    # Check 1: Are SMAs flat? (measure slope over last 10 candles)
    if len(df) >= 10:
        sma20_10ago = df.iloc[-10]["sma20"]
        sma100_10ago = df.iloc[-10]["sma100"]
        
        sma20_slope = abs(sma20 - sma20_10ago) / sma20_10ago * 100 if sma20_10ago > 0 else 999
        sma100_slope = abs(sma100 - sma100_10ago) / sma100_10ago * 100 if sma100_10ago > 0 else 999
        avg_slope = (sma20_slope + sma100_slope) / 2
    else:
        avg_slope = 999
    
    # Check 2: Is price between the SMAs?
    sma_min = min(sma20, sma100)
    sma_max = max(sma20, sma100)
    price_between_smas = sma_min <= price <= sma_max
    
    if price_between_smas:
        price_position = "Between SMAs ✓"
    elif price < sma_min:
        price_position = f"Below SMAs ({abs(price - sma_min) / price * 100:.2f}%)"
    else:
        price_position = f"Above SMAs ({abs(price - sma_max) / price * 100:.2f}%)"
    
    # Check 3: How long has it been in SQZ?
    sqz_duration = 0
    for i in range(len(df) - 1, max(len(df) - 50, -1), -1):
        spread = cluster_spread_pct(df.iloc[i])
        if spread <= 0.30:  # Looser threshold for extended SQZ
            sqz_duration += 1
        else:
            break
    
    # Determine setup quality
    is_session_setup = False
    
    # Elite: Very flat SMAs, price between, extended SQZ
    if avg_slope < 0.10 and price_between_smas and sqz_duration >= 15:
        setup_quality = "ELITE"
        is_session_setup = True
    # High: Flat SMAs, price between, good SQZ
    elif avg_slope < 0.15 and price_between_smas and sqz_duration >= 10:
        setup_quality = "HIGH"
        is_session_setup = True
    # Good: Somewhat flat, price between, decent SQZ
    elif avg_slope < 0.20 and price_between_smas and sqz_duration >= 5:
        setup_quality = "GOOD"
        is_session_setup = True
    else:
        setup_quality = "NONE"
        is_session_setup = False
    
    sma_flatness = f"{'Very flat' if avg_slope < 0.10 else 'Flat' if avg_slope < 0.15 else 'Somewhat flat'} ({avg_slope:.2f}% slope)"
    
    return {
        "is_session_setup": is_session_setup,
        "session": session,
        "minutes_to_open": mins_to_open,
        "sqz_duration": sqz_duration,
        "setup_quality": setup_quality,
        "price_position": price_position,
        "sma_flatness": sma_flatness,
        "avg_slope": avg_slope
    }



# ─────────────────────────────────────────────────────────────────────────────
# KILLER SIGNAL DETECTOR - Highest probability setups (near-guaranteed)
# ─────────────────────────────────────────────────────────────────────────────

def detect_killer_signal(signal_data: dict) -> dict:
    """
    Detect KILLER signals - setups with multiple confirmations stacked.
    These are the highest probability trades (70-85% win rate).
    
    Killer Signal Requirements (ALL must be true):
    1. Compression quality: ELITE (≤0.10% spread)
    2. Clean market: Chop score <40 (no ranging)
    3. BTC bias aligned with signal direction
    4. Strong volume on breakout: ≥2.0x average
    5. Breakout confirmation: HIGH
    6. At least ONE power multiplier:
       - Triple MTF alignment (3m+5m+15m)
       - Session breakout setup (before London/NY)
       - BTC liquidity sweep in same direction
       - Price at major support/resistance bounce
    
    Returns:
        dict: {
            "is_killer": bool,
            "confidence": int (70-95%),
            "multipliers": list of str (what makes it killer),
            "missing": list of str (what's preventing killer status)
        }
    """
    is_killer = False
    confidence = 0
    multipliers = []
    missing = []
    
    # Extract signal data
    signal_type = signal_data.get("signal_type", "")
    quality = signal_data.get("quality", "GOOD")
    chop_score = signal_data.get("chop_score", 0)
    direction = signal_data.get("direction", "")
    breakout_conf = signal_data.get("breakout_confidence", "")
    vol_strength = signal_data.get("volume_strength", "")
    mtf_strength = signal_data.get("mtf_strength", "NONE")
    session_setup = signal_data.get("session_setup", False)
    session_quality = signal_data.get("session_quality", "NONE")
    bias_15m = signal_data.get("bias_15m", "—")
    spread_pct = signal_data.get("spread_pct", 999)
    
    # KILLER signals only for EXPANSION breakouts
    if signal_type != "EXPANSION":
        return {
            "is_killer": False,
            "confidence": 0,
            "multipliers": [],
            "missing": ["Not an expansion breakout"]
        }
    
    # Requirement 1: ELITE compression (≤0.10% spread)
    if quality == "ELITE" or (spread_pct is not None and spread_pct <= 0.10):
        confidence += 15
    else:
        missing.append(f"Need ELITE compression (current: {quality})")
    
    # Requirement 2: Clean market (chop <40)
    if chop_score < 40:
        confidence += 15
    else:
        missing.append(f"Market too choppy (chop score: {chop_score}/100)")
    
    # Requirement 3: BTC bias aligned
    bias_aligned = False
    if direction.lower() == "long" and bias_15m.lower() == "bullish":
        confidence += 20
        bias_aligned = True
    elif direction.lower() == "short" and bias_15m.lower() == "bearish":
        confidence += 20
        bias_aligned = True
    else:
        missing.append(f"BTC bias not aligned (signal: {direction}, BTC: {bias_15m})")
    
    # Requirement 4: Strong volume (≥2.0x)
    strong_volume = False
    if vol_strength and "x" in str(vol_strength):
        try:
            vol_ratio = float(str(vol_strength).replace("x", "").replace("🔥", "").replace("✅", "").replace("⚠️", "").strip())
            if vol_ratio >= 2.0:
                confidence += 20
                strong_volume = True
            else:
                missing.append(f"Need 2.0x+ volume (current: {vol_ratio:.1f}x)")
        except:
            missing.append("Volume data unclear")
    else:
        missing.append("No volume data")
    
    # Requirement 5: HIGH breakout confirmation
    if breakout_conf == "HIGH":
        confidence += 10
    else:
        missing.append(f"Need HIGH breakout confirmation (current: {breakout_conf})")
    
    # Requirement 6: At least ONE power multiplier
    power_multiplier_count = 0
    
    # Power Multiplier A: Triple MTF alignment
    if mtf_strength == "TRIPLE":
        multipliers.append("🎯 TRIPLE MTF ALIGNMENT (all timeframes compressed)")
        confidence += 10
        power_multiplier_count += 1
    
    # Power Multiplier B: Session breakout setup
    if session_setup and session_quality in ["ELITE", "HIGH"]:
        multipliers.append(f"🔔 {session_quality} SESSION BREAKOUT SETUP")
        confidence += 10
        power_multiplier_count += 1
    
    # Power Multiplier C: Could add BTC sweep, support/resistance later
    # For now, MTF and Session are the main multipliers
    
    if power_multiplier_count == 0:
        missing.append("Need at least 1 power multiplier (TRIPLE MTF or Session Setup)")
    
    # Determine if KILLER
    # Need: ELITE + Clean + Bias + Volume + Confirmation + Multiplier
    required_confidence = 80  # All 6 requirements met = 90+ confidence
    
    if confidence >= required_confidence and len(missing) == 0:
        is_killer = True
        # Boost confidence for multiple multipliers
        if power_multiplier_count >= 2:
            confidence = min(95, confidence + 5)
    
    return {
        "is_killer": is_killer,
        "confidence": confidence,
        "multipliers": multipliers,
        "missing": missing,
        "requirements_met": 6 - len([m for m in missing if "Need" in m or "No " in m])
    }



# ─────────────────────────────────────────────────────────────────────────────
# EXIT STRATEGY - Calculate optimal exits based on signal quality
# ─────────────────────────────────────────────────────────────────────────────

def calculate_exit_strategy(signal_data: dict, entry_price: float, df: pd.DataFrame) -> dict:
    """
    Calculate complete exit strategy: stop loss, targets, and position management.
    
    Exit strategy varies by signal quality:
    - KILLER signals: Wider stops, bigger targets, trail longer
    - HIGH signals: Moderate stops/targets
    - MEDIUM signals: Tight stops, conservative targets
    
    Args:
        signal_data: The signal dict from scan_pair
        entry_price: Entry price (current price or planned entry)
        df: DataFrame with price data and indicators
    
    Returns:
        dict: {
            "stop_loss": float,
            "stop_distance": str (% from entry),
            "target_1": float (take 50% profit here),
            "target_2": float (take 30% profit here),
            "target_3": float (let 20% run to here),
            "risk_reward": str (e.g., "1:4.2"),
            "exit_plan": str (description),
            "trailing_stop": str (instructions)
        }
    """
    if df.empty or len(df) < 20:
        return {"error": "Insufficient data"}
    
    last = df.iloc[-1]
    direction = signal_data.get("direction", "").lower()
    tier = signal_data.get("tier", "MEDIUM")
    quality = signal_data.get("quality", "GOOD")
    spread_pct = signal_data.get("spread_pct", 0.20)
    session_setup = signal_data.get("session_setup", False)
    mtf_strength = signal_data.get("mtf_strength", "NONE")
    
    sma20 = last.get("sma20", entry_price)
    sma100 = last.get("sma100", entry_price)
    
    # ═══════════════════════════════════════════════════════════════════
    # STOP LOSS CALCULATION
    # ═══════════════════════════════════════════════════════════════════
    
    if direction == "long":
        # Long stop: Below compression zone (SMA100 or SMA20, whichever is lower)
        base_stop = min(sma20, sma100)
        
        # Adjust stop based on quality
        if tier == "KILLER":
            # Wider stop for KILLER (more room to breathe)
            stop_loss = base_stop * 0.995  # 0.5% below SMA
        elif quality == "ELITE":
            stop_loss = base_stop * 0.997  # 0.3% below
        else:
            stop_loss = base_stop * 0.998  # 0.2% below
        
        stop_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
        
    else:  # short
        # Short stop: Above compression zone (SMA20 or SMA100, whichever is higher)
        base_stop = max(sma20, sma100)
        
        if tier == "KILLER":
            stop_loss = base_stop * 1.005  # 0.5% above SMA
        elif quality == "ELITE":
            stop_loss = base_stop * 1.003  # 0.3% above
        else:
            stop_loss = base_stop * 1.002  # 0.2% above
        
        stop_distance_pct = ((stop_loss - entry_price) / entry_price) * 100
    
    # ═══════════════════════════════════════════════════════════════════
    # PROFIT TARGETS CALCULATION
    # ═══════════════════════════════════════════════════════════════════
    
    # Base targets on signal quality and tier
    if tier == "KILLER":
        # KILLER signals: Expect 3-8%+ moves
        target_1_pct = 3.0   # Conservative first target
        target_2_pct = 5.0   # Medium target
        target_3_pct = 8.0   # Let winners run
        
        # Boost if session setup + MTF alignment
        if session_setup and mtf_strength == "TRIPLE":
            target_3_pct = 12.0  # Ultimate setup
    
    elif tier == "HIGH" or quality == "ELITE":
        # HIGH/ELITE: Expect 2-5% moves
        target_1_pct = 2.0
        target_2_pct = 3.5
        target_3_pct = 5.5
    
    elif tier == "MEDIUM":
        # MEDIUM: Expect 1.5-3% moves
        target_1_pct = 1.5
        target_2_pct = 2.5
        target_3_pct = 4.0
    
    else:
        # Conservative for everything else
        target_1_pct = 1.0
        target_2_pct = 2.0
        target_3_pct = 3.0
    
    # Calculate actual target prices
    if direction == "long":
        target_1 = entry_price * (1 + target_1_pct / 100)
        target_2 = entry_price * (1 + target_2_pct / 100)
        target_3 = entry_price * (1 + target_3_pct / 100)
    else:  # short
        target_1 = entry_price * (1 - target_1_pct / 100)
        target_2 = entry_price * (1 - target_2_pct / 100)
        target_3 = entry_price * (1 - target_3_pct / 100)
    
    # ═══════════════════════════════════════════════════════════════════
    # RISK:REWARD RATIO
    # ═══════════════════════════════════════════════════════════════════
    
    avg_target_pct = (target_1_pct + target_2_pct + target_3_pct) / 3
    risk_reward_ratio = avg_target_pct / stop_distance_pct
    risk_reward = f"1:{risk_reward_ratio:.1f}"
    
    # ═══════════════════════════════════════════════════════════════════
    # EXIT PLAN (Step-by-step instructions)
    # ═══════════════════════════════════════════════════════════════════
    
    exit_plan = f"""
📍 ENTRY: ${entry_price:,.2f}
🛑 STOP LOSS: ${stop_loss:,.2f} ({stop_distance_pct:.2f}% risk)

🎯 TARGET 1 ({target_1_pct:.1f}%): ${target_1:,.2f}
   → Take 50% profit here
   → Move stop to breakeven

🎯 TARGET 2 ({target_2_pct:.1f}%): ${target_2:,.2f}
   → Take 30% more profit (80% total closed)
   → Move stop to Target 1

🎯 TARGET 3 ({target_3_pct:.1f}%): ${target_3:,.2f}
   → Take final 20% profit
   → Or let it run with trailing stop

⚖️ RISK:REWARD: {risk_reward}
"""
    
    # ═══════════════════════════════════════════════════════════════════
    # TRAILING STOP (For remaining position after Target 2)
    # ═══════════════════════════════════════════════════════════════════
    
    if tier == "KILLER":
        trailing_pct = 2.0  # Wide trailing stop (let it run)
        trailing_stop = f"After Target 2: Trail stop {trailing_pct}% behind price (e.g., if price hits ${target_3:,.2f}, stop at ${target_3 * (1 - trailing_pct/100):,.2f})"
    elif tier == "HIGH":
        trailing_pct = 1.5
        trailing_stop = f"After Target 2: Trail stop {trailing_pct}% behind price"
    else:
        trailing_pct = 1.0
        trailing_stop = f"After Target 2: Trail stop {trailing_pct}% behind price"
    
    return {
        "stop_loss": round(stop_loss, 2),
        "stop_distance": f"{stop_distance_pct:.2f}%",
        "target_1": round(target_1, 2),
        "target_1_pct": f"{target_1_pct:.1f}%",
        "target_2": round(target_2, 2),
        "target_2_pct": f"{target_2_pct:.1f}%",
        "target_3": round(target_3, 2),
        "target_3_pct": f"{target_3_pct:.1f}%",
        "risk_reward": risk_reward,
        "exit_plan": exit_plan.strip(),
        "trailing_stop": trailing_stop,
        "position_management": {
            "at_target_1": "Close 50%, move stop to breakeven",
            "at_target_2": "Close 30% more (80% total), move stop to Target 1",
            "at_target_3": "Close final 20% or trail stop"
        }
    }

def confirm_breakout(df: pd.DataFrame, direction: str, chop_score: int) -> dict:
    """
    Confirm if breakout from compression is real or likely to fail.
    
    Real breakout criteria:
    1. Volume surge (≥150% of average, ≥200% if choppy)
    2. Strong candle (elephant or tail with large body/wick)
    3. Momentum continuation (next candle follows direction)
    4. If choppy: needs ALL 3 criteria, otherwise likely fake
    
    Args:
        df: DataFrame with OHLCV + indicators
        direction: "long" or "short"
        chop_score: Chop score (0-100)
    
    Returns:
        dict: {
            "confirmed": bool,
            "confidence": str ("HIGH", "MEDIUM", "LOW"),
            "volume_strength": str,
            "reasons": list of str
        }
    """
    if df.empty or len(df) < 2:
        return {
            "confirmed": False,
            "confidence": "UNKNOWN",
            "volume_strength": "—",
            "reasons": ["Insufficient data"]
        }
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    reasons = []
    confirmation_score = 0
    
    # 1. Volume check
    avg_vol = curr.get("avg_volume", np.nan)
    curr_vol = curr.get("volume", np.nan)
    
    if not pd.isna(avg_vol) and not pd.isna(curr_vol) and avg_vol > 0:
        vol_ratio = curr_vol / avg_vol
        
        # Adjust threshold based on chop
        if chop_score >= 60:
            # Choppy market: need 2x volume minimum
            required_vol = 2.0
        else:
            # Normal market: 1.5x volume sufficient
            required_vol = 1.5
        
        if vol_ratio >= required_vol:
            confirmation_score += 40
            volume_strength = f"{vol_ratio:.1f}x avg"
            if vol_ratio >= 2.5:
                reasons.append(f"🔥 HUGE volume spike ({vol_ratio:.1f}x average)")
                volume_strength = f"🔥 {vol_ratio:.1f}x"
            elif vol_ratio >= 2.0:
                reasons.append(f"✅ Strong volume ({vol_ratio:.1f}x average)")
                volume_strength = f"✅ {vol_ratio:.1f}x"
            else:
                reasons.append(f"Good volume ({vol_ratio:.1f}x average)")
                volume_strength = f"{vol_ratio:.1f}x"
        else:
            reasons.append(f"⚠️ Low volume ({vol_ratio:.1f}x average, need {required_vol}x)")
            volume_strength = f"⚠️ {vol_ratio:.1f}x"
    else:
        volume_strength = "—"
        reasons.append("⚠️ Volume data unavailable")
    
    # 2. Candle strength check
    body = curr.get("body", 0)
    avg_body = curr.get("avg_body", 0)
    rng = curr.get("range", 0)
    
    if avg_body > 0 and rng > 0:
        body_ratio = body / avg_body
        body_pct = (body / rng) * 100
        
        # Strong body = conviction
        if body_ratio >= 2.0 and body_pct >= 50:
            confirmation_score += 30
            reasons.append(f"✅ Strong conviction candle ({body_ratio:.1f}x avg body)")
        elif body_ratio >= 1.5:
            confirmation_score += 20
            reasons.append(f"Good sized candle ({body_ratio:.1f}x avg)")
        else:
            reasons.append(f"⚠️ Weak candle body ({body_ratio:.1f}x avg)")
    
    # 3. Directional follow-through check (is price continuing?)
    if direction == "long":
        # Long breakout: close should be near high
        if curr["close"] >= curr["high"] * 0.95:
            confirmation_score += 30
            reasons.append("✅ Closed near high (strong bulls)")
        elif curr["close"] >= curr["high"] * 0.85:
            confirmation_score += 15
            reasons.append("Closed mid-high (decent)")
        else:
            reasons.append("⚠️ Closed far from high (weak bulls)")
    else:
        # Short breakout: close should be near low
        if curr["close"] <= curr["low"] * 1.05:
            confirmation_score += 30
            reasons.append("✅ Closed near low (strong bears)")
        elif curr["close"] <= curr["low"] * 1.15:
            confirmation_score += 15
            reasons.append("Closed mid-low (decent)")
        else:
            reasons.append("⚠️ Closed far from low (weak bears)")
    
    # 4. Determine confirmation
    # In choppy markets, need higher score
    if chop_score >= 60:
        # Choppy: need 80+ score
        if confirmation_score >= 80:
            confirmed = True
            confidence = "HIGH"
        elif confirmation_score >= 60:
            confirmed = False
            confidence = "MEDIUM"
            reasons.append("⚠️ CHOPPY MARKET: Breakout lacks strong confirmation")
        else:
            confirmed = False
            confidence = "LOW"
            reasons.append("🚫 CHOPPY MARKET: Likely fake breakout")
    else:
        # Normal market: need 60+ score
        if confirmation_score >= 80:
            confirmed = True
            confidence = "HIGH"
        elif confirmation_score >= 60:
            confirmed = True
            confidence = "MEDIUM"
        else:
            confirmed = False
            confidence = "LOW"
            reasons.append("⚠️ Weak breakout signals")
    
    return {
        "confirmed": confirmed,
        "confidence": confidence,
        "volume_strength": volume_strength,
        "confirmation_score": confirmation_score,
        "reasons": reasons
    }

def detect_expansion(df: pd.DataFrame, chop_score: int = 0) -> Optional[dict]:
    """
    Expansion detection — current state only.

    Current candle is the expansion candle IF:
    1. Previous candle WAS in compression (cluster spread <= 0.20%)
    2. Current candle is NOT in compression (broke out)
    3. Current candle is elephant or tail
    4. Breakout is confirmed (volume + strength)

    That is it. No history. No lookback loops.
    """
    if df.empty or len(df) < 3 or "sma20" not in df.columns:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Previous candle must have been in compression
    if not _in_compression(prev):
        return None

    # Current candle must have broken out of compression
    if _in_compression(curr):
        return None

    # Current candle must be elephant or tail
    ctype, direction = _classify_candle(curr)
    if ctype and direction:
        # Check breakout confirmation
        breakout_confirm = confirm_breakout(df, direction, chop_score)
        
        return {
            "direction":   direction,
            "candle_type": ctype,
            "signal_age":  0,
            "breakout_confirmed": breakout_confirm["confirmed"],
            "breakout_confidence": breakout_confirm["confidence"],
            "volume_strength": breakout_confirm["volume_strength"],
            "breakout_reasons": breakout_confirm["reasons"],
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
    BEAR SWEEP: candle high breaks ANY BSL level → closes below it
    BULL SWEEP: candle low breaks ANY SSL level → closes above it

    Checks last 10 candles against ALL pivot levels — not just the extreme one.
    Returns the most recent sweep found.
    """
    if df.empty or len(df) < 6:
        return None

    # Build pivots from all candles except last 2 (need left+right neighbours)
    bsl, ssl = detect_3bar_pivots(df.iloc[:-2])
    last10   = df.tail(10)

    best = None  # most recent sweep

    for idx in range(len(last10) - 1, -1, -1):
        row      = last10.iloc[idx]
        bars_ago = len(last10) - 1 - idx

        # Check against ALL BSL levels — bear sweep
        for level in bsl:
            lp = level["price"]
            if row["high"] > lp and row["close"] < lp:
                wick_pct = (row["high"] - lp) / lp * 100
                if best is None or bars_ago < best["bars_ago"]:
                    best = {
                        "type":      "BEAR SWEEP",
                        "level":     round(lp, 2),
                        "wick_pct":  round(wick_pct, 3),
                        "bars_ago":  bars_ago,
                        "confirmed": True,
                    }

        # Check against ALL SSL levels — bull sweep
        for level in ssl:
            lp = level["price"]
            if row["low"] < lp and row["close"] > lp:
                wick_pct = (lp - row["low"]) / lp * 100
                if best is None or bars_ago < best["bars_ago"]:
                    best = {
                        "type":      "BULL SWEEP",
                        "level":     round(lp, 2),
                        "wick_pct":  round(wick_pct, 3),
                        "bars_ago":  bars_ago,
                        "confirmed": True,
                    }

        # Return as soon as we find sweep on most recent candle
        if best and best["bars_ago"] == bars_ago:
            break

    return best


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
    """
    Pullback bounce off SMA20 — current candle only.

    LONG PULLBACK — price bounces off SMA20 from above:
    - Current candle OPENED above SMA20 (was in uptrend)
    - Current candle LOW touched SMA20 (pulled back down to it)
    - Current candle CLOSED above SMA20 (bounced back up — SMA20 held as support)
    - Current candle is elephant or tail (confirms the bounce)

    SHORT PULLBACK — price bounces off SMA20 from below:
    - Current candle OPENED below SMA20 (was in downtrend)
    - Current candle HIGH touched SMA20 (pulled back up to it)
    - Current candle CLOSED below SMA20 (rejected back down — SMA20 held as resistance)
    - Current candle is elephant or tail (confirms the rejection)
    """
    if df.empty or len(df) < 3 or "sma20" not in df.columns:
        return None

    curr = df.iloc[-1]
    s20  = curr.get("sma20", np.nan)
    if pd.isna(s20):
        return None

    # LONG: price pulled back to SMA20 and BOUNCED back up
    # - Opened above SMA20
    # - Low touched SMA20 (came down to it)
    # - Closed ABOVE the open (bounced — green candle or strong close)
    # - Closed above SMA20 (did not break through)
    long_bounce = (curr["open"]  >  s20               # opened above SMA20
                   and curr["low"]   <= s20 * 1.002    # low touched SMA20
                   and curr["close"] >  curr["open"]   # closed above open — bounce confirmed
                   and curr["close"] >  s20)            # close stayed above SMA20

    # SHORT: price pulled back to SMA20 and was REJECTED back down
    # - Opened below SMA20
    # - High touched SMA20 (came up to it)
    # - Closed BELOW the open (rejected — red candle or strong close down)
    # - Closed below SMA20 (did not break through)
    short_bounce = (curr["open"]  <  s20               # opened below SMA20
                    and curr["high"]  >= s20 * 0.998   # high touched SMA20
                    and curr["close"] <  curr["open"]  # closed below open — rejection confirmed
                    and curr["close"] <  s20)           # close stayed below SMA20

    if not long_bounce and not short_bounce:
        return None

    # Must be elephant or tail to confirm the bounce
    ctype, _ = _classify_candle(curr)
    if not ctype:
        return None

    if long_bounce:
        return {"direction": "long", "candle_type": ctype}
    if short_bounce:
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




# ─────────────────────────────────────────────────────────────────────────────
# BTC SESSION PERFORMANCE - Tracks cumulative return by trading session
# ─────────────────────────────────────────────────────────────────────────────

def get_btc_session_performance(lookback_days: int = 30) -> dict:
    """
    Calculate BTC cumulative returns broken down by trading session.
    
    Sessions (UTC times):
    - Asian:  00:00 - 08:00 (8 hours)
    - London: 08:00 - 13:00 (5 hours)
    - NY:     13:00 - 20:00 (7 hours)
    - After:  20:00 - 24:00 (4 hours)
    
    This shows which session is strongest/weakest for trading.
    
    Args:
        lookback_days: How many days to analyze (default 30)
    
    Returns:
        dict: {
            "asian": {"return_pct": float, "win_rate": float, "avg_move": float, "direction": str},
            "london": {...},
            "ny": {...},
            "after_hours": {...},
            "best_session": str,
            "worst_session": str
        }
    """
    from datetime import datetime, timezone, timedelta
    
    try:
        # Fetch 1H candles for analysis - try 30 days, fallback to 7 days, then 24h
        attempts = [
            (lookback_days, lookback_days * 24),  # 30 days = 720 candles
            (7, 7 * 24),  # 7 days = 168 candles
            (1, 24)  # 24 hours = 24 candles minimum
        ]
        
        df = None
        active = None
        actual_days = 0
        
        for days, limit in attempts:
            df, active = get_btc_data_with_failover("1h", limit)
            if not df.empty and len(df) >= 24:
                actual_days = days
                print(f"[SESSION] Using {days} days of data ({len(df)} candles)")
                break
        
        if df is None or df.empty or len(df) < 24:
            return {
                "asian": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—"},
                "london": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—"},
                "ny": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—"},
                "after_hours": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—"},
                "best_session": "Unknown",
                "worst_session": "Unknown",
                "error": "Insufficient data"
            }
        
        # Debug: Check what we got
        print(f"[SESSION DEBUG] Received {len(df)} candles for session performance")
        
        # IMPROVED: Force timestamp extraction with multiple fallbacks
        import pandas as pd
        from datetime import datetime, timedelta, timezone as tz
        
        # Try multiple methods to get timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        elif hasattr(df.index, 'to_pydatetime'):
            df['timestamp'] = pd.to_datetime(df.index, utc=True)
        else:
            # Create synthetic timestamps going backwards from now
            now = datetime.now(tz.utc)
            df['timestamp'] = [now - timedelta(hours=i) for i in range(len(df)-1, -1, -1)]
        
        # Ensure timezone aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Calculate hourly returns (using close - open for each 1H candle)
        df['return_pct'] = ((df['close'] - df['open']) / df['open'] * 100).fillna(0)
        
        # Extract hour in UTC (0-23)
        df['hour_utc'] = df['timestamp'].dt.hour
        
        # Debug print
        print(f"[SESSION] Hour range: {df['hour_utc'].min()}-{df['hour_utc'].max()}")
        print(f"[SESSION] Sample hours: {df['hour_utc'].head(10).tolist()}")
        
        print(f"[SESSION DEBUG] Sample hours: {df['hour_utc'].head(10).tolist()}")
        print(f"[SESSION DEBUG] Sample returns: {df['return_pct'].head(5).tolist()}")
        
        # Define sessions
        def classify_session(hour):
            if 0 <= hour < 8:
                return 'asian'
            elif 8 <= hour < 13:
                return 'london'
            elif 13 <= hour < 20:
                return 'ny'
            else:
                return 'after_hours'
        
        df['session'] = df['hour_utc'].apply(classify_session)
        
        # Calculate metrics per session
        sessions = {}
        for session_name in ['asian', 'london', 'ny', 'after_hours']:
            session_df = df[df['session'] == session_name]
            
            if len(session_df) > 0:
                cumulative_return = session_df['return_pct'].sum()
                wins = (session_df['return_pct'] > 0).sum()
                total = len(session_df)
                win_rate = (wins / total * 100) if total > 0 else 0
                avg_move = session_df['return_pct'].abs().mean()
                
                # Determine direction
                if cumulative_return > 0.5:
                    direction = "🟢 Bullish"
                elif cumulative_return < -0.5:
                    direction = "🔴 Bearish"
                else:
                    direction = "🟡 Neutral"
                
                sessions[session_name] = {
                    "return_pct": round(cumulative_return, 2),
                    "win_rate": round(win_rate, 1),
                    "avg_move": round(avg_move, 2),
                    "direction": direction,
                    "count": total
                }
            else:
                sessions[session_name] = {
                    "return_pct": 0,
                    "win_rate": 0,
                    "avg_move": 0,
                    "direction": "—",
                    "count": 0
                }
        
        # Find best and worst sessions
        returns = {k: v["return_pct"] for k, v in sessions.items()}
        best_session = max(returns, key=returns.get)
        worst_session = min(returns, key=returns.get)
        
        return {
            "asian": sessions["asian"],
            "london": sessions["london"],
            "ny": sessions["ny"],
            "after_hours": sessions["after_hours"],
            "best_session": best_session.replace("_", " ").title(),
            "worst_session": worst_session.replace("_", " ").title(),
            "lookback_days": actual_days if 'actual_days' in locals() else lookback_days
        }
    
    except Exception as e:
        print(f"[SESSION ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "asian": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—", "count": 0},
            "london": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—", "count": 0},
            "ny": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—", "count": 0},
            "after_hours": {"return_pct": 0, "win_rate": 0, "avg_move": 0, "direction": "—", "count": 0},
            "best_session": "Error",
            "worst_session": "Error",
            "error": str(e),
            "lookback_days": lookback_days
        }



# ─────────────────────────────────────────────────────────────────────────────
# OPEN INTEREST TRACKER - Monitors OI changes to detect squeeze setups
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)  # Cache 3 min
def get_btc_open_interest_state() -> dict:
    """
    Get BTC Open Interest - SIMPLIFIED (Network restrictions workaround)
    Only fetches current OI. Funding/price require manual Coinglass check.
    """
    import requests
    
    exchanges = [
        {"name": "OKX", "url": "https://www.okx.com/api/v5/public/open-interest", "param": "instId", "symbol": "BTC-USDT-SWAP"},
        {"name": "Gate.io", "url": "https://api.gateio.ws/api/v4/futures/usdt/contracts/BTC_USDT"},
    ]
    
    for exch in exchanges:
        try:
            params = {exch["param"]: exch["symbol"]} if "param" in exch else {}
            resp = requests.get(exch["url"], params=params, timeout=2)
            
            if resp.status_code == 200:
                data = resp.json()
                
                if exch["name"] == "OKX":
                    oi = float(data.get("data", [{}])[0].get("oi", 0))
                elif exch["name"] == "Gate.io":
                    oi = float(data.get("long_oi", 0)) + float(data.get("short_oi", 0))
                else:
                    continue
                    
                if oi > 0:
                    return {
                        "current_oi": round(oi, 0),
                        "data_source": f"{exch['name']} ✓",
                        "error": None
                    }
        except:
            continue
    
    return {"current_oi": 0, "data_source": "Failed", "error": "Network blocked"}

def get_btc_regime() -> dict:
    result = {}
    for label, interval, limit in [("15m","15m",130),("1H","1h",60),("4H","4h",60)]:
        df, active = get_btc_data_with_failover(interval, limit)
        if df.empty or len(df) < 20:
            result[label] = {"regime":"Unknown","dir":"—","sep":0,"price":"—","active":active,"comp":"—"}
            continue
        df   = add_indicators(df)
        last = df.iloc[-1]
        s20, s100 = last.get("sma20", np.nan), last.get("sma100", np.nan)
        if pd.isna(s20) or pd.isna(s100) or s100 == 0:
            result[label] = {"regime":"Unknown","dir":"—","sep":0,"price":round(last["close"],2),"active":active,"comp":"—"}
            continue

        sep      = (s20 - s100) / s100
        trending = abs(sep) >= REGIME_SMA_SEP

        # Regime label
        if trending:
            regime = "Trending Up"   if sep > 0 else "Trending Down"
            dir_   = "Up"            if sep > 0 else "Down"
        else:
            regime = "Ranging"
            dir_   = "—"

        # Compression state on this timeframe
        comp = detect_compression_state(df, label)
        comp_label = comp["state"] if comp["state"] != "NONE" else "—"

        result[label] = {
            "regime": regime,
            "dir":    dir_,
            "sep":    round(abs(sep)*100, 2),
            "price":  round(last["close"], 2),
            "sma20":  round(s20, 2),
            "sma100": round(s100, 2),
            "active": active,
            "comp":   comp_label,
            "spread": comp.get("spread_pct"),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PAIR SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan_pair(pair: str, source: str, tf: str) -> Optional[dict]:
    try:
        df = get_tf_klines(pair, source, tf, 130)
        if df.empty or len(df) < 25:
            # Log API failure for debugging
            add_log(source.upper(), pair, "API-FAIL", f"No data for {tf}", "SKIP")
            return None
        df = add_indicators(df)

        price = df.iloc[-1]["close"]
        rsi   = df.iloc[-1].get("rsi14", np.nan)

        # Always check compression state
        comp = detect_compression_state(df, tf)
        
        # Check for chop mode (SMAs useless, ranging market)
        chop = detect_chop(df)

        signal_type = None
        direction   = None
        candle_type = None
        signal_age  = 0
        health_info = None
        
        # Store chop info for later use (don't block signals yet)
        chop_detected = chop["is_chop"]
        chop_score = chop["chop_score"]
        chop_reasons = chop["reasons"]

        # Check expansion from compression
        exp = detect_expansion(df, chop_score)
        if exp:
            bias_15m = get_15m_bias(pair, source)
            # Drop contradictory bias
            if exp["direction"] == "long"  and bias_15m == "bearish":
                pass
            elif exp["direction"] == "short" and bias_15m == "bullish":
                pass
            else:
                signal_type = "EXPANSION"
                direction   = exp["direction"]
                candle_type = exp["candle_type"]
                signal_age  = exp["signal_age"]
        
        # If compression is active but no expansion — report compression
        if not signal_type and comp["state"] != "NONE":
            disp = pair_display(pair, source)
            quality = comp.get('quality', 'GOOD')
            quality_emoji = "🔥" if quality == "ELITE" else "⭐" if quality == "HIGH" else "✓"
            
            # Check for session breakout setup (extended SQZ before London/NY open)
            session_setup = detect_session_breakout_setup(df)
            
            # Check for multi-timeframe alignment (only on lower timeframes)
            # This is optional - if it fails, we still show the compression signal
            mtf_align = None
            if tf in ["3m", "5m", "15m"]:
                try:
                    mtf_align = detect_mtf_alignment(pair, source)
                except Exception:
                    # MTF check failed, continue without it
                    mtf_align = {"aligned": False, "strength": "NONE", "pattern": ""}
            
            # Build parts list with quality and chop warning
            parts_list = [f"{quality_emoji} {quality} quality · {comp.get('spread_pct','—')}% spread"]
            
            # Add session breakout setup if detected
            if session_setup["is_session_setup"]:
                session_emoji = "🔔" if session_setup["setup_quality"] == "ELITE" else "🕐"
                parts_list.append(f"{session_emoji} SESSION SETUP: {session_setup['session']} in {session_setup['minutes_to_open']}min · {session_setup['setup_quality']} · {session_setup['sqz_duration']} candles SQZ · {session_setup['price_position']}")
            
            # Add MTF alignment if detected
            if mtf_align and mtf_align["aligned"]:
                if mtf_align["strength"] == "TRIPLE":
                    parts_list.append(f"🎯 TRIPLE ALIGNMENT: {mtf_align['pattern']}")
                elif mtf_align["strength"] == "DOUBLE":
                    parts_list.append(f"🎯 DOUBLE ALIGNMENT: {mtf_align['pattern']}")
            
            # Check for VOLATILITY BREAKOUT setup (ultra-tight compression)
            if comp.get('spread_pct', 1.0) <= 0.08:
                vol_breakout = detect_volatility_breakout(pair, source, comp.get('spread_pct', 1.0), comp.get('candles_in_comp', 0))
                if vol_breakout.get("detected"):
                    parts_list.append(f"💥 VOLATILITY BREAKOUT: {vol_breakout['ultra_grade']} · {vol_breakout['expected_move']} · {vol_breakout['duration']} candles · Vol collapse {vol_breakout['volatility_collapse']:.1f}%")
            
            # Check for RANGE BREAK CONTINUATION setup (only if not choppy)
            if not chop_detected:
                regime = get_btc_regime()
                range_break = detect_range_break_continuation(pair, source, regime, chop_score)
                if range_break.get("detected"):
                    arrow = "📈" if range_break["direction"] == "long" else "📉"
                    parts_list.append(f"{arrow} TREND CONTINUATION: {range_break['continuation_type']} · {range_break['trend_strength']} · Chop {chop_score}")
            
            # Add chop warning if detected
            if chop_detected:
                parts_list.append(f"⚠️ CHOP WARNING: Score {chop_score}/100 - SMAs unreliable")
                parts_list.extend(chop_reasons)
            
            # Determine tier based on chop
            if chop_detected:
                tier = "RISKY"  # New tier for choppy compressions
                log_tier = f"CHOP-{comp['state']}"
            else:
                tier = "WATCH"
                log_tier = comp["state"]
            
            # Boost tier if session setup or TRIPLE alignment
            if session_setup["is_session_setup"] and session_setup["setup_quality"] in ["ELITE", "HIGH"]:
                tier = "HIGH"  # Session breakout setup is very strong
                log_tier = f"SESSION-{session_setup['session']}"
            elif mtf_align and mtf_align["strength"] == "TRIPLE":
                tier = "HIGH"  # TRIPLE alignment is very strong
                log_tier = "MTF-TRIPLE"
            
            add_log(source.upper(), disp, comp["state"], "—", log_tier)
            
            return {
                "pair":          disp,
                "raw_pair":      pair,
                "source":        source,
                "tf":            tf,
                "signal_type":   "COMPRESSION",
                "direction":     "Watch",
                "candle_type":   "—",
                "score":         0,
                "tier":          tier,
                "parts":         parts_list,
                "quality":       quality,
                "mtf_aligned":   mtf_align["aligned"] if mtf_align else False,
                "mtf_strength":  mtf_align["strength"] if mtf_align else "NONE",
                "mtf_pattern":   mtf_align["pattern"] if mtf_align else "",
                "bias_15m":      "—",
                "rsi":           round(rsi, 1) if not pd.isna(rsi) else "—",
                "room":          "—",
                "room_d":        "",
                "obs":           "None",
                "obs_d":         "",
                "freshness":     f"{comp.get('candles_in_comp',0)} candles",
                "signal_age":    0,
                "comp_state":    comp["state"],
                "chop_detected": chop_detected,
                "chop_score":    chop_score,
                "chop_reasons":  chop_reasons,
                "session_setup": session_setup["is_session_setup"],
                "session_name":  session_setup["session"],
                "session_mins":  session_setup["minutes_to_open"],
                "session_quality": session_setup["setup_quality"],
                "spread_pct":    comp.get("spread_pct"),
                "candles_in_comp": comp.get("candles_in_comp", 0),
                "price":         round(price, 4),
                "health":        None,
            }
        
        # No signal found
        if not signal_type:
            return None

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

        # Build complete signal data for killer detection
        temp_signal = {
            "signal_type": signal_type,
            "quality": comp.get("quality", "GOOD"),
            "chop_score": chop_score,
            "direction": direction,
            "breakout_confidence": exp.get("breakout_confidence", "—") if signal_type == "EXPANSION" else "—",
            "volume_strength": exp.get("volume_strength", "—") if signal_type == "EXPANSION" else "—",
            "mtf_strength": "NONE",  # Will be set if available
            "session_setup": False,
            "session_quality": "NONE",
            "bias_15m": bias_15m,
            "spread_pct": comp.get("spread_pct"),
        }
        
        # Check for KILLER signal (highest probability)
        killer = detect_killer_signal(temp_signal)
        
        # Calculate exit strategy
        exit_strat = calculate_exit_strategy(signal_data=temp_signal, entry_price=price, df=df)
        
        # If KILLER signal detected, upgrade tier and add to parts
        if killer["is_killer"]:
            tier = "KILLER"  # New highest tier
            parts.insert(0, f"🎯 KILLER SIGNAL ({killer['confidence']}% confidence)")
            for mult in killer["multipliers"]:
                parts.append(mult)
        elif killer["confidence"] >= 60:
            # High confidence but missing 1-2 requirements
            parts.append(f"⚠️ Almost KILLER ({killer['confidence']}% confidence) - Missing: {', '.join(killer['missing'][:2])}")

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
            "breakout_confidence": exp.get("breakout_confidence", "—") if signal_type == "EXPANSION" else "—",
            "volume_strength": exp.get("volume_strength", "—") if signal_type == "EXPANSION" else "—",
            "breakout_reasons": exp.get("breakout_reasons", []) if signal_type == "EXPANSION" else [],
            "killer_signal": killer["is_killer"],
            "killer_confidence": killer["confidence"],
            "killer_missing": killer["missing"],
            "exit_strategy": exit_strat,
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def run_scanner(tf: str, exchanges: list = None, watchlist: list = None) -> pd.DataFrame:
    if exchanges is None:
        exchanges = ["okx", "gate", "mexc"]

    pair_sources = []

    # Watchlist pairs — always scanned first, regardless of volume rank
    if watchlist:
        for raw in watchlist:
            raw = raw.strip().upper()
            if not raw:
                continue
            # Try to detect source from format
            if "-USDT-SWAP" in raw or "-USDT" in raw:
                inst = raw if raw.endswith("-SWAP") else raw + "-SWAP"
                pair_sources.insert(0, (inst, "okx"))
            elif "_USDT" in raw:
                pair_sources.insert(0, (raw, "gate"))
            else:
                # Assume OKX format — convert ZAMAUSDT → ZAMA-USDT-SWAP
                base = raw.replace("USDT","").replace("/","")
                inst = f"{base}-USDT-SWAP"
                pair_sources.insert(0, (inst, "okx"))

    existing = {pair_display(x[0], x[1]) for x in pair_sources}

    if "okx" in exchanges:
        for p in okx_top_pairs(30):
            if pair_display(p, "okx") not in existing:
                pair_sources.append((p, "okx"))
                existing.add(pair_display(p, "okx"))
    if "gate" in exchanges:
        for p in gate_top_pairs(20):
            if pair_display(p, "gate") not in existing:
                pair_sources.append((p, "gate"))
                existing.add(pair_display(p, "gate"))
    if "mexc" in exchanges:
        for p in mexc_top_pairs(10):
            if pair_display(p, "mexc") not in existing:
                pair_sources.append((p, "mexc"))
                existing.add(pair_display(p, "mexc"))

    if not pair_sources:
        return pd.DataFrame()

    pair_sources = pair_sources[:60]
    results  = []
    progress = st.empty()

    for i, (pair, source) in enumerate(pair_sources):
        progress.caption(f"Scanning {i+1}/{len(pair_sources)}: {pair_display(pair, source)}")
        sig = scan_pair(pair, source, tf)
        if sig:
            results.append(sig)
        time.sleep(0.1)

    progress.empty()
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    pri = {"REVERSAL": 0, "EXPANSION": 1, "PULLBACK": 2, "COMPRESSION": 3}
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
            r      = regime.get(tf, {})
            reg    = r.get("regime","Unknown")
            sep    = r.get("sep", 0)
            price  = r.get("price","—")
            active = r.get("active","—")
            comp   = r.get("comp","—")
            spread = r.get("spread")

            if "Up" in reg:
                css, icon = "regime-up",    "▲"
            elif "Down" in reg:
                css, icon = "regime-down",  "▼"
            else:
                css, icon = "regime-range", "◆"

            comp_html = ""
            if comp == "SQZ":
                comp_html = f'&nbsp;<span style="background:#fff7e6;color:#f5a623;border-radius:4px;padding:1px 6px;font-size:0.67rem;font-weight:700;">SQZ {spread}%</span>'
            elif comp == "CROSSOVER":
                comp_html = f'&nbsp;<span style="background:#e8f4ff;color:#2196f3;border-radius:4px;padding:1px 6px;font-size:0.67rem;font-weight:700;">CROSSOVER {spread}%</span>'

            st.markdown(f"""
            <div class="{css}">
              <div class="regime-label">{icon} {label} — {reg}{comp_html}</div>
              <div class="regime-meta">SMA Gap:{sep}% | ${price:,} | {active}</div>
            </div>""", unsafe_allow_html=True)
    
def render_liquidity(panel: dict):
    if "error" in panel:
        st.warning(f"⚠ Liquidity engine unavailable: {panel['error']}")
        return
    
    # ── OPEN INTEREST & SQUEEZE DETECTOR ──────────────────────────────────────
    st.markdown('<div style="font-size:0.9rem;font-weight:800;color:#111;margin-bottom:4px;">📊 Open Interest & Squeeze Detection</div>', unsafe_allow_html=True)
    
    st.info("""
    **📊 For OI, Funding Rate & Squeeze Detection:**
    
    Use the standalone HTML dashboard: **`btc_market_dashboard.html`**
    
    **What it shows:**
    - ✅ BTC Price + 24h change %
    - ✅ Open Interest + change %
    - ✅ 24h Volume + change %
    - ✅ Funding Rate (who pays who)
    - ✅ Auto-detects 4 squeeze patterns
    - ✅ Auto-refreshes every 60 seconds
    
    **How to use:**
    1. Open the HTML file in your browser
    2. Keep it open alongside this scanner
    3. It fetches data directly (no network restrictions)
    
    **Available in your project files!**
    """)
