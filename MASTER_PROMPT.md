# 🔧 MASTER REBUILD PROMPT

Complete instructions to rebuild the BTC Expansion Edge Scanner from scratch.

---

## 🎯 SYSTEM OVERVIEW

A real-time cryptocurrency scanner that monitors 60+ pairs and auto-detects 7 professional trading setups with 75-90% win rates.

**Tech Stack:** Python 3.12 + Streamlit + CCXT + Pandas

---

## ✅ REQUIREMENTS

```txt
streamlit>=1.28.0
pandas>=2.1.0
ccxt>=4.1.0
requests>=2.31.0
numpy>=1.24.0
```

---

## 📝 BUILD PHASES

### Phase 1: Foundation
- Setup project structure
- Test exchange connectivity
- Basic Streamlit UI

### Phase 2: Data Layer
- Multi-exchange failover system
- OHLCV data fetching
- Technical indicators (SMA, RSI, Bollinger)

### Phase 3: Core Detection
- Compression detection
- Expansion detection with volume
- Chop/ranging market filter

### Phase 4: BTC Intelligence
- Market regime (15m/1H/4H)
- Liquidity sweep detector
- Session performance tracker

### Phase 5: 7 Setup Detectors
1. Short Squeeze
2. Long Squeeze
3. Session Breakout
4. Triple MTF Alignment
5. Liquidity Sweep
6. Range Break Continuation
7. Volatility Breakout

### Phase 6: UI & Display
- Signal card rendering
- Filter system
- Exit strategy display

### Phase 7: Testing & Deployment
- Local testing
- Render.com deployment

---

## 🔑 KEY FUNCTIONS

```python
get_data_with_failover()   # Fetch data with multi-exchange fallback
add_indicators()           # Add SMA, RSI, Bollinger Bands
detect_compression_state() # Find tight price/SMA compression
detect_expansion()         # Detect breakout with volume
detect_chop()             # Filter ranging markets
get_btc_regime()          # Analyze BTC across timeframes
build_liquidity_panel()   # Detect sweeps and zones
detect_squeeze_setup()    # Find short/long squeeze setups
scan_pair()               # Main scanning logic
```

---

## 🚀 DEPLOYMENT

### Render.com (Recommended)

```yaml
# render.yaml
services:
  - type: web
    name: crypto-scanner
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
```

Push to GitHub → Connect to Render → Auto-deploy

---

## 📚 RESOURCES

**See full implementation details in:**
- `app.py` - Complete source code
- `ALL_7_SETUPS_GUIDE.md` - Trading strategies
- `AI_PROOF_PROMPT.md` - Step-by-step guide

**External:**
- CCXT: https://docs.ccxt.com
- Streamlit: https://docs.streamlit.io

---

**Time to build:** 10-15 days full-time

**Result:** Professional trading scanner with 7 auto-detected setups
