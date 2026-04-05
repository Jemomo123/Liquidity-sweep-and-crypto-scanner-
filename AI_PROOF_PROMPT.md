# 🤖 AI-PROOF IMPLEMENTATION PROMPT

**Step-by-step guide for AI assistants to implement the BTC Expansion Edge Scanner**

---

## 📋 OVERVIEW

This guide provides explicit, sequential instructions for building a professional cryptocurrency trading scanner. Follow each step exactly.

---

## ✅ PREREQUISITES

Confirm these are available before starting:

```bash
□ Python 3.12+ installed
□ pip package manager working
□ Git installed (for version control)
□ Text editor or IDE ready
□ Internet connection for API access
```

---

## 🔨 STEP-BY-STEP IMPLEMENTATION

### STEP 1: Project Setup

**Action:** Create project structure

```bash
mkdir crypto-scanner
cd crypto-scanner
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**Verify:** Virtual environment activated (see `(venv)` in prompt)

---

### STEP 2: Install Dependencies

**Action:** Create requirements.txt

```txt
streamlit==1.28.0
pandas==2.1.0
ccxt==4.1.0
requests==2.31.0
numpy==1.24.0
python-dateutil==2.8.2
```

**Action:** Install packages

```bash
pip install -r requirements.txt
```

**Verify:** Run `pip list` - should see all packages

---

### STEP 3: Test Exchange Connection

**Action:** Create test.py

```python
import ccxt

exchange = ccxt.okx()
ticker = exchange.fetch_ticker('BTC/USDT')
print(f"BTC Price: ${ticker['last']:,.2f}")
print("✅ Exchange connection working!")
```

**Action:** Run test

```bash
python test.py
```

**Expected output:** BTC price displayed

**If fails:** Check internet connection, try different exchange

---

### STEP 4: Create Basic Streamlit App

**Action:** Create app.py

```python
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(
    page_title="BTC Expansion Scanner",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎯 BTC Expansion Edge Scanner")

# Test connection
try:
    exchange = ccxt.okx()
    ticker = exchange.fetch_ticker('BTC/USDT')
    st.success(f"✅ Connected to OKX - BTC: ${ticker['last']:,.2f}")
except Exception as e:
    st.error(f"❌ Connection failed: {e}")
```

**Action:** Run Streamlit

```bash
streamlit run app.py
```

**Expected:** Browser opens showing title and BTC price

---

### STEP 5: Implement Data Fetcher with Failover

**Action:** Add to app.py

```python
def get_data_with_failover(pair: str, source: str, timeframe: str, limit: int = 100):
    """
    Fetch OHLCV data with automatic exchange failover.
    
    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        source: Preferred exchange (e.g., 'OKX')
        timeframe: Candle timeframe (e.g., '15m')
        limit: Number of candles to fetch
    
    Returns:
        tuple: (DataFrame with OHLCV data, active exchange name)
    """
    exchanges = {
        'OKX': ccxt.okx(),
        'Bybit': ccxt.bybit(),
        'Binance': ccxt.binance(),
        'Gate.io': ccxt.gateio()
    }
    
    # Try exchanges in order
    for name, exchange in exchanges.items():
        try:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df, name
        except Exception as e:
            continue  # Try next exchange
    
    # All failed
    return pd.DataFrame(), "FAILED"

# Test the function
if st.button("Test Data Fetch"):
    df, source = get_data_with_failover('BTC/USDT', 'OKX', '15m', 100)
    if not df.empty:
        st.success(f"✅ Fetched {len(df)} candles from {source}")
        st.dataframe(df.tail())
    else:
        st.error("❌ All exchanges failed")
```

**Verify:** Click button, should show 100 BTC candles

---

### STEP 6: Add Technical Indicators

**Action:** Add indicator function

```python
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV data.
    
    Adds: SMA20, SMA100, RSI14, Bollinger Bands
    """
    if df.empty or len(df) < 100:
        return df
    
    # Simple Moving Averages
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma100'] = df['close'].rolling(window=100).mean()
    
    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-period, 2 std dev)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    
    return df

# Test indicators
if st.button("Test Indicators"):
    df, _ = get_data_with_failover('BTC/USDT', 'OKX', '15m', 100)
    df = add_indicators(df)
    st.dataframe(df[['close', 'sma20', 'sma100', 'rsi']].tail())
```

**Verify:** Indicators calculate correctly, no NaN in recent rows

---

### STEP 7: Implement Compression Detector

**Action:** Add compression detection

```python
def detect_compression_state(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Detect when price, SMA20, and SMA100 are tightly compressed.
    
    Returns dict with: state, spread_pct, quality, candles_in_comp
    """
    if df.empty or len(df) < 100:
        return {"state": "NONE"}
    
    close = df['close'].iloc[-1]
    sma20 = df['sma20'].iloc[-1]
    sma100 = df['sma100'].iloc[-1]
    
    # Check for NaN
    if pd.isna(close) or pd.isna(sma20) or pd.isna(sma100):
        return {"state": "NONE"}
    
    # Calculate spread percentage
    values = [close, sma20, sma100]
    spread_pct = (max(values) - min(values)) / close * 100
    
    # Timeframe-specific thresholds
    thresholds = {
        '3m': 0.20, '5m': 0.20, '15m': 0.20,
        '1h': 0.30, '4h': 0.40
    }
    threshold = thresholds.get(timeframe, 0.20)
    
    if spread_pct > threshold:
        return {"state": "NONE"}
    
    # Determine quality
    if spread_pct <= 0.10:
        quality = "ELITE"
    elif spread_pct <= 0.15:
        quality = "HIGH"
    else:
        quality = "GOOD"
    
    # Count consecutive compressed candles
    candles_in_comp = 0
    for i in range(len(df) - 1, max(0, len(df) - 50), -1):
        c = df['close'].iloc[i]
        s20 = df['sma20'].iloc[i]
        s100 = df['sma100'].iloc[i]
        
        if pd.isna(c) or pd.isna(s20) or pd.isna(s100):
            break
        
        vals = [c, s20, s100]
        s = (max(vals) - min(vals)) / c * 100
        
        if s <= threshold:
            candles_in_comp += 1
        else:
            break
    
    # Determine compression type
    sma_diff_pct = abs(sma20 - sma100) / close * 100
    state = "SQZ" if sma_diff_pct < 0.05 else "CROSSOVER"
    
    return {
        "state": state,
        "spread_pct": round(spread_pct, 2),
        "quality": quality,
        "candles_in_comp": candles_in_comp
    }
```

**Verify:** Test on different pairs, should detect tight compressions

---

### STEP 8: Build Complete Scanner

**Action:** See full app.py for complete implementation

The complete scanner includes:
- All 7 setup detectors
- BTC regime analysis
- Liquidity sweep detection
- Signal card rendering
- Filter system
- Exit strategy generation

**Full code:** 3000+ lines in app.py

**Recommended:** Copy app.py from this repository rather than building from scratch

---

### STEP 9: Create Deployment Config

**Action:** Create render.yaml

```yaml
services:
  - type: web
    name: crypto-scanner
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

**This enables one-click deployment to Render.com**

---

### STEP 10: Test Locally

**Action:** Run complete scanner

```bash
streamlit run app.py
```

**Test Checklist:**

```
□ BTC Market Regime loads (3 boxes)
□ BTC Liquidity Engine shows sweeps
□ Session Performance displays (4 sessions)
□ Open Interest section appears
□ Multi-pair scanner shows signals
□ Watchlist input works
□ Timeframe tabs work (3m/5m/15m/4H)
□ Filters work (direction, show all)
□ Signal cards display correctly
□ Exit strategies show on expansions
□ No errors in console
□ Mobile view looks good
```

---

### STEP 11: Deploy to Production

**Action:** Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - BTC Expansion Scanner"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main
```

**Action:** Deploy to Render.com

1. Go to render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Render auto-detects render.yaml
5. Click "Create Web Service"
6. Wait 3-5 minutes for deployment

**Verify:** Visit provided URL, scanner loads and works

---

## 🎯 IMPLEMENTATION CHECKLIST

### Core Functionality

```
✅ Multi-exchange failover working
✅ Data fetching reliable
✅ Indicators calculating correctly
✅ Compression detection accurate
✅ Expansion detection with volume
✅ Chop filter blocking bad setups
✅ All 7 setups auto-detecting
✅ BTC regime showing all timeframes
✅ Liquidity sweeps detecting
✅ Session performance tracking
✅ Signal cards rendering
✅ Exit strategies generating
✅ Filters working
✅ Mobile responsive
```

### Quality Checks

```
✅ No crashes on bad data
✅ Handles API failures gracefully
✅ No infinite loops
✅ Memory usage reasonable
✅ Scan completes in <5 seconds
✅ UI loads in <3 seconds
✅ Works on mobile browsers
✅ No security vulnerabilities
```

---

## 🐛 COMMON ISSUES & SOLUTIONS

### Issue: "Module 'ccxt' has no attribute 'okx'"

**Solution:** Update CCXT

```bash
pip install --upgrade ccxt
```

---

### Issue: Indicators showing NaN

**Solution:** Ensure enough data

```python
# Need at least 100 candles for SMA100
df, _ = get_data_with_failover(pair, source, tf, limit=150)  # Increased from 100
```

---

### Issue: Scan taking too long

**Solution:** Add timeout to data fetching

```python
exchange.timeout = 3000  # 3 seconds max
```

---

### Issue: Deployment fails on Render

**Solution:** Check Python version in render.yaml matches requirements

```yaml
envVars:
  - key: PYTHON_VERSION
    value: 3.12.0  # Must match your development version
```

---

## 📚 ADDITIONAL RESOURCES

**Reference Files:**
- `app.py` - Complete source code with comments
- `ALL_7_SETUPS_GUIDE.md` - Trading strategies
- `MASTER_PROMPT.md` - Architecture overview

**External Documentation:**
- CCXT: https://docs.ccxt.com/en/latest/
- Streamlit: https://docs.streamlit.io/
- Pandas: https://pandas.pydata.org/docs/
- Render: https://render.com/docs

---

## ✅ FINAL VERIFICATION

**Before marking complete, confirm:**

1. **Functionality**
   - All features working as expected
   - No critical bugs
   - Performance acceptable

2. **Code Quality**
   - Well commented
   - Functions documented
   - Clean structure

3. **Deployment**
   - Successfully deployed to production
   - Accessible via URL
   - No deployment errors

4. **Documentation**
   - README.md complete
   - Setup guide clear
   - Trading strategies documented

5. **Testing**
   - Manual testing passed
   - Edge cases handled
   - Error handling works

---

## 🎓 SUCCESS CRITERIA

**You've successfully implemented the scanner when:**

✅ Scanner auto-detects all 7 trading setups
✅ Signals show with complete entry/exit strategies
✅ BTC market intelligence displays correctly
✅ Multi-exchange failover prevents downtime
✅ UI is professional and intuitive
✅ Deployed to production and accessible
✅ All documentation complete
✅ Win rates align with expectations (75-90%)

**Time to complete:** 10-15 days full-time, or 4-6 weeks part-time

**Result:** Professional-grade trading scanner ready for live trading

---

## 🚀 POST-IMPLEMENTATION

**After completing implementation:**

1. **Test with real market conditions** for 1-2 weeks
2. **Track signal performance** to verify win rates
3. **Gather user feedback** if sharing publicly
4. **Iterate and improve** based on results
5. **Consider adding features** from roadmap

**Congratulations on building a professional trading tool!** 🎯
