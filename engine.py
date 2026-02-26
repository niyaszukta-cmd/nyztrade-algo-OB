"""
NYZTrade · SMC Algo Engine
24/7 Background Worker — Render.com
Pine Exact BSP + EMA Signal Logic · Dhan Broker Integration
Built for NIYAS — NYZTrade

How this works:
  - Runs as a Background Worker on Render (not a web server)
  - Wakes up every N minutes via APScheduler
  - Fetches fresh OHLCV from Dhan, computes BSP+EMA, evaluates signal
  - Places orders on Dhan automatically (or paper-trades)
  - Handles carry forward exit automatically next morning
  - Saves all state to state.json (shared with dashboard.py)
  - NEVER needs a browser open
"""

import os, json, time, math, logging, traceback
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.blocking import BlockingScheduler

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("engine.log"),
    ]
)
log = logging.getLogger("nyztrade")
IST = ZoneInfo("Asia/Kolkata")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — all from Render Environment Variables
# ══════════════════════════════════════════════════════════════════════════════
DHAN_CLIENT_ID    = os.environ.get("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
DHAN_BASE         = "https://api.dhan.co/v2"

INDEX_NAME         = os.environ.get("INDEX_NAME",          "NIFTY 50")
INTERVAL           = os.environ.get("INTERVAL",            "5")          # minutes
BSP_LENGTH         = int(os.environ.get("BSP_LENGTH",      "21"))
BSP_BUY_LEVEL      = float(os.environ.get("BSP_BUY_LEVEL", "0.08"))
BSP_SELL_LEVEL     = float(os.environ.get("BSP_SELL_LEVEL","-0.08"))
USE_DAILY_BSP      = os.environ.get("USE_DAILY_BSP",       "true").lower() == "true"
EMA_FILTER         = os.environ.get("EMA_FILTER",          "true").lower() == "true"

PRODUCT_TYPE       = os.environ.get("PRODUCT_TYPE",        "CNC")        # CNC=carry / INTRADAY
TRADE_OPTIONS      = os.environ.get("TRADE_OPTIONS",       "false").lower() == "true"
OPT_TYPE           = os.environ.get("OPT_TYPE",            "CALL")
STRIKE_OFFSET      = int(os.environ.get("STRIKE_OFFSET",   "0"))
EXPIRY_FLAG        = os.environ.get("EXPIRY_FLAG",         "WEEK")

FIXED_LOTS         = int(os.environ.get("FIXED_LOTS",      "1"))
MAX_POSITIONS      = int(os.environ.get("MAX_POSITIONS",   "1"))
SL_PCT             = float(os.environ.get("SL_PCT",        "0"))          # 0 = disabled
TP_PCT             = float(os.environ.get("TP_PCT",        "0"))          # 0 = disabled
INITIAL_CAPITAL    = float(os.environ.get("INITIAL_CAPITAL","500000"))

PAPER_TRADE        = os.environ.get("PAPER_TRADE",         "true").lower() == "true"
CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN","5"))

MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
EOD_SQ_H,       EOD_SQ_M       = 15, 15

STATE_FILE = "state.json"

INDICES = {
    "NIFTY 50":   {"security_id":"13",  "lot_size":75,  "strike_gap":50,  "symbol":"NIFTY"},
    "BANKNIFTY":  {"security_id":"25",  "lot_size":30,  "strike_gap":100, "symbol":"BANKNIFTY"},
    "FINNIFTY":   {"security_id":"27",  "lot_size":65,  "strike_gap":50,  "symbol":"FINNIFTY"},
    "MIDCPNIFTY": {"security_id":"442", "lot_size":75,  "strike_gap":25,  "symbol":"MIDCPNIFTY"},
    "SENSEX":     {"security_id":"1",   "lot_size":10,  "strike_gap":100, "symbol":"SENSEX"},
    "BANKEX":     {"security_id":"12",  "lot_size":15,  "strike_gap":100, "symbol":"BANKEX"},
}

# ══════════════════════════════════════════════════════════════════════════════
# STATE — persisted to disk, shared with dashboard
# ══════════════════════════════════════════════════════════════════════════════
def _default_state() -> dict:
    return {
        "positions":        [],
        "trade_log":        [],
        "equity_curve":     [],
        "capital":          INITIAL_CAPITAL,
        "initial_capital":  INITIAL_CAPITAL,
        "last_signal":      0,
        "last_check_ts":    None,
        "last_bsp":         0,
        "last_close":       0,
        "last_ema20":       0,
        "last_ema50":       0,
        "last_signal_reason": "Not checked yet",
        "engine_status":    "STARTING",
        "signal_log":       [],
        "error_log":        [],
        "total_ticks":      0,
    }

def load_state() -> dict:
    s = _default_state()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s.update(json.load(f))
        except Exception as e:
            log.warning(f"State load error: {e}")
    return s

def save_state(s: dict):
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(s, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        log.error(f"State save error: {e}")

def push_signal_log(s: dict, msg: str, level: str = "INFO"):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    s["signal_log"].append({"ts": ts, "msg": msg, "level": level})
    s["signal_log"] = s["signal_log"][-300:]

def push_error(s: dict, msg: str):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    s["error_log"].append({"ts": ts, "msg": msg})
    s["error_log"] = s["error_log"][-100:]
    log.error(msg)

# ══════════════════════════════════════════════════════════════════════════════
# DHAN API CALLS
# ══════════════════════════════════════════════════════════════════════════════
def _hdrs() -> dict:
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    DHAN_CLIENT_ID,
        "Content-Type": "application/json",
    }

def _parse_ohlcv(data: dict) -> pd.DataFrame | None:
    if not data or "timestamp" not in data or not data["timestamp"]:
        return None
    try:
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True)
                           .tz_convert("Asia/Kolkata").tz_localize(None),
            "open":   pd.to_numeric(data.get("open",  []), errors="coerce"),
            "high":   pd.to_numeric(data.get("high",  []), errors="coerce"),
            "low":    pd.to_numeric(data.get("low",   []), errors="coerce"),
            "close":  pd.to_numeric(data.get("close", []), errors="coerce"),
            "volume": pd.to_numeric(data.get("volume",[]), errors="coerce"),
        }).dropna().sort_values("timestamp").reset_index(drop=True)
        return df if not df.empty else None
    except Exception as e:
        log.error(f"_parse_ohlcv: {e}")
        return None

def fetch_intraday(security_id: str, from_date: str, to_date: str,
                   interval: str) -> pd.DataFrame | None:
    payload = {
        "securityId":      security_id,
        "exchangeSegment": "IDX_I",
        "instrument":      "INDEX",
        "interval":        interval,
        "fromDate":        from_date + " 09:15:00",
        "toDate":          to_date   + " 15:30:00",
    }
    try:
        r = requests.post(f"{DHAN_BASE}/charts/intraday",
                          headers=_hdrs(), json=payload, timeout=25)
        if not r.ok:
            log.error(f"Intraday fetch HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return _parse_ohlcv(data)
    except Exception as e:
        log.error(f"fetch_intraday: {e}")
        return None

def fetch_daily(security_id: str, from_date: str, to_date: str) -> pd.DataFrame | None:
    payload = {
        "securityId":      security_id,
        "exchangeSegment": "IDX_I",
        "instrument":      "INDEX",
        "fromDate":        from_date,
        "toDate":          to_date,
        "expiryCode":      0,
    }
    try:
        r = requests.post(f"{DHAN_BASE}/charts/historical",
                          headers=_hdrs(), json=payload, timeout=25)
        if not r.ok:
            return None
        data = r.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return _parse_ohlcv(data)
    except Exception as e:
        log.error(f"fetch_daily: {e}")
        return None

def place_order_dhan(order: dict) -> dict:
    if PAPER_TRADE:
        oid = f"PAPER-{int(time.time())}"
        log.info(f"[PAPER] {order.get('transactionType')} "
                 f"{order.get('quantity')} qty | sym={order.get('tradingSymbol','?')}")
        return {"orderId": oid, "orderStatus": "PAPER_TRADED", "_paper": True}
    try:
        r = requests.post(f"{DHAN_BASE}/orders",
                          headers=_hdrs(), json=order, timeout=15)
        resp = r.json() if r.content else {}
        resp["_http"] = r.status_code
        log.info(f"Order response: {resp}")
        return resp
    except Exception as e:
        log.error(f"place_order: {e}")
        return {"error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS — exact Pine Script replicas
# ══════════════════════════════════════════════════════════════════════════════
def ema_series(s: pd.Series, p: int) -> pd.Series:
    """Pine ta.ema() = ewm span, adjust=False"""
    return s.ewm(span=p, adjust=False).mean()

def calc_bsp_intraday(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Pine exact:
      ad = close==high and close==low or high==low ? 0
             : ((2*close-low-high)/(high-low)) * volume
      mf = sum(ad, length) / sum(volume, length)
    """
    hl   = df["high"] - df["low"]
    flat = (hl == 0) | ((df["close"] == df["high"]) & (df["close"] == df["low"]))
    ad   = np.where(
        flat, 0.0,
        ((2*df["close"] - df["low"] - df["high"]) / hl.replace(0, 1.0)) * df["volume"]
    )
    ad = pd.Series(ad, index=df.index)
    return (ad.rolling(length).sum() / df["volume"].rolling(length).sum()).fillna(0)

def calc_bsp_from_daily(df_intra: pd.DataFrame, df_daily: pd.DataFrame,
                        length: int) -> pd.Series:
    """
    Pine: bsp = request.security(syminfo.tickerid, 'D', mf)
    Compute BSP on daily bars → stamp each intraday bar with that day's value.
    Falls back to intraday BSP if daily data missing.
    """
    if df_daily is None or df_daily.empty:
        return calc_bsp_intraday(df_intra, length)
    hl   = df_daily["high"] - df_daily["low"]
    flat = (hl == 0) | ((df_daily["close"]==df_daily["high"]) & (df_daily["close"]==df_daily["low"]))
    ad   = np.where(
        flat, 0.0,
        ((2*df_daily["close"] - df_daily["low"] - df_daily["high"]) / hl.replace(0, 1.0)) * df_daily["volume"]
    )
    ad    = pd.Series(ad, index=df_daily.index)
    bspd  = (ad.rolling(length).sum() / df_daily["volume"].rolling(length).sum()).fillna(0)
    d2b   = dict(zip(pd.to_datetime(df_daily["timestamp"]).dt.normalize(), bspd.values))
    dates = pd.to_datetime(df_intra["timestamp"]).dt.normalize()
    return pd.Series(dates, index=df_intra.index).map(d2b).ffill().fillna(0)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_signal(df: pd.DataFrame, df_daily: pd.DataFrame | None) -> dict:
    """
    Compute indicators on latest bars and return signal + all values.
    Uses Pine Exact logic: same conditions as Pine longCondition / shortCondition.
    """
    min_bars = max(BSP_LENGTH + 10, 60)
    if df is None or len(df) < min_bars:
        return {
            "signal": 0, "reason": f"Insufficient bars ({len(df) if df is not None else 0} < {min_bars})",
            "bsp": 0, "ema20": 0, "ema50": 0, "close": 0, "timestamp": ""
        }

    df = df.copy().reset_index(drop=True)
    df["ema20"] = ema_series(df["close"], 20)
    df["ema50"] = ema_series(df["close"], 50)

    if USE_DAILY_BSP and df_daily is not None:
        df["bsp"] = calc_bsp_from_daily(df, df_daily, BSP_LENGTH)
    else:
        df["bsp"] = calc_bsp_intraday(df, BSP_LENGTH)

    last  = df.iloc[-1]
    bsp   = float(last["bsp"])
    e20   = float(last["ema20"])
    e50   = float(last["ema50"])
    cls   = float(last["close"])
    ts    = str(last["timestamp"])

    # EMA trend conditions (Pine: close > ema20 AND ema20 > ema50)
    bull = (cls > e20 and e20 > e50) if EMA_FILTER else True
    bear = (cls < e20 and e20 < e50) if EMA_FILTER else True

    signal = 0
    reason = "Neutral — no condition met"

    # Pine Exact (default):
    # longCondition  = bsp > bspBuyLevel  AND close > ema20 AND ema20 > ema50
    # shortCondition = bsp < bspSellLevel AND close < ema20 AND ema20 < ema50
    if bsp > BSP_BUY_LEVEL and bull:
        signal = 1
        reason = (f"BUY: BSP {bsp:.4f} > {BSP_BUY_LEVEL}"
                  + (f" | Close {cls:.0f} > EMA20 {e20:.0f} > EMA50 {e50:.0f}" if EMA_FILTER else ""))
    elif bsp < BSP_SELL_LEVEL and bear:
        signal = -1
        reason = (f"EXIT: BSP {bsp:.4f} < {BSP_SELL_LEVEL}"
                  + (f" | Close {cls:.0f} < EMA20 {e20:.0f} < EMA50 {e50:.0f}" if EMA_FILTER else ""))

    return {
        "signal":    signal,
        "reason":    reason,
        "bsp":       round(bsp, 4),
        "ema20":     round(e20, 2),
        "ema50":     round(e50, 2),
        "close":     round(cls, 2),
        "timestamp": ts,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ORDER BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_option_order(side: str, spot: float, idx_cfg: dict) -> dict:
    """
    Build Dhan order payload for a single option leg.
    securityId "0" is a placeholder — for live trading you must fetch the
    real security_id from Dhan option chain API using fetch_option_chain().
    Paper trade works fine with "0".
    """
    gap    = idx_cfg["strike_gap"]
    atm    = int(round(spot / gap) * gap)
    strike = atm + STRIKE_OFFSET * gap
    suffix = "CE" if OPT_TYPE == "CALL" else "PE"
    sym    = idx_cfg["symbol"]
    exch   = "BSE_FNO" if INDEX_NAME in ("SENSEX","BANKEX") else "NSE_FNO"
    return {
        "dhanClientId":      DHAN_CLIENT_ID,
        "transactionType":   side,                     # "BUY" or "SELL"
        "exchangeSegment":   exch,
        "productType":       PRODUCT_TYPE,             # "CNC" or "INTRADAY"
        "orderType":         "MARKET",
        "validity":          "DAY",
        "securityId":        "0",                      # ← replace with real ID for live
        "tradingSymbol":     f"{sym}{strike}{suffix}",
        "quantity":          FIXED_LOTS * idx_cfg["lot_size"],
        "price":             0,
        "triggerPrice":      0,
        "disclosedQuantity": 0,
        "afterMarketOrder":  False,
    }

def build_index_order(side: str, idx_cfg: dict) -> dict:
    """Futures-style index order (no options)."""
    exch = "BSE_FNO" if INDEX_NAME in ("SENSEX","BANKEX") else "NSE_FNO"
    return {
        "dhanClientId":    DHAN_CLIENT_ID,
        "transactionType": side,
        "exchangeSegment": exch,
        "productType":     PRODUCT_TYPE,
        "orderType":       "MARKET",
        "validity":        "DAY",
        "securityId":      idx_cfg["security_id"],
        "tradingSymbol":   idx_cfg["symbol"],
        "quantity":        FIXED_LOTS * idx_cfg["lot_size"],
        "price":           0, "triggerPrice": 0, "disclosedQuantity": 0,
        "afterMarketOrder": False,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════════════════════════
def market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5: return False
    t = (now.hour, now.minute)
    return (MARKET_OPEN_H, MARKET_OPEN_M) <= t <= (MARKET_CLOSE_H, MARKET_CLOSE_M)

def eod_time() -> bool:
    now = datetime.now(IST)
    t   = (now.hour, now.minute)
    return (EOD_SQ_H, EOD_SQ_M) <= t <= (MARKET_CLOSE_H, MARKET_CLOSE_M)

# ══════════════════════════════════════════════════════════════════════════════
# POSITION ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
def close_position(pos: dict, current_price: float, reason: str,
                   state: dict, idx_cfg: dict):
    """Close an open position: place exit order, record P&L, update state."""
    if TRADE_OPTIONS:
        order = build_option_order("SELL", current_price, idx_cfg)
    else:
        order = build_index_order("SELL", idx_cfg)

    resp = place_order_dhan(order)

    lot_size = idx_cfg["lot_size"]
    qty      = pos.get("lots", FIXED_LOTS) * lot_size
    entry_p  = float(pos.get("entry_price", current_price))
    pnl      = (current_price - entry_p) * qty
    pnl      = round(pnl, 2)

    pos.update({
        "status":       "CLOSED",
        "exit_time":    datetime.now(IST).isoformat(),
        "exit_price":   round(current_price, 2),
        "exit_reason":  reason,
        "realized_pnl": pnl,
        "exit_order":   resp,
    })

    state["capital"] = round(state.get("capital", INITIAL_CAPITAL) + pnl, 2)

    state["trade_log"].append({
        "id":          pos["id"],
        "entry_time":  pos.get("entry_time", ""),
        "exit_time":   pos["exit_time"],
        "entry_price": entry_p,
        "exit_price":  round(current_price, 2),
        "lots":        pos.get("lots", FIXED_LOTS),
        "pnl":         pnl,
        "reason":      reason,
        "carry":       pos.get("carry", False),
        "paper":       PAPER_TRADE,
    })

    state["equity_curve"].append({
        "ts":     datetime.now(IST).isoformat(),
        "equity": state["capital"],
    })

    push_signal_log(state,
        f"CLOSED {pos['id']} | P&L ₹{pnl:,.2f} | {reason} | "
        f"{'PAPER' if PAPER_TRADE else 'LIVE'}",
        "EXIT"
    )
    log.info(f"Position closed: {pos['id']} | P&L ₹{pnl:,.2f} | {reason}")

def check_sl_tp(pos: dict, current_price: float) -> tuple[bool, str]:
    """Returns (should_exit, reason) if SL or TP hit."""
    ep = float(pos.get("entry_price", 0))
    if ep == 0:
        return False, ""
    chg = (current_price - ep) / ep * 100
    if SL_PCT > 0 and chg <= -SL_PCT:
        return True, f"Stop Loss {chg:.1f}%"
    if TP_PCT > 0 and chg >= TP_PCT:
        return True, f"Take Profit +{chg:.1f}%"
    return False, ""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE TICK
# ══════════════════════════════════════════════════════════════════════════════
def engine_tick():
    """
    Core tick — runs every CHECK_INTERVAL_MIN minutes via APScheduler.

    Sequence:
    1. Load state
    2. Guard: market hours check
    3. Fetch OHLCV (intraday + daily if needed)
    4. Compute signal (Pine Exact)
    5. EOD square-off for intraday positions
    6. Carry position management:
       - Check open carry positions for SL/TP
       - If exit signal fires → close carry position (KEY: works even after overnight)
    7. New entry if BUY signal and under position limit
    8. Save state
    """
    state = load_state()
    state["total_ticks"] = state.get("total_ticks", 0) + 1
    now_ist = datetime.now(IST)
    now_str = now_ist.strftime("%Y-%m-%d %H:%M:%S")
    state["last_check_ts"] = now_str

    log.info(f"── Tick #{state['total_ticks']} at {now_str} IST ──")

    # ── 1. Market hours guard ────────────────────────────────────────────────
    if not market_open():
        log.info("Outside market hours — skip")
        state["engine_status"] = "MARKET_CLOSED"
        push_signal_log(state, f"Market closed — {now_str}", "INFO")
        save_state(state)
        return

    state["engine_status"] = "RUNNING"

    # ── 2. Validate config ───────────────────────────────────────────────────
    idx_cfg = INDICES.get(INDEX_NAME)
    if not idx_cfg:
        push_error(state, f"Unknown INDEX_NAME: {INDEX_NAME}")
        save_state(state)
        return
    if not DHAN_ACCESS_TOKEN:
        push_error(state, "DHAN_ACCESS_TOKEN env var is empty — set it in Render dashboard")
        save_state(state)
        return

    # ── 3. Fetch data ────────────────────────────────────────────────────────
    today     = now_ist.strftime("%Y-%m-%d")
    from_date = (now_ist - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        df = fetch_intraday(idx_cfg["security_id"], from_date, today, INTERVAL)
        if df is None or df.empty:
            push_error(state, f"No intraday data from Dhan for {INDEX_NAME}")
            save_state(state)
            return
        log.info(f"Fetched {len(df)} bars ({INDEX_NAME} {INTERVAL}m)")

        df_daily = None
        if USE_DAILY_BSP:
            df_daily = fetch_daily(idx_cfg["security_id"], from_date, today)
            if df_daily is not None:
                log.info(f"Daily BSP: {len(df_daily)} daily bars")

    except Exception as e:
        push_error(state, f"Data fetch error: {e}\n{traceback.format_exc()}")
        save_state(state)
        return

    # ── 4. Compute signal ────────────────────────────────────────────────────
    try:
        sig = evaluate_signal(df, df_daily)
    except Exception as e:
        push_error(state, f"Signal error: {e}\n{traceback.format_exc()}")
        save_state(state)
        return

    state["last_signal"]       = sig["signal"]
    state["last_bsp"]          = sig["bsp"]
    state["last_close"]        = sig["close"]
    state["last_ema20"]        = sig["ema20"]
    state["last_ema50"]        = sig["ema50"]
    state["last_signal_reason"]= sig["reason"]

    sig_name = {1:"BUY", -1:"EXIT", 0:"NEUTRAL"}.get(sig["signal"],"?")
    log.info(f"Signal: {sig_name} | BSP={sig['bsp']:.4f} | "
             f"Close=₹{sig['close']:,.0f} | {sig['reason']}")
    push_signal_log(state,
        f"{sig_name} | BSP={sig['bsp']:.4f} Close=₹{sig['close']:,.0f} "
        f"EMA20=₹{sig['ema20']:,.0f} EMA50=₹{sig['ema50']:,.0f} | {sig['reason']}",
        sig_name
    )

    open_positions = [p for p in state["positions"] if p.get("status") == "OPEN"]
    cur_price      = sig["close"]

    # ── 5. EOD square-off (intraday only) ────────────────────────────────────
    if PRODUCT_TYPE == "INTRADAY" and eod_time():
        for pos in open_positions:
            log.info(f"EOD square-off: {pos['id']}")
            close_position(pos, cur_price, "EOD Square-off", state, idx_cfg)
        save_state(state)
        return

    # ── 6. Carry position management ─────────────────────────────────────────
    # This is the KEY feature: carry positions are managed AUTOMATICALLY
    # even if the app was closed overnight. Engine wakes up next morning,
    # checks carry positions, and closes on exit signal or SL/TP.
    for pos in open_positions:
        # Check stop loss / take profit first
        sl_tp_hit, sl_tp_reason = check_sl_tp(pos, cur_price)
        if sl_tp_hit:
            close_position(pos, cur_price, sl_tp_reason, state, idx_cfg)
            continue

        # Check exit signal (Pine shortCondition)
        if sig["signal"] == -1:
            close_position(pos, cur_price, "Exit Signal (Pine shortCondition)", state, idx_cfg)
            continue

        log.info(f"Carry position {pos['id']} still OPEN — no exit condition met")

    # ── 7. New entry ─────────────────────────────────────────────────────────
    open_now = len([p for p in state["positions"] if p.get("status") == "OPEN"])

    if sig["signal"] == 1 and open_now < MAX_POSITIONS:
        log.info(f"BUY — entering position ({open_now}/{MAX_POSITIONS} open)")

        if TRADE_OPTIONS:
            order = build_option_order("BUY", cur_price, idx_cfg)
        else:
            order = build_index_order("BUY", idx_cfg)

        resp = place_order_dhan(order)
        pos_id = f"POS-{now_ist.strftime('%Y%m%d%H%M%S')}"

        pos = {
            "id":             pos_id,
            "entry_time":     now_ist.isoformat(),
            "entry_price":    cur_price,
            "bsp_at_entry":   sig["bsp"],
            "ema20_at_entry": sig["ema20"],
            "ema50_at_entry": sig["ema50"],
            "lots":           FIXED_LOTS,
            "status":         "OPEN",
            "carry":          PRODUCT_TYPE == "CNC",
            "product":        PRODUCT_TYPE,
            "entry_order":    resp,
            "signal_reason":  sig["reason"],
        }
        state["positions"].append(pos)
        state["equity_curve"].append({
            "ts": now_ist.isoformat(), "equity": state["capital"]
        })
        push_signal_log(state,
            f"ENTERED {pos_id} | ₹{cur_price:,.2f} | "
            f"{'CNC/Carry' if PRODUCT_TYPE=='CNC' else 'INTRADAY'} | "
            f"{'PAPER' if PAPER_TRADE else 'LIVE'} | Order={resp.get('orderId','?')}",
            "BUY"
        )

    elif sig["signal"] == 1 and open_now >= MAX_POSITIONS:
        log.info(f"BUY signal but max positions ({MAX_POSITIONS}) reached — skip")
        push_signal_log(state, f"BUY signal skipped — max {MAX_POSITIONS} positions open", "INFO")

    elif sig["signal"] == -1 and open_now == 0:
        log.info("EXIT signal but no open position")

    elif sig["signal"] == 0:
        log.info("Neutral — no action")

    # ── 8. Save state ────────────────────────────────────────────────────────
    open_final = len([p for p in state["positions"] if p.get("status") == "OPEN"])
    save_state(state)
    log.info(f"Tick done | Capital ₹{state['capital']:,.2f} | "
             f"Open: {open_final} | Trades: {len(state['trade_log'])}")


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 64)
    log.info("NYZTrade · SMC Algo Engine")
    log.info(f"  Index:       {INDEX_NAME}")
    log.info(f"  Interval:    {INTERVAL}m candles")
    log.info(f"  BSP TF:      {'Daily (Pine default)' if USE_DAILY_BSP else 'Chart interval'}")
    log.info(f"  BSP Levels:  Buy>{BSP_BUY_LEVEL}  Sell<{BSP_SELL_LEVEL}")
    log.info(f"  EMA Filter:  {EMA_FILTER}")
    log.info(f"  Product:     {PRODUCT_TYPE}  ({'Carry Forward' if PRODUCT_TYPE=='CNC' else 'Intraday'})")
    log.info(f"  Options:     {'Yes — ' + OPT_TYPE + ' ' + ('ATM' if STRIKE_OFFSET==0 else f'ATM+{STRIKE_OFFSET}') if TRADE_OPTIONS else 'No (Index mode)'}")
    log.info(f"  Lots:        {FIXED_LOTS}  MaxPos: {MAX_POSITIONS}")
    log.info(f"  SL/TP:       {'SL ' + str(SL_PCT) + '%' if SL_PCT > 0 else 'No SL'}  "
             f"{'TP ' + str(TP_PCT) + '%' if TP_PCT > 0 else 'No TP'}")
    log.info(f"  Paper Trade: {PAPER_TRADE}")
    log.info(f"  Tick every:  {CHECK_INTERVAL_MIN} min")
    log.info("=" * 64)

    # Initialise state file if missing
    if not os.path.exists(STATE_FILE):
        s = _default_state()
        s["engine_status"] = "STARTING"
        save_state(s)
        log.info("Initialised state.json")

    # Run one tick immediately on startup
    try:
        engine_tick()
    except Exception as e:
        log.error(f"Startup tick failed: {e}\n{traceback.format_exc()}")

    # Schedule recurring ticks
    sched = BlockingScheduler(timezone="Asia/Kolkata")
    sched.add_job(
        engine_tick,
        trigger="interval",
        minutes=CHECK_INTERVAL_MIN,
        id="tick",
        max_instances=1,
        misfire_grace_time=60,
    )
    log.info(f"Scheduler running — tick every {CHECK_INTERVAL_MIN} min")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Engine stopped.")


if __name__ == "__main__":
    main()
