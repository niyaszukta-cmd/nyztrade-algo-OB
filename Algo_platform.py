"""
NYZTrade · SMC Algo Platform
Live Trading Engine | Dhan Broker | Carry Forward + Intraday Options
Built for NIYAS — NYZTrade
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
import math
import threading
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NYZTrade · Algo Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main { background-color: #0a0d14; }
  section[data-testid="stSidebar"] { background: #0e1117; border-right: 1px solid #1e2130; }
  div[data-testid="metric-container"] {
      background: #131720; border-radius: 8px;
      padding: 14px 16px; border: 1px solid #1e2130;
  }
  .sidebar-hdr {
      font-size: 0.68rem; font-weight: 800; text-transform: uppercase;
      letter-spacing: 0.12em; color: #555; margin-top: 1.2rem; margin-bottom: 0.3rem;
  }
  .status-live   { background:#0d2b1a; border:1px solid #00d4aa; border-radius:6px; padding:8px 14px; color:#00d4aa; font-weight:700; }
  .status-flat   { background:#1a1d29; border:1px solid #444;    border-radius:6px; padding:8px 14px; color:#888; }
  .status-paused { background:#2b1d0a; border:1px solid #f5a623; border-radius:6px; padding:8px 14px; color:#f5a623; font-weight:700; }
  .leg-card      { background:#131720; border:1px solid #1e2130; border-radius:8px; padding:12px; margin-bottom:8px; }
  .pnl-pos { color:#00d4aa; font-weight:700; font-size:1.3rem; }
  .pnl-neg { color:#ff4b6e; font-weight:700; font-size:1.3rem; }
  .order-card    { background:#131720; border:1px solid #1e2130; border-radius:6px; padding:10px 14px; margin-bottom:6px; font-size:0.85rem; }
  .stTabs [data-baseweb="tab"] { background:#131720; border-radius:6px; border:1px solid #1e2130; }
  .stTabs [aria-selected="true"] { background:#0052cc; }
  hr { border-color: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DHAN_BASE = "https://api.dhan.co/v2"

INDICES = {
    "NIFTY 50":   {"security_id": "13",  "lot_size": 75,  "strike_gap": 50,  "symbol": "NIFTY"},
    "BANKNIFTY":  {"security_id": "25",  "lot_size": 30,  "strike_gap": 100, "symbol": "BANKNIFTY"},
    "FINNIFTY":   {"security_id": "27",  "lot_size": 65,  "strike_gap": 50,  "symbol": "FINNIFTY"},
    "MIDCPNIFTY": {"security_id": "442", "lot_size": 75,  "strike_gap": 25,  "symbol": "MIDCPNIFTY"},
    "SENSEX":     {"security_id": "1",   "lot_size": 10,  "strike_gap": 100, "symbol": "SENSEX"},
    "BANKEX":     {"security_id": "12",  "lot_size": 15,  "strike_gap": 100, "symbol": "BANKEX"},
}

PRODUCT_TYPES = {
    "Intraday (MIS)":         "INTRADAY",
    "Carry Forward (CNC/NRML)": "CNC",
    "Margin (MTF)":           "MTF",
}

ORDER_TYPES = {
    "Market":         "MARKET",
    "Limit":          "LIMIT",
    "Stop Loss":      "STOP_LOSS",
    "SL-Market":      "STOP_LOSS_MARKET",
}

EXCHANGE_MAP = {
    "NIFTY 50": "NSE", "BANKNIFTY": "NSE", "FINNIFTY": "NSE",
    "MIDCPNIFTY": "NSE", "SENSEX": "BSE", "BANKEX": "BSE",
}

# ══════════════════════════════════════════════════════════════════════════════
# DHAN CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DhanConfig:
    client_id:    str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcyMTY3OTgxLCJhcHBfaWQiOiJjOTNkM2UwOSIsImlhdCI6MTc3MjA4MTU4MSwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.Kry8jyKMhIR-f1H5R0a2A4I9UHnWdDDE3LMmnXgOiE2U5pXWP3P0Scohw4j4IPvBPy3bPienE2vrWdU78bdJ0w"   # ← update daily

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "algo_running":      False,
        "algo_paused":       False,
        "positions":         [],         # list of open position dicts
        "orders":            [],         # all order records
        "trade_log":         [],         # closed trade records
        "equity_curve":      [],         # (timestamp, equity) tuples
        "initial_capital":   500_000.0,
        "current_capital":   500_000.0,
        "last_signal":       None,
        "last_bar_ts":       None,
        "signal_count":      0,
        "error_log":         [],
        "last_refresh":      None,
        "strategy_params":   {},
        "live_df":           None,
        "carry_positions":   [],         # positions carried overnight
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════════════════════
# DHAN API LAYER
# ══════════════════════════════════════════════════════════════════════════════
class DhanAPI:
    """Thin wrapper around Dhan REST API v2."""

    def __init__(self, config: DhanConfig):
        self.cfg = config
        self.headers = {
            "access-token": config.access_token,
            "client-id":    config.client_id,
            "Content-Type": "application/json",
        }

    # ── Market Data ──────────────────────────────────────────────────────────
    def get_quote(self, security_id: str, exchange: str = "NSE") -> dict | None:
        """Fetch LTP for an instrument."""
        try:
            r = requests.post(
                f"{DHAN_BASE}/marketfeed/ltp",
                headers=self.headers,
                json={"NSE_EQ": [security_id]} if exchange == "NSE" else {"BSE_EQ": [security_id]},
                timeout=8
            )
            if r.ok:
                return r.json()
        except Exception as e:
            self._log_error(f"get_quote: {e}")
        return None

    def get_index_quote(self, security_id: str) -> float | None:
        """Get index LTP via marketfeed."""
        try:
            r = requests.post(
                f"{DHAN_BASE}/marketfeed/ltp",
                headers=self.headers,
                json={"IDX_I": [security_id]},
                timeout=8
            )
            if r.ok:
                data = r.json()
                for segment_data in data.get("data", {}).values():
                    for sid, info in segment_data.items():
                        ltp = info.get("last_price") or info.get("ltp")
                        if ltp:
                            return float(ltp)
        except Exception as e:
            self._log_error(f"get_index_quote: {e}")
        return None

    def fetch_ohlcv(self, security_id: str, from_date: str, to_date: str,
                    interval: str = "5", is_index: bool = True) -> pd.DataFrame | None:
        """Fetch historical OHLCV bars."""
        payload = {
            "securityId":      security_id,
            "exchangeSegment": "IDX_I" if is_index else "NSE_EQ",
            "instrument":      "INDEX"  if is_index else "EQUITY",
            "interval":        interval,
            "fromDate":        from_date,
            "toDate":          to_date,
        }
        try:
            r = requests.post(f"{DHAN_BASE}/charts/intraday",
                              headers=self.headers, json=payload, timeout=20)
            if not r.ok:
                return None
            data = r.json()
            if not data.get("timestamp"):
                return None
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True)
                               .tz_convert("Asia/Kolkata").tz_localize(None),
                "open":   pd.to_numeric(data["open"],   errors="coerce"),
                "high":   pd.to_numeric(data["high"],   errors="coerce"),
                "low":    pd.to_numeric(data["low"],    errors="coerce"),
                "close":  pd.to_numeric(data["close"],  errors="coerce"),
                "volume": pd.to_numeric(data["volume"], errors="coerce"),
            }).dropna().sort_values("timestamp").reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            self._log_error(f"fetch_ohlcv: {e}")
        return None

    def fetch_option_chain(self, symbol: str, expiry_date: str) -> dict | None:
        """Fetch option chain for a symbol and expiry."""
        try:
            r = requests.get(
                f"{DHAN_BASE}/optionchain",
                headers=self.headers,
                params={"UnderlyingScrip": symbol, "ExpiryDate": expiry_date},
                timeout=15
            )
            if r.ok:
                return r.json()
        except Exception as e:
            self._log_error(f"fetch_option_chain: {e}")
        return None

    # ── Order Management ──────────────────────────────────────────────────────
    def place_order(self, order: dict) -> dict | None:
        """Place an order. Returns response dict."""
        try:
            r = requests.post(
                f"{DHAN_BASE}/orders",
                headers=self.headers,
                json=order,
                timeout=10
            )
            resp = r.json() if r.content else {}
            resp["_http_status"] = r.status_code
            return resp
        except Exception as e:
            self._log_error(f"place_order: {e}")
            return {"error": str(e), "_http_status": 0}

    def modify_order(self, order_id: str, updates: dict) -> dict | None:
        try:
            r = requests.put(
                f"{DHAN_BASE}/orders/{order_id}",
                headers=self.headers,
                json=updates,
                timeout=10
            )
            return r.json() if r.content else {}
        except Exception as e:
            self._log_error(f"modify_order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            r = requests.delete(
                f"{DHAN_BASE}/orders/{order_id}",
                headers=self.headers,
                timeout=10
            )
            return r.ok
        except Exception as e:
            self._log_error(f"cancel_order: {e}")
            return False

    def get_positions(self) -> list:
        """Fetch all open positions from Dhan."""
        try:
            r = requests.get(f"{DHAN_BASE}/positions", headers=self.headers, timeout=10)
            if r.ok:
                return r.json().get("data", [])
        except Exception as e:
            self._log_error(f"get_positions: {e}")
        return []

    def get_orders(self) -> list:
        """Fetch today's order book."""
        try:
            r = requests.get(f"{DHAN_BASE}/orders", headers=self.headers, timeout=10)
            if r.ok:
                return r.json().get("data", [])
        except Exception as e:
            self._log_error(f"get_orders: {e}")
        return []

    def get_holdings(self) -> list:
        """Fetch demat holdings (carry forward positions)."""
        try:
            r = requests.get(f"{DHAN_BASE}/holdings", headers=self.headers, timeout=10)
            if r.ok:
                return r.json().get("data", [])
        except Exception as e:
            self._log_error(f"get_holdings: {e}")
        return []

    def get_funds(self) -> dict | None:
        """Fetch available margin / funds."""
        try:
            r = requests.get(f"{DHAN_BASE}/fundlimit", headers=self.headers, timeout=10)
            if r.ok:
                return r.json()
        except Exception as e:
            self._log_error(f"get_funds: {e}")
        return None

    def kill_switch(self) -> bool:
        """Kill switch — cancel all open orders and square off all positions."""
        # Cancel all open orders
        orders = self.get_orders()
        for o in orders:
            if o.get("orderStatus") in ("PENDING", "TRANSIT", "PARTIALLY_FILLED"):
                self.cancel_order(o.get("orderId", ""))
        # Square off all positions
        positions = self.get_positions()
        success = True
        for p in positions:
            if int(p.get("netQty", 0)) != 0:
                side = "SELL" if int(p["netQty"]) > 0 else "BUY"
                order = {
                    "dhanClientId":     self.cfg.client_id,
                    "transactionType":  side,
                    "exchangeSegment":  p.get("exchangeSegment", "NSE_FNO"),
                    "productType":      p.get("productType", "CNC"),
                    "orderType":        "MARKET",
                    "validity":         "DAY",
                    "securityId":       p.get("securityId", ""),
                    "quantity":         abs(int(p["netQty"])),
                    "price":            0,
                    "triggerPrice":     0,
                }
                resp = self.place_order(order)
                if not resp or resp.get("_http_status", 0) not in (200, 201):
                    success = False
        return success

    def _log_error(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state["error_log"].append(f"[{ts}] {msg}")
        if len(st.session_state["error_log"]) > 100:
            st.session_state["error_log"] = st.session_state["error_log"][-100:]

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS  (reused from backtest app)
# ══════════════════════════════════════════════════════════════════════════════
def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def calc_bsp(df: pd.DataFrame, length: int) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    ad = ((2 * df["close"] - df["low"] - df["high"]) / hl) * df["volume"]
    return (ad.rolling(length).sum() / df["volume"].rolling(length).sum()).fillna(0)

def get_atm_strike(spot: float, gap: int) -> int:
    return int(round(spot / gap) * gap)

def get_strike_label(offset: int) -> str:
    if offset == 0:  return "ATM"
    if offset > 0:   return f"ATM+{offset}"
    return f"ATM{offset}"

def compute_strike(spot: float, gap: int, offset: int) -> int:
    return get_atm_strike(spot, gap) + offset * gap

def check_signals(df: pd.DataFrame, params: dict) -> dict:
    """
    Compute latest bar signal from BSP + EMA.
    Returns {"signal": 1/-1/0, "bsp": float, "ema20": float, ...}
    """
    if df is None or len(df) < 50:
        return {"signal": 0, "bsp": 0, "reason": "insufficient data"}

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["bsp"]   = calc_bsp(df, params.get("bsp_length", 21))

    last = df.iloc[-1]
    bsp  = float(last["bsp"])
    e20  = float(last["ema20"])
    e50  = float(last["ema50"])
    cls  = float(last["close"])

    buy_lvl  = params.get("bsp_buy_lvl", 0.08)
    sell_lvl = params.get("bsp_sell_lvl", -0.08)
    mode     = params.get("signal_mode", "Level Hold")
    use_ema  = params.get("ema_filter", True)

    bull = (cls > e20 and e20 > e50) if use_ema else True
    bear = (cls < e20 and e20 < e50) if use_ema else True

    prev_bsp = float(df["bsp"].iloc[-2]) if len(df) > 1 else 0

    signal = 0
    reason = "No signal"

    if mode == "Flip":
        if bsp > buy_lvl and prev_bsp <= buy_lvl and bull:
            signal = 1;  reason = "BSP crossed above buy level"
        elif bsp < sell_lvl and prev_bsp >= sell_lvl and bear:
            signal = -1; reason = "BSP crossed below sell level"
    elif mode == "Level Hold":
        if bsp > buy_lvl and bull:
            signal = 1;  reason = "BSP above buy level + bull trend"
        elif bsp < sell_lvl and bear:
            signal = -1; reason = "BSP below sell level + bear trend"
        elif abs(bsp) < 0.01:
            signal = 0;  reason = "BSP in neutral zone — exit signal"
    else:  # BSP Only
        if bsp > buy_lvl:
            signal = 1;  reason = "BSP above buy level"
        elif bsp < sell_lvl:
            signal = -1; reason = "BSP below sell level"

    return {
        "signal":    signal,
        "reason":    reason,
        "bsp":       round(bsp, 4),
        "ema20":     round(e20, 2),
        "ema50":     round(e50, 2),
        "close":     round(cls, 2),
        "timestamp": last["timestamp"],
    }

# ══════════════════════════════════════════════════════════════════════════════
# ORDER BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_option_order(cfg: DhanConfig, index_name: str, spot: float,
                       leg: dict, lots: int, side: str,
                       product_type: str, order_type: str = "MARKET",
                       limit_price: float = 0) -> dict:
    """
    Build a Dhan order payload for one option leg.
    leg: {"opt_type": "CALL"/"PUT", "offset": int, "strike_lbl": str, "lots": int}
    side: "BUY" or "SELL"
    product_type: "INTRADAY" or "CNC"
    """
    idx_cfg   = INDICES[index_name]
    lot_size  = idx_cfg["lot_size"]
    strike_gap= idx_cfg["strike_gap"]
    symbol    = idx_cfg["symbol"]
    exchange  = EXCHANGE_MAP.get(index_name, "NSE")
    seg       = "NSE_FNO" if exchange == "NSE" else "BSE_FNO"

    strike    = compute_strike(spot, strike_gap, leg["offset"])
    opt_suffix= "CE" if leg["opt_type"] == "CALL" else "PE"

    # Dhan requires security_id for option — we use 0 as placeholder for market order
    # In production, look up security_id from option chain
    return {
        "dhanClientId":    cfg.client_id,
        "transactionType": side,
        "exchangeSegment": seg,
        "productType":     product_type,
        "orderType":       order_type,
        "validity":        "DAY",
        "securityId":      "0",          # ← replace with real security_id from option chain
        "tradingSymbol":   f"{symbol}{strike}{opt_suffix}",
        "quantity":        lots * lot_size,
        "price":           limit_price,
        "triggerPrice":    0,
        "disclosedQuantity": 0,
        "afterMarketOrder":  False,
        "_meta": {                       # internal metadata (not sent to Dhan)
            "index":      index_name,
            "spot":       spot,
            "strike":     strike,
            "opt_type":   leg["opt_type"],
            "opt_suffix": opt_suffix,
            "offset":     leg["offset"],
            "strike_lbl": leg["strike_lbl"],
            "lots":       lots,
            "leg_lots":   leg.get("lots", 1),
            "product":    product_type,
        }
    }

def execute_spread(api: DhanAPI, index_name: str, spot: float,
                   legs: list, scale_lots: int, side: str,
                   product_type: str, order_type: str,
                   dry_run: bool = True) -> list:
    """
    Execute all legs of a spread. Returns list of order responses.
    dry_run=True: simulate without sending to Dhan.
    """
    responses = []
    for leg in legs:
        direction = leg["direction"]  # BUY/SELL per leg
        # For exit, flip direction
        actual_side = direction if side == "ENTER" else ("SELL" if direction == "BUY" else "BUY")
        order = build_option_order(
            api.cfg, index_name, spot, leg,
            scale_lots * leg.get("lots", 1),
            actual_side, product_type, order_type
        )
        meta = order.pop("_meta")
        if dry_run:
            resp = {
                "orderId":   f"DRY-{int(time.time())}-{leg['strike_lbl']}",
                "orderStatus": "SIMULATED",
                "tradingSymbol": order["tradingSymbol"],
                "quantity":  order["quantity"],
                "side":      actual_side,
                "_meta":     meta,
                "_dry_run":  True,
            }
        else:
            resp = api.place_order(order)
            if resp:
                resp["_meta"] = meta
                resp["side"]  = actual_side
        responses.append(resp)
        time.sleep(0.15)  # 150ms between legs to avoid rate limits
    return responses

# ══════════════════════════════════════════════════════════════════════════════
# POSITION TRACKER
# ══════════════════════════════════════════════════════════════════════════════
def open_position(responses: list, spot: float, signal_info: dict,
                  product_type: str, legs: list) -> dict:
    """Create a position record from order responses."""
    return {
        "id":           f"POS-{int(time.time())}",
        "entry_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entry_spot":   spot,
        "signal":       signal_info.get("signal"),
        "bsp_at_entry": signal_info.get("bsp"),
        "reason":       signal_info.get("reason"),
        "product_type": product_type,
        "legs":         legs,
        "orders":       responses,
        "status":       "OPEN",
        "carry_flag":   product_type == "CNC",
        "mtm_pnl":      0.0,
        "realized_pnl": 0.0,
    }

def update_mtm(position: dict, current_prices: dict) -> float:
    """Compute mark-to-market P&L for an open position."""
    total_pnl = 0.0
    for resp in position.get("orders", []):
        meta = resp.get("_meta", {})
        key  = f"{meta.get('strike', '')}{meta.get('opt_suffix', '')}"
        if key in current_prices:
            current_price = current_prices[key]
            entry_price   = meta.get("entry_price", 0)
            qty           = meta.get("lots", 1) * INDICES.get(
                meta.get("index", "NIFTY 50"), {}).get("lot_size", 75)
            direction     = 1 if resp.get("side") == "BUY" else -1
            total_pnl    += direction * (current_price - entry_price) * qty
    return round(total_pnl, 2)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚀 NYZTrade Algo")
    st.markdown('<div class="sidebar-hdr">🔐 Credentials</div>', unsafe_allow_html=True)

    cfg = DhanConfig()
    cfg.client_id    = st.text_input("Client ID",    value=cfg.client_id)
    cfg.access_token = st.text_input("Access Token", value=cfg.access_token, type="password",
                                     help="Paste today's token — regenerates daily from Dhan portal")
    api = DhanAPI(cfg)

    token_ok = cfg.access_token and cfg.access_token != "paste_your_token_here"
    if token_ok:
        st.success("✅ Token set")
    else:
        st.error("❌ Token not configured")

    st.markdown('<div class="sidebar-hdr">📊 Instrument</div>', unsafe_allow_html=True)
    index_name  = st.selectbox("Index", list(INDICES.keys()))
    idx_cfg     = INDICES[index_name]
    lot_size    = idx_cfg["lot_size"]
    strike_gap  = idx_cfg["strike_gap"]
    st.caption(f"Lot: **{lot_size}** · Strike gap: **{strike_gap}**")

    intervals = {"1 Min":"1","3 Min":"3","5 Min":"5","15 Min":"15","25 Min":"25","Daily":"D"}
    interval_lbl = st.selectbox("Candle Interval", list(intervals.keys()), index=2)
    interval     = intervals[interval_lbl]

    st.markdown('<div class="sidebar-hdr">📅 Product Type</div>', unsafe_allow_html=True)
    product_lbl  = st.selectbox("Order Type", list(PRODUCT_TYPES.keys()), index=1,
                                help="CNC/NRML = carry forward overnight. MIS = intraday auto-square-off.")
    product_type = PRODUCT_TYPES[product_lbl]
    is_carry     = product_type == "CNC"

    if not is_carry:
        eod_time_str = st.time_input("EOD Square-off Time", value=datetime.strptime("15:15", "%H:%M").time(),
                                      help="Auto-close all positions at this time (Intraday mode only)")
    else:
        eod_time_str = None
        st.info("📦 Carry Forward: positions roll overnight. Manual exit required or use Kill Switch.")

    order_lbl    = st.selectbox("Execution Order Type", list(ORDER_TYPES.keys()), index=0)
    order_type   = ORDER_TYPES[order_lbl]

    st.markdown('<div class="sidebar-hdr">📌 Option Legs</div>', unsafe_allow_html=True)
    PRESETS = {
        "Custom":           None,
        "Bull Call Spread": [("CALL","ATM",0,"BUY"),  ("CALL","ATM+1",1,"SELL")],
        "Bear Put Spread":  [("PUT","ATM",0,"BUY"),   ("PUT","ATM-1",-1,"SELL")],
        "Long Straddle":    [("CALL","ATM",0,"BUY"),  ("PUT","ATM",0,"BUY")],
        "Short Strangle":   [("CALL","ATM+1",1,"SELL"),("PUT","ATM-1",-1,"SELL")],
        "Iron Condor":      [("PUT","ATM-2",-2,"BUY"),("PUT","ATM-1",-1,"SELL"),
                             ("CALL","ATM+1",1,"SELL"),("CALL","ATM+2",2,"BUY")],
        "Naked Call":       [("CALL","ATM",0,"BUY")],
        "Naked Put":        [("PUT","ATM",0,"BUY")],
    }
    expiry_flag = st.radio("Expiry", ["WEEK","MONTH"], horizontal=True)
    preset      = st.selectbox("Strategy Template", list(PRESETS.keys()), index=6)
    n_legs      = st.selectbox("Legs", [1,2,3,4], index=0)

    offsets_list  = list(range(-10, 11))
    offset_labels = [get_strike_label(o) for o in offsets_list]
    option_legs   = []

    for i in range(n_legs):
        with st.expander(f"Leg {i+1}", expanded=(i == 0)):
            if preset != "Custom" and PRESETS[preset] and i < len(PRESETS[preset]):
                p_type, p_lbl, p_off, p_dir = PRESETS[preset][i]
                d_type = 0 if p_type == "CALL" else 1
                d_dir  = 0 if p_dir  == "BUY"  else 1
                d_lbl  = p_lbl
            else:
                d_type = 0; d_dir = 0; d_lbl = "ATM"

            c1, c2 = st.columns(2)
            with c1:
                lt = st.radio("Type",      ["CE","PE"],       index=d_type, horizontal=True, key=f"lt{i}")
            with c2:
                ld = st.radio("Direction", ["BUY","SELL"],    index=d_dir,  horizontal=True, key=f"ld{i}")
            ls_lbl = st.select_slider("Strike", options=offset_labels,
                                       value=d_lbl if d_lbl in offset_labels else "ATM", key=f"ls{i}")
            ls_off = offsets_list[offset_labels.index(ls_lbl)]
            ll     = st.number_input("Lots", 1, 50, 1, key=f"ll{i}")
            option_legs.append({
                "opt_type":   "CALL" if lt == "CE" else "PUT",
                "direction":  ld,
                "offset":     ls_off,
                "strike_lbl": ls_lbl,
                "lots":       int(ll),
            })

    st.markdown('<div class="sidebar-hdr">⚙️ Strategy Parameters</div>', unsafe_allow_html=True)
    bsp_length   = st.slider("BSP Length",    5, 50, 21)
    bsp_buy_lvl  = st.number_input("BSP Buy Level",  value=0.08,  step=0.01, format="%.2f")
    bsp_sell_lvl = st.number_input("BSP Sell Level", value=-0.08, step=0.01, format="%.2f")
    signal_mode  = st.selectbox("Signal Mode", ["Flip","Level Hold","BSP Only"], index=1)
    ema_filter   = st.checkbox("EMA Filter", value=True, disabled=(signal_mode == "BSP Only"))

    st.markdown('<div class="sidebar-hdr">💼 Capital & Risk</div>', unsafe_allow_html=True)
    init_capital = st.number_input("Capital (₹)", value=500_000, step=50_000)
    if "initial_capital" not in st.session_state or st.session_state["initial_capital"] != init_capital:
        st.session_state["initial_capital"] = init_capital
        st.session_state["current_capital"] = init_capital

    sizing_mode = st.radio("Sizing", ["% of Capital","Fixed Lots"], horizontal=True)
    if sizing_mode == "% of Capital":
        size_pct   = st.slider("Size (%)", 5, 100, 20) / 100
        fixed_lots_val = None
    else:
        fixed_lots_val = st.number_input("Fixed Lots", 1, 100, 1)
        size_pct       = 0.2

    max_positions = st.number_input("Max Open Positions", 1, 10, 3)
    sl_pct        = st.number_input("Stop Loss (%)", 0.0, 50.0, 10.0, step=0.5,
                                     help="Per-trade stop loss as % of entry premium. 0 = disabled.")
    tp_pct        = st.number_input("Take Profit (%)", 0.0, 100.0, 20.0, step=0.5,
                                     help="Per-trade take profit as % of entry premium. 0 = disabled.")

    st.markdown('<div class="sidebar-hdr">🔄 Refresh</div>', unsafe_allow_html=True)
    refresh_secs = st.selectbox("Auto-refresh", [10, 15, 30, 60, 120], index=2,
                                 help="How often to fetch new bar data and check signals")

    st.markdown('<div class="sidebar-hdr">🎭 Execution Mode</div>', unsafe_allow_html=True)
    dry_run = st.checkbox("🧪 Paper Trade (no real orders)", value=True,
                           help="Test strategy logic without sending orders to Dhan.")
    if not dry_run:
        st.warning("⚠️ **LIVE MODE** — Real orders will be placed on Dhan!", icon="🔴")

    strategy_params = {
        "bsp_length": bsp_length, "bsp_buy_lvl": bsp_buy_lvl, "bsp_sell_lvl": bsp_sell_lvl,
        "signal_mode": signal_mode, "ema_filter": ema_filter,
        "sl_pct": sl_pct, "tp_pct": tp_pct,
    }
    st.session_state["strategy_params"] = strategy_params

# ══════════════════════════════════════════════════════════════════════════════
# HEADER + ENGINE CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🚀 NYZTrade · SMC Algo Platform")
st.caption(f"Live Trading Engine | BSP + EMA | {index_name} | Dhan API | {'📦 Carry Forward' if is_carry else '⚡ Intraday'}")

col_status, col_c1, col_c2, col_c3, col_c4 = st.columns([2,1,1,1,1])
with col_status:
    if st.session_state["algo_running"] and not st.session_state["algo_paused"]:
        st.markdown('<div class="status-live">⚡ ALGO RUNNING</div>', unsafe_allow_html=True)
    elif st.session_state["algo_paused"]:
        st.markdown('<div class="status-paused">⏸ PAUSED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-flat">⬤ STOPPED</div>', unsafe_allow_html=True)

with col_c1:
    if st.button("▶ START", type="primary", use_container_width=True,
                  disabled=st.session_state["algo_running"] or not token_ok):
        st.session_state["algo_running"]  = True
        st.session_state["algo_paused"]   = False
        st.rerun()

with col_c2:
    if st.button("⏸ PAUSE", use_container_width=True,
                  disabled=not st.session_state["algo_running"]):
        st.session_state["algo_paused"] = not st.session_state["algo_paused"]
        st.rerun()

with col_c3:
    if st.button("⏹ STOP", use_container_width=True,
                  disabled=not st.session_state["algo_running"]):
        st.session_state["algo_running"] = False
        st.session_state["algo_paused"]  = False
        st.rerun()

with col_c4:
    if st.button("🛑 KILL", use_container_width=True, type="secondary",
                  help="Cancel all orders + square off ALL positions immediately"):
        if token_ok:
            with st.spinner("Executing kill switch…"):
                ok = api.kill_switch()
            st.success("Kill switch executed.") if ok else st.error("Kill switch partially failed — check Dhan app.")
            st.session_state["algo_running"] = False
            st.session_state["positions"]    = []
        else:
            st.error("Configure token first.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_live, tab_pos, tab_orders, tab_trades, tab_risk, tab_carry, tab_log = st.tabs([
    "📡 Live Signal",
    "📂 Positions",
    "📋 Orders",
    "📊 Trade Log",
    "⚖️ Risk & P&L",
    "📦 Carry Forward",
    "🔧 System Log",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNAL + CHART
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### 📡 Live Signal Monitor")

    refresh_col, fetch_col = st.columns([3, 1])
    with fetch_col:
        manual_refresh = st.button("🔄 Refresh Now", use_container_width=True)

    # Fetch recent bars (last 2 days for indicators to warm up)
    fetch_from = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    fetch_to   = datetime.now().strftime("%Y-%m-%d")

    should_fetch = (
        manual_refresh or
        st.session_state["live_df"] is None or
        st.session_state.get("last_refresh") is None or
        (datetime.now() - st.session_state["last_refresh"]).seconds >= refresh_secs
    )

    if should_fetch and token_ok:
        with st.spinner("📡 Fetching live bars…"):
            live_df = api.fetch_ohlcv(
                idx_cfg["security_id"], fetch_from, fetch_to,
                interval=interval if interval != "D" else "5",
                is_index=True
            )
        if live_df is not None:
            st.session_state["live_df"]      = live_df
            st.session_state["last_refresh"] = datetime.now()
    elif not token_ok:
        st.warning("Configure your Dhan token in the sidebar to fetch live data.")

    live_df = st.session_state.get("live_df")

    # ── Signal Check ─────────────────────────────────────────────────────────
    if live_df is not None:
        sig_info = check_signals(live_df, strategy_params)
        last_refresh_str = st.session_state.get("last_refresh")
        if last_refresh_str:
            st.caption(f"Last refresh: {last_refresh_str.strftime('%H:%M:%S')} | Bars: {len(live_df):,}")

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        sig_val  = sig_info["signal"]
        sig_icon = "🟢 BUY" if sig_val == 1 else ("🔴 SELL/EXIT" if sig_val == -1 else "⚪ NEUTRAL")
        k1.metric("Signal",    sig_icon)
        k2.metric("BSP",       f"{sig_info['bsp']:.4f}")
        k3.metric("EMA 20",    f"₹{sig_info['ema20']:,.0f}")
        k4.metric("EMA 50",    f"₹{sig_info['ema50']:,.0f}")
        k5.metric("Last Close",f"₹{sig_info['close']:,.2f}")
        st.caption(f"💬 {sig_info['reason']}")

        # ── Auto-execute logic ────────────────────────────────────────────────
        if st.session_state["algo_running"] and not st.session_state["algo_paused"]:
            open_pos_count = len([p for p in st.session_state["positions"] if p["status"] == "OPEN"])

            # ENTRY
            if sig_val == 1 and open_pos_count < max_positions:
                spot     = sig_info["close"]
                spot_int = int(round(spot))

                # Compute scale lots
                if fixed_lots_val is not None:
                    scale_lots = int(fixed_lots_val)
                else:
                    # Estimate cost from ATM premium (rough: 1% of spot)
                    est_premium = spot * 0.01
                    lot_cost    = est_premium * lot_size
                    scale_lots  = max(int(st.session_state["current_capital"] * size_pct / lot_cost), 1)

                with st.spinner(f"🚀 Placing entry orders for {sig_info['reason']}…"):
                    responses = execute_spread(
                        api, index_name, spot, option_legs, scale_lots,
                        "ENTER", product_type, order_type, dry_run=dry_run
                    )

                position = open_position(responses, spot, sig_info, product_type, option_legs)
                st.session_state["positions"].append(position)
                st.session_state["orders"].extend(responses)
                st.session_state["signal_count"] += 1
                st.success(f"✅ Entered {'(PAPER)' if dry_run else '(LIVE)'}: {len(responses)} leg(s) | Scale lots: {scale_lots}")

            # EXIT — signal reversal
            elif sig_val == -1:
                for pos in st.session_state["positions"]:
                    if pos["status"] == "OPEN":
                        spot = sig_info["close"]
                        with st.spinner("Closing position…"):
                            responses = execute_spread(
                                api, index_name, spot, pos["legs"], 1,
                                "EXIT", pos["product_type"], order_type, dry_run=dry_run
                            )
                        pos["status"]        = "CLOSED"
                        pos["exit_time"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        pos["exit_spot"]     = spot
                        pos["exit_orders"]   = responses
                        pos["realized_pnl"]  = pos.get("mtm_pnl", 0)

                        trade_rec = {
                            "entry_time":  pos["entry_time"],
                            "exit_time":   pos["exit_time"],
                            "entry_spot":  pos["entry_spot"],
                            "exit_spot":   pos["exit_spot"],
                            "strategy":    " | ".join(
                                f"{l['direction']} {l['lots']}L {l['strike_lbl']} {l['opt_type']}"
                                for l in pos["legs"]
                            ),
                            "pnl":         pos["realized_pnl"],
                            "product":     pos["product_type"],
                            "signal":      pos["signal"],
                            "reason":      pos["reason"],
                        }
                        st.session_state["trade_log"].append(trade_rec)
                        st.session_state["orders"].extend(responses)
                        total_pnl = sum(t["pnl"] for t in st.session_state["trade_log"])
                        st.session_state["current_capital"] = init_capital + total_pnl
                        cap_now = st.session_state["current_capital"]
                        st.session_state["equity_curve"].append(
                            {"timestamp": datetime.now().isoformat(), "equity": cap_now}
                        )
                        st.success(f"✅ Exited position | P&L: ₹{pos['realized_pnl']:,.2f}")

        # ── Intraday EOD check ────────────────────────────────────────────────
        if not is_carry and eod_time_str:
            now_time = datetime.now().time()
            eod_t    = eod_time_str
            if now_time >= eod_t:
                open_pos = [p for p in st.session_state["positions"] if p["status"] == "OPEN"]
                if open_pos:
                    st.warning(f"⏰ EOD square-off time ({eod_t}) reached — closing all intraday positions.")
                    for pos in open_pos:
                        pos["status"]       = "CLOSED_EOD"
                        pos["exit_time"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        pos["realized_pnl"] = pos.get("mtm_pnl", 0)

        # ── Live Chart ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📈 Live Chart")
        chart_df = live_df.copy()
        chart_df["ema20"] = ema(chart_df["close"], 20)
        chart_df["ema50"] = ema(chart_df["close"], 50)
        chart_df["bsp"]   = calc_bsp(chart_df, bsp_length)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.70, 0.30], vertical_spacing=0.03,
            subplot_titles=["Price + EMA", "BSP Oscillator"]
        )
        fig.add_trace(go.Candlestick(
            x=chart_df["timestamp"], open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"],  close=chart_df["close"], name=index_name,
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4b6e"
        ), row=1, col=1)
        for e_col, e_color, e_name in [("ema20","#f5a623","EMA 20"), ("ema50","#a78bfa","EMA 50")]:
            fig.add_trace(go.Scatter(
                x=chart_df["timestamp"], y=chart_df[e_col],
                line=dict(color=e_color, width=1), name=e_name
            ), row=1, col=1)

        bsp_colors = chart_df["bsp"].apply(lambda v: "#00d4aa" if v > 0 else "#ff4b6e")
        fig.add_trace(go.Bar(
            x=chart_df["timestamp"], y=chart_df["bsp"],
            name="BSP", marker_color=bsp_colors
        ), row=2, col=1)
        fig.add_hline(y=bsp_buy_lvl,  line_color="#00d4aa", line_dash="dash", row=2, col=1)
        fig.add_hline(y=bsp_sell_lvl, line_color="#ff4b6e", line_dash="dash", row=2, col=1)
        fig.add_hline(y=0, line_color="#333", row=2, col=1)

        # Annotate entry/exit on chart from trade log
        for t in st.session_state["trade_log"][-20:]:
            try:
                et = pd.to_datetime(t["entry_time"])
                fig.add_vline(x=et, line_color="#00d4aa", line_dash="dot", line_width=1, row=1, col=1)
            except: pass

        fig.update_layout(
            height=550, template="plotly_dark",
            paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
            xaxis_rangeslider_visible=False,
            margin=dict(t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))
        )
        fig.update_xaxes(showgrid=True, gridcolor="#1a1d29")
        fig.update_yaxes(showgrid=True, gridcolor="#1a1d29")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet — click **Refresh Now** or configure your token.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pos:
    st.markdown("### 📂 Open Positions")

    # Sync from Dhan (if live)
    if token_ok:
        col_sync, col_sq = st.columns([1, 1])
        with col_sync:
            if st.button("🔄 Sync from Dhan", use_container_width=True):
                dhan_positions = api.get_positions()
                st.session_state["_dhan_positions"] = dhan_positions
        with col_sq:
            if st.button("❌ Square Off ALL", use_container_width=True, type="secondary"):
                if not dry_run:
                    ok = api.kill_switch()
                    st.success("All positions squared off.") if ok else st.error("Partial failure.")
                else:
                    st.info("Paper trade mode — no real orders.")

    # Dhan live positions table
    dhan_pos = st.session_state.get("_dhan_positions", [])
    if dhan_pos:
        st.markdown("#### Dhan Account Positions")
        dp_df = pd.DataFrame(dhan_pos)
        display_cols = [c for c in ["tradingSymbol","netQty","avgCostPrice","lastTradedPrice",
                                     "unrealizedProfit","realizedProfit","productType","exchangeSegment"]
                        if c in dp_df.columns]
        if display_cols:
            st.dataframe(dp_df[display_cols], use_container_width=True)
        else:
            st.json(dhan_pos[:5])

    # Algorithm tracked positions
    algo_positions = st.session_state["positions"]
    open_pos  = [p for p in algo_positions if p["status"] == "OPEN"]
    closed_pos = [p for p in algo_positions if p["status"] != "OPEN"]

    st.markdown(f"#### Algo-Tracked Positions ({len(open_pos)} open, {len(closed_pos)} closed)")

    if not open_pos:
        st.info("No open algo positions.")
    else:
        for i, pos in enumerate(open_pos):
            carry_badge = "📦 CARRY" if pos.get("carry_flag") else "⚡ INTRADAY"
            with st.expander(
                f"{'✅' if pos['signal']==1 else '🔴'} {carry_badge} | "
                f"Entry: {pos['entry_time']} | Spot: ₹{pos['entry_spot']:,.0f}",
                expanded=True
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Entry Spot",   f"₹{pos['entry_spot']:,.0f}")
                c2.metric("Product",      pos["product_type"])
                c3.metric("MTM P&L",      f"₹{pos.get('mtm_pnl', 0):,.2f}")

                st.caption(f"Signal: {pos.get('reason', 'N/A')} | BSP: {pos.get('bsp_at_entry', 0):.4f}")

                st.markdown("**Legs:**")
                for l in pos.get("legs", []):
                    st.markdown(
                        f"  • `{l['direction']}` **{l['lots']}L** {l['strike_lbl']} {l['opt_type']}"
                    )

                if not dry_run:
                    if st.button(f"❌ Close Position #{i+1}", key=f"close_pos_{i}"):
                        spot_now = sig_info.get("close", pos["entry_spot"]) if live_df is not None else pos["entry_spot"]
                        responses = execute_spread(
                            api, index_name, spot_now, pos["legs"], 1,
                            "EXIT", pos["product_type"], order_type, dry_run=dry_run
                        )
                        pos["status"]       = "CLOSED"
                        pos["exit_time"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        pos["exit_spot"]    = spot_now
                        st.success("Position closed.")
                        st.rerun()
                else:
                    if st.button(f"📄 Simulate Close #{i+1}", key=f"sim_close_{i}"):
                        pos["status"]      = "CLOSED"
                        pos["exit_time"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        pos["exit_spot"]   = pos["entry_spot"]
                        pos["realized_pnl"]= 0.0
                        st.info("Paper close recorded.")
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ORDERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_orders:
    st.markdown("### 📋 Order Book")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Fetch from Dhan", key="fetch_orders") and token_ok:
            dhan_orders = api.get_orders()
            st.session_state["_dhan_orders"] = dhan_orders

    dhan_orders = st.session_state.get("_dhan_orders", [])
    if dhan_orders:
        st.markdown("#### Today's Orders (Dhan)")
        o_df = pd.DataFrame(dhan_orders)
        show_cols = [c for c in ["orderId","tradingSymbol","transactionType","orderType",
                                   "quantity","price","orderStatus","productType","createTime"]
                     if c in o_df.columns]
        if show_cols:
            def color_order(v):
                if isinstance(v, str):
                    if v in ("TRADED","COMPLETE"): return "color:#00d4aa"
                    if v in ("REJECTED","CANCELLED"): return "color:#ff4b6e"
                    if v in ("PENDING","TRANSIT"): return "color:#f5a623"
                return ""
            st.dataframe(
                o_df[show_cols].style.applymap(color_order, subset=["orderStatus"] if "orderStatus" in o_df.columns else []),
                use_container_width=True, height=400
            )

    # Algo-generated orders (paper + live)
    algo_orders = st.session_state.get("orders", [])
    if algo_orders:
        st.markdown("#### Algo-Generated Orders")
        ao_rows = []
        for o in algo_orders:
            meta = o.get("_meta", {})
            ao_rows.append({
                "Order ID":  o.get("orderId", "-"),
                "Status":    o.get("orderStatus", "-"),
                "Symbol":    o.get("tradingSymbol", meta.get("index","-")),
                "Side":      o.get("side", "-"),
                "Qty":       o.get("quantity", meta.get("lots","-")),
                "Product":   meta.get("product", "-"),
                "Paper":     "✅" if o.get("_dry_run") else "🔴 LIVE",
            })
        st.dataframe(pd.DataFrame(ao_rows), use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_trades:
    st.markdown("### 📊 Closed Trade Log")
    tlog = st.session_state["trade_log"]
    if not tlog:
        st.info("No closed trades yet.")
    else:
        tdf = pd.DataFrame(tlog)
        tdf["cum_pnl"] = tdf["pnl"].cumsum().round(2)
        tdf["pnl"]     = tdf["pnl"].round(2)

        styled = tdf.style.format({"pnl":"{:.2f}","cum_pnl":"{:.2f}"}).applymap(
            lambda v: ("color:#00d4aa" if isinstance(v,(int,float)) and v>0
                       else "color:#ff4b6e" if isinstance(v,(int,float)) and v<0 else ""),
            subset=["pnl","cum_pnl"]
        )
        st.dataframe(styled, use_container_width=True, height=400)

        c1, c2, c3, c4 = st.columns(4)
        total_pnl = tdf["pnl"].sum()
        wins = tdf[tdf["pnl"] > 0]
        c1.metric("Total P&L",     f"₹{total_pnl:,.2f}")
        c2.metric("Trades",        len(tdf))
        c3.metric("Win Rate",      f"{len(wins)/len(tdf)*100:.1f}%")
        c4.metric("Avg P&L",       f"₹{tdf['pnl'].mean():,.2f}")

        csv = tdf.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv, "nyztrade_trades.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK & EQUITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("### ⚖️ Risk Dashboard & Equity Curve")

    # Funds from Dhan
    if token_ok:
        if st.button("🔄 Fetch Funds from Dhan"):
            funds = api.get_funds()
            st.session_state["_funds"] = funds

    funds = st.session_state.get("_funds")
    if funds:
        st.markdown("#### Account Funds")
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Available Margin",  f"₹{float(funds.get('availabelBalance', 0)):,.2f}")
        fc2.metric("Used Margin",       f"₹{float(funds.get('utilizedAmount',   0)):,.2f}")
        fc3.metric("Total Balance",     f"₹{float(funds.get('sodLimit',         0)):,.2f}")
        fc4.metric("Unrealized P&L",    f"₹{float(funds.get('unrealizedProfit', 0)):,.2f}")

    st.divider()

    # Equity curve
    eq_data = st.session_state["equity_curve"]
    cur_cap  = st.session_state["current_capital"]
    init_cap = st.session_state["initial_capital"]
    total_ret= (cur_cap - init_cap) / init_cap * 100

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Current Capital",  f"₹{cur_cap:,.2f}",
               delta=f"{total_ret:+.2f}%",
               delta_color="normal" if total_ret >= 0 else "inverse")
    ec2.metric("Initial Capital",  f"₹{init_cap:,.2f}")
    ec3.metric("Net P&L",          f"₹{cur_cap - init_cap:,.2f}")

    if eq_data:
        eq_df = pd.DataFrame(eq_data)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        eq_df["peak"]      = eq_df["equity"].cummax()
        eq_df["dd_pct"]    = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100

        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.65,0.35], vertical_spacing=0.05,
                               subplot_titles=["Equity Curve","Drawdown %"])
        fig_eq.add_trace(go.Scatter(
            x=eq_df["timestamp"], y=eq_df["equity"],
            fill="tozeroy", name="Equity",
            line=dict(color="#00d4aa", width=2),
            fillcolor="rgba(0,212,170,0.08)"
        ), row=1, col=1)
        fig_eq.add_hline(y=init_cap, line_color="#555", line_dash="dash", row=1, col=1)
        fig_eq.add_trace(go.Scatter(
            x=eq_df["timestamp"], y=eq_df["dd_pct"],
            fill="tozeroy", name="Drawdown",
            line=dict(color="#ff4b6e", width=1.5),
            fillcolor="rgba(255,75,110,0.1)"
        ), row=2, col=1)
        fig_eq.update_layout(
            height=420, template="plotly_dark",
            paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
            margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Equity curve will populate as trades close.")

    # Risk metrics
    tlog = st.session_state["trade_log"]
    if tlog:
        st.divider()
        st.markdown("#### Risk Metrics")
        pnls  = [t["pnl"] for t in tlog]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]
        gp, gl = sum(wins), abs(sum(losses))

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("Win Rate",      f"{len(wins)/len(pnls)*100:.1f}%")
        rc2.metric("Profit Factor", f"{gp/gl:.2f}" if gl > 0 else "∞")
        rc3.metric("Avg Win",       f"₹{np.mean(wins):,.2f}"   if wins   else "—")
        rc4.metric("Avg Loss",      f"₹{np.mean(losses):,.2f}" if losses else "—")
        rc5.metric("Largest Loss",  f"₹{min(pnls):,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CARRY FORWARD MANAGER
# ══════════════════════════════════════════════════════════════════════════════
with tab_carry:
    st.markdown("### 📦 Carry Forward Manager")
    st.info(
        "Carry Forward (CNC/NRML) positions roll overnight — they are **not** auto-squared at EOD. "
        "Manage them here or use the Kill Switch for emergency exit."
    )

    # Fetch Dhan holdings
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        if st.button("🔄 Load Holdings from Dhan") and token_ok:
            holdings = api.get_holdings()
            st.session_state["_holdings"] = holdings
    with c_col2:
        if st.button("🔄 Load Positions (F&O Carry)") and token_ok:
            carry_pos = [p for p in api.get_positions()
                         if p.get("productType") in ("CNC","NRML") and int(p.get("netQty",0)) != 0]
            st.session_state["_carry_positions"] = carry_pos

    # Holdings (equity)
    holdings = st.session_state.get("_holdings", [])
    if holdings:
        st.markdown("#### Equity Holdings (Overnight)")
        h_df = pd.DataFrame(holdings)
        h_cols = [c for c in ["tradingSymbol","isin","totalQty","avgCostPrice",
                                "lastTradedPrice","totalProfit","dpQty"] if c in h_df.columns]
        if h_cols:
            st.dataframe(h_df[h_cols].style.format(
                {c: "{:.2f}" for c in ["avgCostPrice","lastTradedPrice","totalProfit"] if c in h_cols}
            ), use_container_width=True)

    # F&O carry positions
    carry_pos = st.session_state.get("_carry_positions", [])
    if carry_pos:
        st.markdown("#### F&O Carry Positions (NRML)")
        cp_df = pd.DataFrame(carry_pos)
        cp_cols = [c for c in ["tradingSymbol","netQty","avgCostPrice","lastTradedPrice",
                                 "unrealizedProfit","productType"] if c in cp_df.columns]
        if cp_cols:
            st.dataframe(cp_df[cp_cols], use_container_width=True)

        # Per-position close button
        for j, cp in enumerate(carry_pos):
            sym   = cp.get("tradingSymbol", f"Position {j+1}")
            qty   = int(cp.get("netQty", 0))
            upnl  = float(cp.get("unrealizedProfit", 0))
            side  = "SELL" if qty > 0 else "BUY"
            upnl_color = "pnl-pos" if upnl >= 0 else "pnl-neg"
            st.markdown(
                f"**{sym}** | Qty: `{qty}` | "
                f"<span class='{upnl_color}'>₹{upnl:,.2f}</span>",
                unsafe_allow_html=True
            )
            sq_col, _ = st.columns([1, 3])
            with sq_col:
                if st.button(f"Square Off {sym[:20]}", key=f"sq_carry_{j}"):
                    order = {
                        "dhanClientId":    cfg.client_id,
                        "transactionType": side,
                        "exchangeSegment": cp.get("exchangeSegment", "NSE_FNO"),
                        "productType":     cp.get("productType", "NRML"),
                        "orderType":       "MARKET",
                        "validity":        "DAY",
                        "securityId":      cp.get("securityId", ""),
                        "quantity":        abs(qty),
                        "price":           0,
                        "triggerPrice":    0,
                    }
                    if not dry_run:
                        resp = api.place_order(order)
                        st.success(f"Square-off order placed: {resp.get('orderId','N/A')}")
                    else:
                        st.info(f"Paper mode — would place {side} {abs(qty)} {sym}")

    if not holdings and not carry_pos:
        st.info("Click **Load Holdings** or **Load Positions** to see carry forward data.")

    # Manual add carry position
    st.divider()
    st.markdown("#### ➕ Manually Log a Carry Position")
    with st.form("manual_carry"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        m_sym    = mc1.text_input("Symbol",  placeholder="NIFTY24500CE")
        m_qty    = mc2.number_input("Qty",   min_value=1, value=75)
        m_price  = mc3.number_input("Entry ₹", min_value=0.0, value=0.0, format="%.2f")
        m_side   = mc4.selectbox("Side", ["BUY","SELL"])
        m_submit = st.form_submit_button("Log Position")
        if m_submit and m_sym:
            st.session_state["carry_positions"].append({
                "symbol": m_sym, "qty": m_qty, "entry_price": m_price,
                "side": m_side, "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "product": "CNC"
            })
            st.success(f"Carry position logged: {m_side} {m_qty} {m_sym} @ ₹{m_price:.2f}")

    if st.session_state["carry_positions"]:
        st.markdown("#### Manually Logged Carry Positions")
        st.dataframe(pd.DataFrame(st.session_state["carry_positions"]), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SYSTEM LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown("### 🔧 System & Error Log")

    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        if st.button("🗑 Clear Log"):
            st.session_state["error_log"] = []
            st.rerun()
    with col_l2:
        st.metric("Signals Fired", st.session_state.get("signal_count", 0))

    errors = st.session_state.get("error_log", [])
    if errors:
        st.markdown("**Error / Event Log:**")
        for msg in reversed(errors[-50:]):
            color = "red" if "error" in msg.lower() or "❌" in msg else "gray"
            st.markdown(f"<small style='color:{color}'>{msg}</small>", unsafe_allow_html=True)
    else:
        st.success("✅ No errors logged.")

    st.divider()
    st.markdown("**Session State Summary:**")
    summary = {
        "algo_running":    st.session_state["algo_running"],
        "algo_paused":     st.session_state["algo_paused"],
        "open_positions":  len([p for p in st.session_state["positions"] if p["status"]=="OPEN"]),
        "closed_positions":len([p for p in st.session_state["positions"] if p["status"]!="OPEN"]),
        "total_orders":    len(st.session_state["orders"]),
        "closed_trades":   len(st.session_state["trade_log"]),
        "signals_fired":   st.session_state["signal_count"],
        "current_capital": f"₹{st.session_state['current_capital']:,.2f}",
        "last_refresh":    str(st.session_state.get("last_refresh","—")),
        "dry_run":         dry_run,
        "product_type":    product_type,
    }
    st.json(summary)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH (when algo running)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["algo_running"] and not st.session_state["algo_paused"]:
    time.sleep(0.5)
    st.rerun()
