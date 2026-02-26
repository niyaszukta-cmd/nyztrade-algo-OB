"""
NYZTrade SMC Liquidity Lens Backtester
Index Options Backtest | Dhan API | Built for NIYAS
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NYZTrade | SMC Liquidity Lens",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="metric-container"] {
        background: #1a1d29; border-radius: 8px;
        padding: 12px; border: 1px solid #2d3139;
    }
    .sidebar-header {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #666;
        margin-top: 1.2rem; margin-bottom: 0.3rem;
    }
    .token-box {
        background: #1a1d29; border: 1px solid #2d3139;
        border-radius: 8px; padding: 12px; margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1d29; border-radius: 6px; border: 1px solid #2d3139;
    }
    .stTabs [aria-selected="true"] { background: #0052cc; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DHAN CONFIG — identical pattern to GEX Dashboard
# Update access_token daily. client_id never changes.
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass

@dataclass
class DhanConfig:
    client_id:    str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcyMTY3OTgxLCJhcHBfaWQiOiJjOTNkM2UwOSIsImlhdCI6MTc3MjA4MTU4MSwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.Kry8jyKMhIR-f1H5R0a2A4I9UHnWdDDE3LMmnXgOiE2U5pXWP3P0Scohw4j4IPvBPy3bPienE2vrWdU78bdJ0w"   # ← paste fresh token here daily

DHAN_BASE = "https://api.dhan.co/v2"

INDICES = {
    "NIFTY 50":   {"security_id": 13,  "lot_size": 75,  "strike_gap": 50},
    "BANKNIFTY":  {"security_id": 25,  "lot_size": 30,  "strike_gap": 100},
    "FINNIFTY":   {"security_id": 27,  "lot_size": 65,  "strike_gap": 50},
    "MIDCPNIFTY": {"security_id": 442, "lot_size": 75,  "strike_gap": 25},
    "SENSEX":     {"security_id": 1,   "lot_size": 10,  "strike_gap": 100},
    "BANKEX":     {"security_id": 12,  "lot_size": 15,  "strike_gap": 100},
}


# ══════════════════════════════════════════════════════════════════════════════
# DHAN FETCHER CLASS — mirrors GEX Dashboard's UnifiedOptionsFetcher exactly
# No token passed anywhere. self.headers used on every request.
# ══════════════════════════════════════════════════════════════════════════════

class DhanFetcher:
    def __init__(self, config: DhanConfig):
        self.config = config
        # Identical header structure to GEX Dashboard
        self.headers = {
            "access-token": config.access_token,
            "client-id":    config.client_id,
            "Content-Type": "application/json",
        }
        self.base_url = DHAN_BASE

    def _handle_error(self, r: requests.Response) -> None:
        code, body = r.status_code, r.text[:400]
        if code == 401:
            st.error("❌ **401** — Token expired/invalid. Update `access_token` in DhanConfig and restart.")
        elif code == 429:
            st.error("❌ **429** — Rate limit hit. Wait 60 s and retry.")
        elif code == 400:
            st.error(f"❌ **400 Bad Request** — `{body}`")
        else:
            st.error(f"❌ **HTTP {code}** — `{body}`")

    def fetch_index_ohlcv(self, security_id: int, from_date: str,
                           to_date: str, interval: str,
                           debug: bool = False) -> pd.DataFrame | None:
        """
        Fetch index OHLCV — same approach as GEX dashboard fetch_rolling_data.

        Intraday (1/5/15/25/60):  POST /charts/intraday
                                  exchangeSegment=IDX_I, instrument=INDEX
                                  fromDate/toDate include time  HH:MM:SS

        Daily (D):                POST /charts/historical
                                  exchangeSegment=IDX_I, instrument=INDEX
        """
        # Dhan natively supports: 1, 5, 15, 25, 60
        # 3-min and 10-min are fetched as 1-min and resampled client-side
        RESAMPLE_MAP = {"3": ("1", 3), "10": ("1", 10)}
        resample_to = None
        fetch_interval = interval
        if interval in RESAMPLE_MAP:
            fetch_interval, resample_to = RESAMPLE_MAP[interval]

        intraday_set = {"1", "5", "15", "25", "60"}

        if fetch_interval in intraday_set:
            from_dt    = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt      = datetime.strptime(to_date,   "%Y-%m-%d")
            total_days = max((to_dt - from_dt).days, 1)
            all_dfs, fetched = [], 0
            prog = st.progress(0, text="Fetching intraday data…")
            cur  = from_dt

            while cur <= to_dt:
                end = min(cur + timedelta(days=90), to_dt)
                payload = {
                    "securityId":      security_id,
                    "exchangeSegment": "IDX_I",
                    "instrument":      "INDEX",
                    "interval":        interval,
                    "fromDate":        cur.strftime("%Y-%m-%d") + " 09:15:00",
                    "toDate":          end.strftime("%Y-%m-%d") + " 15:30:00",
                }
                if debug:
                    st.write("📤 Intraday payload:", payload)
                try:
                    r = requests.post(f"{self.base_url}/charts/intraday",
                                      headers=self.headers, json=payload, timeout=30)
                    if debug:
                        st.write(f"📥 Response {r.status_code}:", r.text[:600])
                    if not r.ok:
                        self._handle_error(r)
                        prog.empty()
                        return None
                    data = r.json()
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                    df = _parse_ohlcv(data)
                    if df is not None:
                        all_dfs.append(df)
                except Exception as e:
                    st.error(f"Request error: {e}")
                    prog.empty()
                    return None

                fetched += max((end - cur).days, 1)
                prog.progress(min(fetched / total_days, 1.0),
                              text=f"Fetched {cur.date()} → {end.date()}")
                cur = end + timedelta(days=1)
                time.sleep(0.3)

            prog.empty()
            if not all_dfs:
                st.warning("⚠️ No data returned. Try Daily timeframe or shorter date range.")
                return None
            return (pd.concat(all_dfs)
                      .drop_duplicates("timestamp")
                      .sort_values("timestamp")
                      .reset_index(drop=True))

        else:
            payload = {
                "securityId":      security_id,
                "exchangeSegment": "IDX_I",
                "instrument":      "INDEX",
                "fromDate":        from_date,
                "toDate":          to_date,
                "expiryCode":      0,
            }
            if debug:
                st.write("📤 Daily payload:", payload)
            try:
                r = requests.post(f"{self.base_url}/charts/historical",
                                  headers=self.headers, json=payload, timeout=30)
                if debug:
                    st.write(f"📥 Response {r.status_code}:", r.text[:600])
                if not r.ok:
                    self._handle_error(r)
                    return None
                data = r.json()
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                return _parse_ohlcv(data)
            except Exception as e:
                st.error(f"Request error: {e}")
            return None

    def fetch_rolling_option(self, security_id: int, strike_offset: int,
                              opt_type: str, from_date: str, to_date: str,
                              interval: str = "25", expiry_flag: str = "WEEK",
                              debug: bool = False) -> pd.DataFrame | None:
        """
        Fetch real option OHLCV via /charts/rollingoption — chunked in 89-day windows
        to avoid DH-905 (max 90 days per call).
        opt_type: "CALL" or "PUT"
        """
        strike_str = "ATM" if strike_offset == 0 else (
            f"ATM+{strike_offset}" if strike_offset > 0 else f"ATM{strike_offset}"
        )
        key = "ce" if opt_type == "CALL" else "pe"

        from_dt    = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt      = datetime.strptime(to_date,   "%Y-%m-%d")
        total_days = max((to_dt - from_dt).days, 1)
        all_dfs, fetched = [], 0
        prog = st.progress(0, text=f"Fetching {strike_str} {opt_type} option data…")
        cur  = from_dt

        while cur <= to_dt:
            end = min(cur + timedelta(days=89), to_dt)   # max 89 days per call (DH-905)
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval":        interval,
                "securityId":      security_id,
                "instrument":      "OPTIDX",
                "expiryFlag":      expiry_flag,
                "expiryCode":      1,
                "strike":          strike_str,
                "drvOptionType":   opt_type,
                "requiredData":    ["open", "high", "low", "close", "volume", "oi"],
                "fromDate":        cur.strftime("%Y-%m-%d"),
                "toDate":          end.strftime("%Y-%m-%d"),
            }
            if debug:
                st.write(f"📤 Rolling option payload ({cur.date()} → {end.date()}):", payload)
            try:
                r = requests.post(f"{self.base_url}/charts/rollingoption",
                                  headers=self.headers, json=payload, timeout=30)
                if debug:
                    st.write(f"📥 Response {r.status_code}:", r.text[:600])
                if not r.ok:
                    self._handle_error(r)
                    prog.empty()
                    return None
                raw = r.json().get("data", {}).get(key, {})
                if raw and raw.get("timestamp"):
                    chunk = pd.DataFrame({
                        "timestamp": pd.to_datetime(raw["timestamp"], unit="s", utc=True)
                                       .tz_convert("Asia/Kolkata").tz_localize(None),
                        "open":   pd.to_numeric(raw.get("open",   []), errors="coerce"),
                        "high":   pd.to_numeric(raw.get("high",   []), errors="coerce"),
                        "low":    pd.to_numeric(raw.get("low",    []), errors="coerce"),
                        "close":  pd.to_numeric(raw.get("close",  []), errors="coerce"),
                        "volume": pd.to_numeric(raw.get("volume", []), errors="coerce"),
                        "oi":     pd.to_numeric(raw.get("oi", [np.nan]*len(raw["timestamp"])), errors="coerce"),
                    }).dropna(subset=["close"])
                    if not chunk.empty:
                        all_dfs.append(chunk)
            except Exception as e:
                st.error(f"Rolling option chunk error: {e}")
                prog.empty()
                return None

            fetched += max((end - cur).days, 1)
            prog.progress(min(fetched / total_days, 1.0),
                          text=f"Option data: {cur.date()} → {end.date()}")
            cur = end + timedelta(days=1)
            time.sleep(0.3)

        prog.empty()
        if not all_dfs:
            st.warning("⚠️ No option data returned for this strike/range.")
            return None
        return (pd.concat(all_dfs)
                  .drop_duplicates("timestamp")
                  .sort_values("timestamp")
                  .reset_index(drop=True))


# Module-level fetcher instance — uses hardcoded DhanConfig
_fetcher = DhanFetcher(DhanConfig())


def _parse_ohlcv(data: dict) -> pd.DataFrame | None:
    if not data or "open" not in data or not data.get("timestamp"):
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


# ══════════════════════════════════════════════════════════════════════════════
# STRIKE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_atm_strike(spot: float, strike_gap: int) -> int:
    """Round spot to nearest strike_gap."""
    return int(round(spot / strike_gap) * strike_gap)


def get_strike_label(offset: int) -> str:
    if offset == 0:   return "ATM"
    if offset > 0:    return f"ATM+{offset}"
    return f"ATM{offset}"


def compute_strike_price(spot: float, strike_gap: int, offset: int) -> int:
    """ATM + offset * strike_gap."""
    atm = get_atm_strike(spot, strike_gap)
    return atm + offset * strike_gap


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def calc_bsp(df: pd.DataFrame, length: int) -> pd.Series:
    """
    BSP (Balance of Selling Pressure) — exact Pine Script translation.
    Pine: ad = (close==high and close==low) or high==low ? 0 : ((2*close-low-high)/(high-low))*volume
    mf  = sum(ad, length) / sum(volume, length)
    """
    hl = df["high"] - df["low"]
    flat = (hl == 0) | ((df["close"] == df["high"]) & (df["close"] == df["low"]))
    ad   = np.where(flat, 0.0, ((2 * df["close"] - df["low"] - df["high"]) / hl.replace(0, 1.0)) * df["volume"])
    ad   = pd.Series(ad, index=df.index)
    vol_sum = df["volume"].rolling(length).sum()
    return (ad.rolling(length).sum() / vol_sum).fillna(0)


def calc_bsp_daily(df_intraday: pd.DataFrame, daily_df: pd.DataFrame, length: int) -> pd.Series:
    """
    Pine uses request.security(syminfo.tickerid, 'D', mf) — BSP is computed on
    DAILY bars and then stamp-forwarded onto each intraday bar (same value all day).
    This function replicates that: compute BSP on daily_df, then merge onto intraday.
    """
    if daily_df is None or daily_df.empty:
        return calc_bsp(df_intraday, length)  # fallback to intraday BSP
    # Compute on daily bars
    hl = daily_df["high"] - daily_df["low"]
    flat = (hl == 0) | ((daily_df["close"] == daily_df["high"]) & (daily_df["close"] == daily_df["low"]))
    ad   = np.where(flat, 0.0,
                    ((2 * daily_df["close"] - daily_df["low"] - daily_df["high"])
                     / hl.replace(0, 1.0)) * daily_df["volume"])
    ad   = pd.Series(ad, index=daily_df.index)
    vol_sum = daily_df["volume"].rolling(length).sum()
    daily_bsp = (ad.rolling(length).sum() / vol_sum).fillna(0)
    daily_bsp.index = pd.to_datetime(daily_df["timestamp"].values).normalize()
    # Stamp onto intraday bars by date
    intra_dates = pd.to_datetime(df_intraday["timestamp"].values).normalize()
    return pd.Series(intra_dates, index=df_intraday.index).map(
        daily_bsp.to_dict()
    ).ffill().fillna(0)


def pivot_highs(df: pd.DataFrame, n: int) -> list[dict]:
    out = []
    for i in range(n, len(df) - n):
        win = df["high"].iloc[i - n: i + n + 1]
        if df["high"].iloc[i] == win.max():
            out.append({"idx": i, "price": df["high"].iloc[i],
                        "open": df["open"].iloc[i], "close": df["close"].iloc[i],
                        "volume": df["volume"].iloc[i]})
    return out


def pivot_lows(df: pd.DataFrame, n: int) -> list[dict]:
    out = []
    for i in range(n, len(df) - n):
        win = df["low"].iloc[i - n: i + n + 1]
        if df["low"].iloc[i] == win.min():
            out.append({"idx": i, "price": df["low"].iloc[i],
                        "open": df["open"].iloc[i], "close": df["close"].iloc[i],
                        "volume": df["volume"].iloc[i]})
    return out


def order_blocks(df: pd.DataFrame, ph: list, pl: list,
                 vt: float = 1.5) -> list[dict]:
    vsma = df["volume"].rolling(14).mean()
    obs  = []
    for p in ph:
        i      = p["idx"]
        strong = p["volume"] > vsma.iloc[i] * vt if pd.notna(vsma.iloc[i]) else False
        end    = next((j for j in range(i + 1, len(df))
                       if df["close"].iloc[j] > p["price"]), len(df) - 1)
        obs.append({"type": "bearish", "top": p["price"],
                    "btm": max(p["open"], p["close"]),
                    "start": i, "end": end, "strong": strong})
    for p in pl:
        i      = p["idx"]
        strong = p["volume"] > vsma.iloc[i] * vt if pd.notna(vsma.iloc[i]) else False
        end    = next((j for j in range(i + 1, len(df))
                       if df["close"].iloc[j] < p["price"]), len(df) - 1)
        obs.append({"type": "bullish", "top": min(p["open"], p["close"]),
                    "btm": p["price"],
                    "start": i, "end": end, "strong": strong})
    return obs


def fair_value_gaps(df: pd.DataFrame) -> list[dict]:
    fvgs = []
    for i in range(2, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            mid = (df["low"].iloc[i] + df["high"].iloc[i - 2]) / 2
            if (df["low"].iloc[i] - df["high"].iloc[i - 2]) / mid > 0.001:
                fvgs.append({"type": "bullish", "top": df["low"].iloc[i],
                              "btm": df["high"].iloc[i - 2], "start": i - 2})
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            mid = (df["low"].iloc[i - 2] + df["high"].iloc[i]) / 2
            if (df["low"].iloc[i - 2] - df["high"].iloc[i]) / mid > 0.001:
                fvgs.append({"type": "bearish", "top": df["low"].iloc[i - 2],
                              "btm": df["high"].iloc[i], "start": i - 2})
    return fvgs


# ══════════════════════════════════════════════════════════════════════════════
# OPTION BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Normal CDF using math.erf — no scipy needed."""
    import math
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes_price(S, K, T, r, sigma, option_type="CE"):
    """Black-Scholes using pure math — no scipy."""
    import math
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0) if option_type in ("CE", "CALL") else max(K - S, 0)
        return max(intrinsic, 0.05)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type in ("CE", "CALL"):
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    except Exception:
        return 0.05


def simulate_option_prices(df: pd.DataFrame, strike_offset: int,
                            strike_gap: int, opt_type: str,
                            dte: int = 7, iv: float = 0.15) -> pd.DataFrame:
    """
    Simulate option prices for each bar using Black-Scholes.
    strike_offset: integer offset from ATM (e.g. 0=ATM, 1=ATM+1, -1=ATM-1)
    """
    df = df.copy()
    r  = 0.065  # RBI repo rate approx
    prices = []

    for i, row in df.iterrows():
        S = row["close"]
        remaining_dte = max(dte - i * (dte / len(df)), 0.01)
        T = remaining_dte / 365
        K = compute_strike_price(S, strike_gap, strike_offset)
        price = black_scholes_price(S, K, T, r, iv, opt_type)
        prices.append({"strike": K, "opt_price": round(price, 2)})

    opt_df = pd.DataFrame(prices, index=df.index)
    df["strike"]    = opt_df["strike"]
    df["opt_price"] = opt_df["opt_price"]
    return df


def generate_signals(df: pd.DataFrame, buy_lvl: float, sell_lvl: float,
                      ema_filter: bool = True, signal_mode: str = "Pine Exact") -> pd.DataFrame:
    """
    Signal modes:

    Pine Exact  ← DEFAULT — exact replica of TradingView Pine Script logic:
                  ENTRY: bsp > buyLevel AND close > ema20 AND ema20 > ema50
                  EXIT:  bsp < sellLevel AND close < ema20 AND ema20 < ema50
                  (strategy.entry BUY + strategy.close BUY — no shorts taken)

    Level Hold  — entry while BSP sustained above level; exit at neutral BSP
                  (more trades on long historical data)

    Flip        — entry only on BSP level CROSS; exit only on cross below sell level

    BSP Only    — no EMA filter, pure BSP threshold (most trades)
    """
    df = df.copy()

    # EMA trend conditions — shared by all modes
    has_ema = "ema20" in df.columns and "ema50" in df.columns
    if has_ema and ema_filter:
        bull_trend = (df["close"] > df["ema20"]) & (df["ema20"] > df["ema50"])
        bear_trend = (df["close"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    else:
        bull_trend = pd.Series(True, index=df.index)
        bear_trend = pd.Series(True, index=df.index)

    bsp = df["bsp"]

    if signal_mode == "Pine Exact":
        # ── EXACT Pine Script replication ────────────────────────────────────
        # longCondition  = bsp > bspBuyLevel  AND close > ema20 AND ema20 > ema50
        # shortCondition = bsp < bspSellLevel AND close < ema20 AND ema20 < ema50
        # strategy.entry("BUY", strategy.long) when longCondition
        # strategy.close("BUY")               when shortCondition
        long_entry  = (bsp > buy_lvl)  & bull_trend
        exit_signal = (bsp < sell_lvl) & bear_trend
        df["signal"] = np.where(long_entry, 1, np.where(exit_signal, -1, 0))
        return df

    elif signal_mode == "Level Hold":
        long_entry  = (bsp > buy_lvl)  & bull_trend
        short_entry = (bsp < sell_lvl) & bear_trend
        long_exit   = bsp < 0
        short_exit  = bsp > 0
        sigs = [0] * len(df)
        for i in range(1, len(df)):
            prev = sigs[i - 1]
            if long_entry.iloc[i]:
                sigs[i] = 1
            elif short_entry.iloc[i]:
                sigs[i] = -1
            elif prev == 1 and (long_exit.iloc[i] or short_entry.iloc[i]):
                sigs[i] = -1
            elif prev == -1 and (short_exit.iloc[i] or long_entry.iloc[i]):
                sigs[i] = 1
            else:
                sigs[i] = 0
        df["signal"] = sigs
        return df

    elif signal_mode == "BSP Only":
        long_entry  = bsp > buy_lvl
        short_entry = bsp < sell_lvl

    else:  # Flip — BSP level cross
        long_entry  = (bsp > buy_lvl)  & (bsp.shift(1) <= buy_lvl)  & bull_trend
        short_entry = (bsp < sell_lvl) & (bsp.shift(1) >= sell_lvl) & bear_trend

    df["signal"] = np.where(long_entry, 1, np.where(short_entry, -1, 0))
    return df


def run_backtest(df: pd.DataFrame, capital: float, size_pct: float,
                 comm_pct: float, lot_size: int,
                 trade_options: bool = False,
                 fixed_lots: int = None,
                 spread_legs: list = None,
                 is_intraday: bool = False,
                 eod_exit_time=None) -> tuple[pd.DataFrame, list]:
    """
    Spread-aware lot-based backtest.

    Single leg:  price_col = opt_price (or close for index mode)
    Multi-leg:   spread_price column = sum of signed leg prices already on df.
                 Per-leg P&L breakdown recorded in each trade dict.

    Sizing:
      fixed_lots → that many SCALE lots (each leg uses its own leg["lots"] ratio)
      % of Capital → lots = floor(budget / |spread_cost_per_lot|)

    is_intraday: force-close all positions at eod_exit_time each trading day
    eod_exit_time: datetime.time object for EOD square-off (default 15:15)
    """
    cash, pos_lots, entry_spread, entry_time = capital, 0, 0.0, None
    in_trade = False
    equities, trades = [], []

    # Determine price column
    if trade_options and spread_legs and "spread_price" in df.columns:
        price_col = "spread_price"
        is_spread  = True
    elif trade_options and "opt_price" in df.columns:
        price_col = "opt_price"
        is_spread  = False
    else:
        price_col = "close"
        is_spread  = False

    # Build per-leg price lookups for spread mode
    leg_price_series = {}
    if is_spread and spread_legs:
        ts_index = df["timestamp"].values
        for i, ld in enumerate(spread_legs):
            leg_price_series[i] = ld["prices"].reindex(
                pd.Index(ts_index)
            ).ffill().fillna(0).values

    df_ts = df["timestamp"].values  # fast numpy access

    import datetime as _dt
    # Resolve EOD exit time
    if is_intraday and eod_exit_time is not None:
        _eod_h, _eod_m = eod_exit_time.hour, eod_exit_time.minute
    else:
        _eod_h, _eod_m = 15, 15   # default fallback

    for row_i, row in df.iterrows():
        sig = row["signal"]
        ts  = row["timestamp"]
        raw_price = row[price_col]

        if pd.isna(raw_price):
            equities.append(cash + pos_lots * (entry_spread if in_trade else 0))
            continue

        # Spread price = net debit/credit per scale-lot
        spread_price = float(raw_price)

        # ── INTRADAY EOD FORCE-CLOSE ──────────────────────────────────────
        # If Intraday mode: square off at/after eod_exit_time on each day
        if is_intraday and in_trade:
            bar_time = ts.time() if hasattr(ts, "time") else (
                _dt.datetime.fromtimestamp(ts.timestamp()).time()
                if hasattr(ts, "timestamp") else None
            )
            if bar_time and bar_time >= _dt.time(_eod_h, _eod_m):
                eod_spread = spread_price
                qty        = pos_lots * lot_size
                pnl        = (eod_spread - entry_spread) * qty - abs(eod_spread) * qty * comm_pct
                cash      += abs(entry_spread) * qty + pnl
                trade_rec  = {
                    "entry_time":  entry_time, "exit_time": ts,
                    "entry_price": round(entry_spread, 2), "exit_price": round(eod_spread, 2),
                    "qty": qty, "lots": pos_lots,
                    "pnl": round(pnl, 2),
                    "return_pct": round((eod_spread / entry_spread - 1) * 100, 2) if entry_spread != 0 else 0.0,
                    "exit_reason": "EOD Square-off",
                    "strike": row.get("strike", "-"),
                }
                if is_spread and spread_legs:
                    for i, ld in enumerate(spread_legs):
                        lp_arr = leg_price_series.get(i)
                        if lp_arr is not None and row_i < len(lp_arr):
                            try:
                                eidx = list(df_ts).index(entry_time)
                                ep = float(lp_arr[eidx]); xp = float(lp_arr[row_i])
                            except (ValueError, IndexError):
                                ep = xp = 0.0
                            sign = 1 if ld["direction"] == "BUY" else -1
                            trade_rec[f"leg{i+1}_{ld['strike_lbl']}_{ld['opt_type']}"] = round(
                                sign * (xp - ep) * ld["lots"] * lot_size, 2)
                trades.append(trade_rec)
                pos_lots, in_trade = 0, False
                equities.append(cash)
                continue  # skip further processing for this bar

        # ── ENTRY — trigger on BUY signal (or SELL signal for short legs) ──
        # Entry fires on sig==1 (long) or sig==-1 when no position open
        # In Intraday mode: block new entries after EOD time
        _bar_time_now = ts.time() if hasattr(ts, "time") else None
        _block_entry  = is_intraday and _bar_time_now and _bar_time_now >= _dt.time(_eod_h, _eod_m)
        entry_triggered = (sig == 1 and not in_trade and not _block_entry)

        if entry_triggered:
            # Cost basis: |spread_price| × lot_size per scale-lot
            cost_per_lot = abs(spread_price) * lot_size
            if cost_per_lot < 0.01:
                equities.append(cash)
                continue

            if fixed_lots is not None:
                scale_lots = int(fixed_lots)
            else:
                scale_lots = max(int(cash * size_pct / cost_per_lot), 1)

            total_cost = cost_per_lot * scale_lots * (1 + comm_pct)
            if total_cost <= cash:
                pos_lots     = scale_lots
                cash        -= total_cost
                entry_spread = spread_price
                entry_time   = ts
                in_trade     = True

        # ── EXIT — on sell signal OR new buy when already in trade ─────────
        elif in_trade and sig == -1:
            exit_spread = spread_price
            qty         = pos_lots * lot_size  # total underlying units

            gross_pnl = (exit_spread - entry_spread) * qty
            comm_cost = abs(exit_spread) * qty * comm_pct
            pnl       = gross_pnl - comm_cost
            cash     += abs(entry_spread) * qty + pnl  # return capital + P&L

            trade_rec = {
                "entry_time":    entry_time,
                "exit_time":     ts,
                "entry_price":   round(entry_spread, 2),
                "exit_price":    round(exit_spread,  2),
                "qty":           qty,
                "lots":          pos_lots,
                "pnl":           round(pnl, 2),
                "return_pct":    round((exit_spread / entry_spread - 1) * 100, 2)
                                 if entry_spread != 0 else 0.0,
                "exit_reason":   "Signal",
                "strike":        row.get("strike", "-"),
            }

            # Per-leg P&L breakdown for spread trades
            if is_spread and spread_legs:
                for i, ld in enumerate(spread_legs):
                    lp_arr  = leg_price_series.get(i)
                    if lp_arr is not None and row_i < len(lp_arr):
                        # Find entry row index (approx by timestamp match)
                        try:
                            entry_idx = list(df_ts).index(entry_time)
                            ep = float(lp_arr[entry_idx])
                            xp = float(lp_arr[row_i])
                        except (ValueError, IndexError):
                            ep = xp = 0.0
                        sign      = 1 if ld["direction"] == "BUY" else -1
                        leg_pnl   = sign * (xp - ep) * ld["lots"] * lot_size
                        trade_rec[f"leg{i+1}_{ld['strike_lbl']}_{ld['opt_type']}"] = round(leg_pnl, 2)

            trades.append(trade_rec)
            pos_lots, in_trade = 0, False

        # Mark-to-market equity
        mtm = spread_price * pos_lots * lot_size if in_trade else 0
        equities.append(cash + mtm)

    # Force-close open position at last bar
    if in_trade and pos_lots > 0:
        lp = float(df[price_col].iloc[-1])
        qty = pos_lots * lot_size
        pnl = (lp - entry_spread) * qty - abs(lp) * qty * comm_pct
        cash += abs(entry_spread) * qty + pnl
        trade_rec = {
            "entry_time":  entry_time, "exit_time": df["timestamp"].iloc[-1],
            "entry_price": round(entry_spread, 2), "exit_price": round(lp, 2),
            "qty": qty, "lots": pos_lots,
            "pnl": round(pnl, 2),
            "return_pct": round((lp / entry_spread - 1) * 100, 2) if entry_spread != 0 else 0.0,
            "exit_reason": "End of Data",
            "strike": df["strike"].iloc[-1] if "strike" in df.columns else "-",
        }
        if is_spread and spread_legs:
            for i, ld in enumerate(spread_legs):
                lp_arr = leg_price_series.get(i)
                if lp_arr is not None:
                    try:
                        entry_idx = list(df_ts).index(entry_time)
                        ep = float(lp_arr[entry_idx])
                        xp = float(lp_arr[-1])
                    except (ValueError, IndexError):
                        ep = xp = 0.0
                    sign    = 1 if ld["direction"] == "BUY" else -1
                    leg_pnl = sign * (xp - ep) * ld["lots"] * lot_size
                    trade_rec[f"leg{i+1}_{ld['strike_lbl']}_{ld['opt_type']}"] = round(leg_pnl, 2)
        trades.append(trade_rec)
        if equities:
            equities[-1] = cash

    while len(equities) < len(df):
        equities.append(equities[-1] if equities else capital)

    results = df[["timestamp"]].copy()
    results["equity"]       = equities[:len(df)]
    results["peak"]         = results["equity"].cummax()
    results["drawdown_pct"] = (results["equity"] - results["peak"]) / results["peak"] * 100
    return results, trades


def metrics(results: pd.DataFrame, trades: list, capital: float) -> dict:
    import math

    final = results["equity"].iloc[-1]
    pnl   = final - capital
    ret   = round(pnl / capital * 100, 2)

    # CAGR — clamp to ±10000% to avoid astronomical display
    n_days = max((results["timestamp"].iloc[-1] - results["timestamp"].iloc[0]).days, 1)
    ratio  = max(final / capital, 1e-10) if capital > 0 else 1.0
    try:
        cagr_raw = (ratio ** (365.0 / n_days) - 1) * 100
        cagr = round(max(min(cagr_raw, 10_000.0), -100.0), 2)
    except (OverflowError, ZeroDivisionError):
        cagr = 10_000.0 if ret > 0 else -100.0

    max_dd = round(results["drawdown_pct"].min(), 2)

    # Sharpe/Sortino on DAILY returns (not bar-by-bar — avoids intraday inflation)
    try:
        daily  = results.set_index("timestamp")["equity"].resample("D").last().dropna()
        d_rets = daily.pct_change().dropna()
        sharpe  = round(float(d_rets.mean() / d_rets.std() * math.sqrt(252)), 3) if len(d_rets) > 1 and d_rets.std() > 0 else 0.0
        neg     = d_rets[d_rets < 0]
        sortino = round(float(d_rets.mean() / neg.std() * math.sqrt(252)), 3) if len(neg) > 1 and neg.std() > 0 else 0.0
        sharpe  = max(min(sharpe,  999.0), -999.0)
        sortino = max(min(sortino, 999.0), -999.0)
    except Exception:
        sharpe = sortino = 0.0

    calmar = round(cagr / abs(max_dd), 2) if max_dd != 0 else 0.0
    calmar = max(min(calmar, 999.0), -999.0)

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins), abs(sum(losses))

    return {
        "total_return":  ret,
        "total_pnl":     round(pnl, 2),
        "cagr":          cagr,
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "total_trades":  len(trades),
        "win_rate":      round(len(wins)/len(trades)*100, 1) if trades else 0.0,
        "profit_factor": round(gp/gl, 2) if gl > 0 else (round(gp, 2) if gp > 0 else 0.0),
        "avg_win":       round(float(np.mean(wins)),   2) if wins   else 0.0,
        "avg_loss":      round(float(np.mean(losses)), 2) if losses else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Token Status ─────────────────────────────────────────────────────────
    cfg = DhanConfig()
    if cfg.access_token and cfg.access_token != "paste_your_token_here":
        st.success(f"✅ Token configured · Client: `{cfg.client_id}`")
    else:
        st.error("❌ Token not set — update `access_token` in DhanConfig at top of app.py")

    # ── Index Selection ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">📊 Index Settings</div>', unsafe_allow_html=True)

    index_name = st.selectbox("Select Index", list(INDICES.keys()), index=0)
    idx_cfg    = INDICES[index_name]
    lot_size   = idx_cfg["lot_size"]
    strike_gap = idx_cfg["strike_gap"]

    st.caption(f"Lot Size: **{lot_size}** | Strike Gap: **{strike_gap}**")

    # ── Trade Style ────────────────────────────────────────────────────────
    trade_style = st.radio(
        "Trade Style",
        ["📈 Intraday", "🌙 Carry Forward"],
        horizontal=True,
        help="Intraday: positions force-squared off at EOD each day.\n"
             "Carry Forward: positions held overnight until exit signal fires.",
    )
    is_intraday = trade_style == "📈 Intraday"

    if is_intraday:
        INTRADAY_TF = {
            "1 Min": "1", "3 Min": "3", "5 Min": "5",
            "10 Min": "10", "15 Min": "15", "25 Min": "25",
        }
        interval_lbl = st.selectbox("Timeframe", list(INTRADAY_TF.keys()), index=2,
                                     help="Intraday timeframes only")
        interval     = INTRADAY_TF[interval_lbl]
        c_eod1, c_eod2 = st.columns(2)
        with c_eod1:
            eod_hour   = st.number_input("EOD Exit Hour",   min_value=9,  max_value=15, value=15, step=1)
        with c_eod2:
            eod_minute = st.number_input("EOD Exit Minute", min_value=0,  max_value=59, value=15, step=5)
        import datetime as _dt2
        eod_exit_time = _dt2.time(int(eod_hour), int(eod_minute))
    else:
        CF_TF = {
            "5 Min": "5", "15 Min": "15", "25 Min": "25",
            "60 Min": "60", "Daily": "D",
        }
        interval_lbl = st.selectbox("Timeframe", list(CF_TF.keys()), index=4,
                                     help="Carry Forward — positions held overnight")
        interval     = CF_TF[interval_lbl]
        eod_exit_time = None

    bsp_tf = st.selectbox(
        "BSP Timeframe",
        ["Daily (Pine Default ✅)", "Same as Chart"],
        index=0,
        help=(
            "Daily (Pine Default): matches TradingView Pine Script exactly. "
            "Pine uses request.security(..., 'D', mf) — BSP is ALWAYS computed on Daily bars "
            "regardless of chart timeframe.\n\n"
            "Same as Chart: BSP computed on your selected candle interval."
        )
    )
    use_daily_bsp = "Daily" in bsp_tf

    c1, c2 = st.columns(2)
    with c1:
        default_from = (datetime.now() - timedelta(days=30)) if is_intraday else (datetime.now() - timedelta(days=365))
        from_date = st.date_input("From", default_from)
    with c2:
        to_date = st.date_input("To", datetime.now())

    if is_intraday and (to_date - from_date).days > 90:
        st.warning("⚠️ Intraday data: Dhan API allows max ~90 days per call. Consider shorter ranges for speed.")

    # ── Backtest Mode ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">🎯 Backtest Mode</div>', unsafe_allow_html=True)

    backtest_mode = st.radio(
        "Trade on",
        ["Index (Futures-style)", "Options (Real Data)"],
        index=0,
        help="Index mode: signals on index OHLCV. Options mode: uses Dhan rollingoption API for real premium data."
    )
    trade_options = backtest_mode == "Options (Real Data)"

    if trade_options:
        st.markdown('<div class="sidebar-header">📌 Option Legs (Spread / Hedge)</div>', unsafe_allow_html=True)

        expiry_flag = st.radio("Expiry Type", ["WEEK", "MONTH"], horizontal=True,
                               help="Weekly or Monthly expiry contract")

        n_legs = st.selectbox("Number of Legs", [1, 2, 3, 4], index=0,
                              help="1 = Single option | 2 = Spread | 3-4 = Complex (Iron Condor etc.)")

        # Preset templates
        PRESETS = {
            "Custom": None,
            "Bull Call Spread":  [("CALL","ATM",0,"BUY"),  ("CALL","ATM+1",1,"SELL")],
            "Bear Put Spread":   [("PUT", "ATM",0,"BUY"),  ("PUT", "ATM-1",-1,"SELL")],
            "Bull Put Spread":   [("PUT", "ATM-1",-1,"SELL"),("PUT","ATM-2",-2,"BUY")],
            "Bear Call Spread":  [("CALL","ATM+1",1,"SELL"),("CALL","ATM+2",2,"BUY")],
            "Long Strangle":     [("CALL","ATM+1",1,"BUY"), ("PUT","ATM-1",-1,"BUY")],
            "Short Strangle":    [("CALL","ATM+1",1,"SELL"),("PUT","ATM-1",-1,"SELL")],
            "Long Straddle":     [("CALL","ATM",0,"BUY"),  ("PUT","ATM",0,"BUY")],
            "Iron Condor":       [("PUT","ATM-2",-2,"BUY"),("PUT","ATM-1",-1,"SELL"),
                                  ("CALL","ATM+1",1,"SELL"),("CALL","ATM+2",2,"BUY")],
        }

        preset = st.selectbox("📋 Strategy Template", list(PRESETS.keys()), index=0)

        offsets_list  = list(range(-10, 11))
        offset_labels = [get_strike_label(o) for o in offsets_list]

        option_legs = []
        for i in range(n_legs):
            with st.expander(f"Leg {i+1}", expanded=True):
                # Auto-fill from preset
                if preset != "Custom" and PRESETS[preset] and i < len(PRESETS[preset]):
                    p_type, p_label, p_off, p_dir = PRESETS[preset][i]
                    def_type = 0 if p_type == "CALL" else 1
                    def_off  = p_label
                    def_dir  = 0 if p_dir == "BUY" else 1
                else:
                    def_type = 0; def_off = "ATM"; def_dir = 0

                c1, c2 = st.columns(2)
                with c1:
                    leg_type = st.radio(f"Type##leg{i}", ["CE (Call)", "PE (Put)"],
                                        index=def_type, horizontal=True, key=f"leg_type_{i}")
                with c2:
                    leg_dir  = st.radio(f"Direction##leg{i}", ["BUY", "SELL"],
                                        index=def_dir, horizontal=True, key=f"leg_dir_{i}")

                leg_strike_lbl = st.select_slider(
                    f"Strike##leg{i}",
                    options=offset_labels,
                    value=def_off if def_off in offset_labels else "ATM",
                    key=f"leg_strike_{i}"
                )
                leg_offset = offsets_list[offset_labels.index(leg_strike_lbl)]
                leg_lots   = st.number_input(f"Lots##leg{i}", min_value=1, max_value=50,
                                              value=1, key=f"leg_lots_{i}")

                option_legs.append({
                    "opt_type":    "CALL" if "CE" in leg_type else "PUT",
                    "direction":   leg_dir,          # "BUY" or "SELL"
                    "offset":      leg_offset,
                    "strike_lbl":  leg_strike_lbl,
                    "lots":        int(leg_lots),
                })

        # Backward compat single-leg variables
        opt_type      = option_legs[0]["opt_type"]
        strike_offset = option_legs[0]["offset"]

        # Strategy summary
        leg_summaries = " | ".join(
            f"{lg['direction']} {lg['lots']}L {lg['strike_lbl']} {lg['opt_type']}"
            for lg in option_legs
        )
        st.info(f"**Strategy:** {leg_summaries}")

    # ── Strategy Parameters ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">⚙️ Strategy Parameters</div>', unsafe_allow_html=True)

    pivot_length = st.slider("Pivot Lookback",  3, 20, 5)
    bsp_length   = st.slider("BSP Length",      5, 50, 21)
    bsp_buy_lvl  = st.number_input("BSP Buy Level",  value=0.08,  step=0.01, format="%.2f")
    bsp_sell_lvl = st.number_input("BSP Sell Level", value=-0.08, step=0.01, format="%.2f")
    vol_threshold = st.slider("Volume Threshold",   1.0, 3.0, 1.5, 0.1)
    vol_period    = st.slider("Volume SMA Period",   5,  30,  14)

    st.markdown('<div class="sidebar-header">🎚️ Signal Mode</div>', unsafe_allow_html=True)
    signal_mode = st.selectbox(
        "Signal Mode",
        ["Flip", "Level Hold", "BSP Only"],
        index=1,
        help=(
            "Flip: trade only on BSP level CROSS — fewest signals, can miss trades on long history\n"
            "Level Hold: enter while BSP above threshold, exit when BSP returns to 0 — balanced\n"
            "BSP Only: no EMA filter, pure BSP — most signals, good for long backtests"
        )
    )
    ema_filter = st.checkbox(
        "EMA Trend Filter",
        value=(signal_mode != "BSP Only"),
        help="Require EMA20 > EMA50 for longs (and vice-versa). Uncheck for more signals on longer histories.",
        disabled=(signal_mode == "BSP Only")
    )

    st.markdown('<div class="sidebar-header">📈 EMA Display</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        show_e20  = st.checkbox("EMA 20",  True)
        show_e50  = st.checkbox("EMA 50",  True)
    with c2:
        show_e100 = st.checkbox("EMA 100", True)
        show_e200 = st.checkbox("EMA 200", True)

    st.markdown('<div class="sidebar-header">💼 Capital & Sizing</div>', unsafe_allow_html=True)
    init_capital  = st.number_input("Initial Capital (₹)", value=500000, step=50000)
    sizing_mode   = st.radio("Position Sizing", ["% of Capital", "Fixed Lots"], horizontal=True)
    if sizing_mode == "% of Capital":
        pos_size_pct = st.slider("Position Size (%)", 5, 100, 50)
        fixed_lots   = None
    else:
        fixed_lots   = st.number_input("Fixed Lots per Trade", min_value=1, max_value=100, value=1, step=1)
        pos_size_pct = 100   # unused in fixed mode, placeholder
    comm_pct = st.number_input("Commission (%)", value=0.03, step=0.01, format="%.3f")

    st.markdown('<div class="sidebar-header">🔄 Trade Style</div>', unsafe_allow_html=True)
    trade_style = st.radio(
        "Position Carrying",
        ["📅 Intraday (MIS)", "📆 Carry Forward (NRML)"],
        index=0,
        help=(
            "Intraday (MIS): All open positions are force-closed at 15:15 IST each day.\n"
            "Carry Forward (NRML): Positions can be held overnight across multiple days."
        )
    )
    is_intraday = (trade_style == "📅 Intraday (MIS)")

    if is_intraday:
        eod_exit_time = st.time_input(
            "EOD Square-off Time",
            value=__import__("datetime").time(15, 15),
            help="Force-close all positions at this time each day (default 15:15 IST)"
        )
    else:
        eod_exit_time = None

    st.markdown('<div class="sidebar-header">🛠️ Debug</div>', unsafe_allow_html=True)
    debug_mode = st.checkbox("Show API request/response", False,
                              help="Shows raw payloads and responses to diagnose API errors")

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"## 📡 NYZTrade · SMC Liquidity Lens — Index Backtest")
st.caption("Smart Money Concepts + BSP Oscillator + FVG | Indian Indices | Dhan API")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    # ── 1. Fetch Index OHLCV ─────────────────────────────────────────────────
    with st.spinner(f"📡 Fetching {index_name} OHLCV from Dhan…"):
        df = _fetcher.fetch_index_ohlcv(
            idx_cfg["security_id"],
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
            interval,
            debug=debug_mode
        )

    if df is None or df.empty:
        st.error("❌ No index data returned. Try: shorter date range, Daily timeframe, or refresh your token.")
        st.stop()

    st.success(f"✅ Loaded **{len(df):,} bars** for **{index_name}** ({interval_lbl})")

    # ── 2. Indicators & Signals ───────────────────────────────────────────────
    with st.spinner("⚙️ Computing indicators…"):
        df["ema20"]   = ema(df["close"], 20)
        df["ema50"]   = ema(df["close"], 50)
        df["ema100"]  = ema(df["close"], 100)
        df["ema200"]  = ema(df["close"], 200)
        df["vol_sma"] = df["volume"].rolling(vol_period).mean()

        # BSP Timeframe — Pine uses request.security(..., 'D', mf) by default
        if use_daily_bsp and interval != "D":
            with st.spinner("📅 Fetching Daily bars for BSP calculation (Pine-accurate)…"):
                daily_df = _fetcher.fetch_index_ohlcv(
                    idx_cfg["security_id"],
                    from_date.strftime("%Y-%m-%d"),
                    to_date.strftime("%Y-%m-%d"),
                    interval="D",
                    debug=False
                )
            df["bsp"] = calc_bsp_daily(df, daily_df, bsp_length)
            st.caption("📅 BSP computed on **Daily** timeframe (matches TradingView Pine Script)")
        else:
            df["bsp"] = calc_bsp(df, bsp_length)

        ph   = pivot_highs(df, pivot_length)
        pl   = pivot_lows(df, pivot_length)
        obs  = order_blocks(df, ph, pl, vol_threshold)
        fvgs = fair_value_gaps(df)
        df   = generate_signals(df, bsp_buy_lvl, bsp_sell_lvl, ema_filter=ema_filter, signal_mode=signal_mode)

    # ── 2b. Signal Diagnostics ───────────────────────────────────────────────
    total_bars   = len(df)
    n_buy_sigs   = int((df["signal"] == 1).sum())
    n_sell_sigs  = int((df["signal"] == -1).sum())
    bsp_min      = round(float(df["bsp"].min()), 4)
    bsp_max      = round(float(df["bsp"].max()), 4)

    with st.expander("🔍 Signal Diagnostics", expanded=(n_buy_sigs == 0 and n_sell_sigs == 0)):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Bars",   f"{total_bars:,}")
        c2.metric("Buy Signals",  n_buy_sigs,  delta=None if n_buy_sigs  > 0 else "⚠️ None")
        c3.metric("Sell Signals", n_sell_sigs, delta=None if n_sell_sigs > 0 else "⚠️ None")
        c4.metric("BSP Min",  bsp_min)
        c5.metric("BSP Max",  bsp_max)

        if n_buy_sigs == 0 and n_sell_sigs == 0:
            st.warning(
                f"**No signals generated.**\n\n"
                f"BSP range on this data: **{bsp_min:.4f} → {bsp_max:.4f}**\n\n"
                f"Your thresholds: Buy > **{bsp_buy_lvl}**, Sell < **{bsp_sell_lvl}**\n\n"
                f"**Fix options (try in order):**\n"
                f"1. Switch Signal Mode to **BSP Only** or **Level Hold** (sidebar)\n"
                f"2. Loosen BSP levels — set Buy to **{round(bsp_max*0.6,3)}** and Sell to **{round(bsp_min*0.6,3)}**\n"
                f"3. Uncheck **EMA Trend Filter**\n"
                f"4. Increase BSP Length (longer smoothing)"
            )
            # Auto-suggest levels based on actual BSP range
            suggested_buy  = round(bsp_max * 0.5, 3)
            suggested_sell = round(bsp_min * 0.5, 3)
            st.info(f"💡 Auto-suggested levels for this dataset: Buy **{suggested_buy}** / Sell **{suggested_sell}**")

    # ── 3. Fetch Option Legs ─────────────────────────────────────────────────
    leg_dfs = []   # one price-series df per leg, aligned to main df index
    if trade_options:
        fetch_interval = interval if interval != "D" else "25"
        fd = from_date.strftime("%Y-%m-%d")
        td = to_date.strftime("%Y-%m-%d")

        for i, leg in enumerate(option_legs):
            lbl = f"Leg {i+1}: {leg['direction']} {leg['lots']}L {leg['strike_lbl']} {leg['opt_type']}"
            with st.spinner(f"📡 Fetching {lbl}…"):
                raw = _fetcher.fetch_rolling_option(
                    idx_cfg["security_id"],
                    leg["offset"], leg["opt_type"],
                    fd, td,
                    interval=fetch_interval,
                    expiry_flag=expiry_flag,
                    debug=debug_mode
                )

            if raw is not None and not raw.empty:
                st.success(f"✅ {lbl} — {len(raw):,} bars")
                # Merge onto main df timestamps
                leg_price = raw[["timestamp","close"]].rename(
                    columns={"close": f"leg{i}_price"}
                )
                merged = pd.merge_asof(
                    df[["timestamp"]].sort_values("timestamp"),
                    leg_price.sort_values("timestamp"),
                    on="timestamp", direction="nearest",
                    tolerance=pd.Timedelta("30min")
                )
                leg_dfs.append({**leg, "prices": merged.set_index("timestamp")[f"leg{i}_price"]})
            else:
                st.warning(f"⚠️ {lbl} — no data, falling back to Black-Scholes simulation.")
                # BS fallback for this leg
                sim = simulate_option_prices(
                    df, leg["offset"], strike_gap, leg["opt_type"], 7, 0.15
                )
                leg_dfs.append({**leg, "prices": sim.set_index("timestamp")["opt_price"]})

        # Build combined spread price column on df:
        # net_price = sum(signed_price per leg)
        # BUY leg  → pay premium  → negative cash at entry, positive at exit
        # SELL leg → receive prem → positive cash at entry, negative at exit
        df = df.set_index("timestamp")
        df["spread_price"] = 0.0
        for ld in leg_dfs:
            sign = 1 if ld["direction"] == "BUY" else -1
            p    = ld["prices"].reindex(df.index).ffill().fillna(0)
            df["spread_price"] += sign * p * ld["lots"]
        df = df.reset_index()

        # Legacy single-leg compat (used by chart subplot)
        if leg_dfs:
            df["opt_price"] = leg_dfs[0]["prices"].reindex(
                df.set_index("timestamp").index
            ).ffill().fillna(0).values

    # ── 4. Run Backtest ───────────────────────────────────────────────────────
    with st.spinner("📊 Running backtest…"):
        results, trades = run_backtest(
            df, init_capital, pos_size_pct / 100,
            comm_pct / 100, lot_size, trade_options,
            fixed_lots=fixed_lots,
            spread_legs=leg_dfs if trade_options else None,
            is_intraday=is_intraday,
            eod_exit_time=eod_exit_time
        )
        m = metrics(results, trades, init_capital)

    # ── 5. Mode Banner ────────────────────────────────────────────────────────
    style_tag = "📅 MIS (Intraday)" if is_intraday else "📆 NRML (Carry Forward)"
    eod_tag   = f" · EOD {eod_exit_time.strftime('%H:%M')}" if is_intraday and eod_exit_time else ""
    if trade_options:
        leg_summaries = " | ".join(
            f"{lg['direction']} {lg['lots']}L {lg['strike_lbl']} {lg['opt_type']}"
            for lg in option_legs
        )
        st.info(f"📌 **Options Mode** · {style_tag}{eod_tag} | {index_name} | {interval_lbl} | {leg_summaries} | {expiry_flag}")
    else:
        st.info(f"**{style_tag}**{eod_tag} · Index Mode | {index_name} | {interval_lbl} | Lot Size: {lot_size}")

    # ── 4. KPIs ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Performance")
    k = st.columns(6)
    k[0].metric("Total Return",  f"{m['total_return']:.1f}%",  delta=f"₹{m['total_pnl']:,.0f}")
    k[1].metric("Profit Factor", f"{m['profit_factor']:.2f}")
    k[2].metric("Win Rate",      f"{m['win_rate']:.1f}%")
    k[3].metric("Max Drawdown",  f"{m['max_drawdown']:.1f}%",  delta_color="inverse")
    k[4].metric("Total Trades",  str(m['total_trades']))
    k[5].metric("Sharpe",        f"{m['sharpe']:.2f}")
    st.divider()

    # ── 5. Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "📋 Trades", "📉 Equity & DD", "📑 Stats"])

    # ─ Chart ──────────────────────────────────────────────────────────────────
    with tab1:
        rows   = 4 if trade_options else 3
        h_rows = [0.50, 0.15, 0.17, 0.18] if trade_options else [0.60, 0.20, 0.20]
        subs   = (["Price + SMC Levels", "Volume", "BSP Oscillator", f"Option Premium ({get_strike_label(strike_offset)} {opt_type})"]
                  if trade_options else
                  ["Price + SMC Levels", "Volume", "BSP Oscillator"])

        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=h_rows, vertical_spacing=0.025,
            subplot_titles=subs
        )

        # Candles
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=index_name,
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4b6e"
        ), row=1, col=1)

        # EMAs
        for show, col, name, color in [
            (show_e20,  "ema20",  "EMA 20",  "#ff4b6e"),
            (show_e50,  "ema50",  "EMA 50",  "#ff9900"),
            (show_e100, "ema100", "EMA 100", "#00bcd4"),
            (show_e200, "ema200", "EMA 200", "#3f8ef5"),
        ]:
            if show:
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=name,
                                          line=dict(color=color, width=1),
                                          opacity=0.85), row=1, col=1)

        # Order Blocks
        for ob in obs[:60]:
            fc = "rgba(255,75,110,0.10)" if ob["type"] == "bearish" else "rgba(0,212,170,0.10)"
            bc = "#ff4b6e"              if ob["type"] == "bearish" else "#00d4aa"
            x0 = df["timestamp"].iloc[ob["start"]]
            x1 = df["timestamp"].iloc[min(ob["end"], len(df) - 1)]
            fig.add_hrect(y0=ob["btm"], y1=ob["top"], x0=x0, x1=x1,
                          fillcolor=fc, line_color=bc, line_width=0.5,
                          annotation_text="OB★" if ob["strong"] else "OB",
                          annotation_font_color=bc, annotation_font_size=8,
                          row=1, col=1)

        # FVGs
        for fvg in fvgs[:40]:
            x0 = df["timestamp"].iloc[fvg["start"]]
            x1 = df["timestamp"].iloc[min(fvg["start"] + 15, len(df) - 1)]
            fig.add_hrect(y0=fvg["btm"], y1=fvg["top"], x0=x0, x1=x1,
                          fillcolor="rgba(150,0,255,0.09)",
                          line_color="#9000ff", line_width=0.5,
                          annotation_text="FVG",
                          annotation_font_color="#b060ff", annotation_font_size=8,
                          row=1, col=1)

        # Signals
        buys  = df[df["signal"] ==  1]
        sells = df[df["signal"] == -1]
        fig.add_trace(go.Scatter(x=buys["timestamp"],  y=buys["low"]  * 0.998,
                                  mode="markers", name="BUY",
                                  marker=dict(symbol="triangle-up",   size=9, color="#00d4aa")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"] * 1.002,
                                  mode="markers", name="EXIT",
                                  marker=dict(symbol="triangle-down", size=9, color="#ff4b6e")), row=1, col=1)

        # Volume
        vcol = ["#00d4aa" if c >= o else "#ff4b6e"
                for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"],
                              name="Volume", marker_color=vcol, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vol_sma"],
                                  name="Vol SMA", line=dict(color="#ff9900", width=1)), row=2, col=1)

        # BSP
        bsp_col = ["#00d4aa" if v > 0 else "#ff4b6e" for v in df["bsp"].fillna(0)]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["bsp"],
                              name="BSP", marker_color=bsp_col, opacity=0.85), row=3, col=1)
        fig.add_hline(y=bsp_buy_lvl,  line_color="#00d4aa", line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=bsp_sell_lvl, line_color="#ff4b6e", line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=0,            line_color="#444",    line_width=1,      row=3, col=1)

        # Option Premium panel
        if trade_options and "opt_price" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["opt_price"],
                name=f"{get_strike_label(strike_offset)} {opt_type}",
                line=dict(color="#d966ff", width=1.5), fill="tozeroy",
                fillcolor="rgba(150,0,255,0.08)"
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=buys["timestamp"],  y=buys["opt_price"],
                mode="markers", name="Buy Option",
                marker=dict(symbol="triangle-up", size=9, color="#00d4aa")
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=sells["timestamp"], y=sells["opt_price"],
                mode="markers", name="Exit Option",
                marker=dict(symbol="triangle-down", size=9, color="#ff4b6e")
            ), row=4, col=1)

        fig.update_layout(
            height=900, template="plotly_dark", showlegend=True,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(t=35, b=20)
        )
        fig.update_xaxes(showgrid=True, gridcolor="#1e2130")
        fig.update_yaxes(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

    # ─ Trades ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Trade Log")
        if trades:
            tdf = pd.DataFrame(trades)

            # Round core numeric columns to 2dp
            for col in ["entry_price","exit_price","pnl","return_pct"]:
                if col in tdf.columns:
                    tdf[col] = tdf[col].round(2)

            # Cumulative P&L
            tdf["cum_pnl"] = tdf["pnl"].cumsum().round(2)

            # Detect leg breakdown columns (leg1_ATM_CALL etc.)
            leg_cols = [c for c in tdf.columns if c.startswith("leg") and "_" in c[3:]]

            # Core columns always shown
            base_cols = ["entry_time","exit_time","entry_price","exit_price",
                         "qty","lots","pnl","cum_pnl","return_pct","exit_reason"]
            # Insert leg breakdown after exit_price if present
            all_cols = ["entry_time","exit_time","entry_price","exit_price",
                        "qty","lots"] + leg_cols + ["pnl","cum_pnl","return_pct","exit_reason"]
            all_cols = [c for c in all_cols if c in tdf.columns]

            rename_map = {
                "entry_time":  "Entry",   "exit_time":   "Exit",
                "entry_price": "Net Entry ₹" if len(leg_cols) > 0 else "Entry ₹",
                "exit_price":  "Net Exit ₹"  if len(leg_cols) > 0 else "Exit ₹",
                "qty":         "Qty",     "lots":        "Scale Lots" if len(leg_cols) > 0 else "Lots",
                "pnl":         "Net P&L ₹",  "cum_pnl":     "Cum. P&L ₹",
                "return_pct":  "Return %",   "exit_reason": "Reason",
            }
            # Pretty-name leg columns: leg1_ATM+1_CALL → "L1 ATM+1 CE ₹"
            for lc in leg_cols:
                parts = lc.split("_", 1)
                leg_num = parts[0].replace("leg", "L")
                rest    = parts[1].replace("_CALL", " CE").replace("_PUT", " PE")
                rename_map[lc] = f"{leg_num} {rest} ₹"

            display = tdf[all_cols].rename(columns=rename_map)

            # All float columns → 2dp format
            float_cols = display.select_dtypes(include="number").columns.tolist()
            fmt_dict   = {c: "{:.2f}" for c in float_cols}

            # Color: green for positive, red for negative
            color_cols = [rename_map.get("pnl","Net P&L ₹"),
                          rename_map.get("cum_pnl","Cum. P&L ₹"),
                          rename_map.get("return_pct","Return %")]
            color_cols += [rename_map[lc] for lc in leg_cols if lc in rename_map]
            color_cols  = [c for c in color_cols if c in display.columns]

            styled = display.style.format(fmt_dict).applymap(
                lambda v: ("color:#00d4aa" if isinstance(v,(int,float)) and v > 0
                           else "color:#ff4b6e" if isinstance(v,(int,float)) and v < 0
                           else ""),
                subset=color_cols
            )

            st.dataframe(styled, use_container_width=True, height=500)

            # Summary bar
            total_pnl = tdf["pnl"].sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net P&L", f"₹{total_pnl:,.2f}",
                      delta=f"{'▲' if total_pnl>0 else '▼'} {total_pnl/init_capital*100:.2f}%")
            c2.metric("Trades",        len(tdf))
            c3.metric("Avg P&L/Trade", f"₹{tdf['pnl'].mean():,.2f}")
            c4.metric("Best Trade",    f"₹{tdf['pnl'].max():,.2f}")

            # Leg-level P&L summary for spreads
            if leg_cols:
                st.markdown("##### 📊 Leg-wise P&L Summary")
                leg_summary_cols = st.columns(len(leg_cols))
                for j, lc in enumerate(leg_cols):
                    nice = rename_map.get(lc, lc)
                    total = tdf[lc].sum()
                    leg_summary_cols[j].metric(nice, f"₹{total:,.2f}",
                        delta=f"{'▲' if total>0 else '▼'}")

            csv = tdf.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv,
                               f"{index_name}_trades.csv", "text/csv")
        else:
            st.info("No trades generated. Try adjusting BSP levels or timeframe.")

    # ─ Equity & Drawdown ──────────────────────────────────────────────────────
    with tab3:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.65, 0.35], vertical_spacing=0.05,
                              subplot_titles=("Equity Curve (₹)", "Drawdown (%)"))
        fig2.add_trace(go.Scatter(
            x=results["timestamp"], y=results["equity"],
            name="Portfolio", fill="tozeroy",
            line=dict(color="#00d4aa", width=2),
            fillcolor="rgba(0,212,170,0.08)"
        ), row=1, col=1)
        fig2.add_hline(y=init_capital, line_color="#555", line_dash="dash", row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=results["timestamp"], y=results["drawdown_pct"],
            name="Drawdown", fill="tozeroy",
            line=dict(color="#ff4b6e", width=1),
            fillcolor="rgba(255,75,110,0.12)"
        ), row=2, col=1)
        fig2.update_layout(
            height=500, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(t=35, b=20)
        )
        fig2.update_xaxes(showgrid=True, gridcolor="#1e2130")
        fig2.update_yaxes(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)

    # ─ Stats ──────────────────────────────────────────────────────────────────
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Return Metrics**")
            for k2, v in [
                ("Total Return (%)",  f"{m['total_return']:.2f}%"),
                ("Total P&L (₹)",     f"₹{m['total_pnl']:,.2f}"),
                ("CAGR (%)",          f"{m['cagr']:.2f}%"),
                ("Sharpe Ratio",      f"{m['sharpe']:.3f}"),
                ("Sortino Ratio",     f"{m['sortino']:.3f}"),
                ("Calmar Ratio",      f"{m['calmar']:.3f}"),
            ]:
                st.markdown(f"`{k2}` &nbsp; **{v}**")
        with c2:
            st.markdown("**Trade Metrics**")
            for k2, v in [
                ("Total Trades",     m["total_trades"]),
                ("Win Rate (%)",     f"{m['win_rate']:.1f}%"),
                ("Profit Factor",    f"{m['profit_factor']:.3f}"),
                ("Avg Win (₹)",      f"₹{m['avg_win']:,.2f}"),
                ("Avg Loss (₹)",     f"₹{m['avg_loss']:,.2f}"),
                ("Max Drawdown (%)", f"{m['max_drawdown']:.2f}%"),
            ]:
                st.markdown(f"`{k2}` &nbsp; **{v}**")

        if len(results) > 30:
            st.markdown("#### Monthly Returns")
            r2 = results.copy()
            r2["month"] = r2["timestamp"].dt.to_period("M")
            monthly = r2.groupby("month")["equity"].last().pct_change() * 100
            monthly.index = monthly.index.astype(str)
            monthly = monthly.dropna()
            fig3 = go.Figure(go.Bar(
                x=monthly.index, y=monthly.values,
                marker_color=["#00d4aa" if v >= 0 else "#ff4b6e" for v in monthly.values],
                text=[f"{v:.1f}%" for v in monthly.values],
                textposition="outside", textfont=dict(size=9)
            ))
            fig3.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                xaxis=dict(tickangle=-45), margin=dict(t=10, b=50),
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig3, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; opacity:0.65;">
        <h2>📡 SMC Liquidity Lens — Index Backtester</h2>
        <p style="font-size:1.05rem;">
            Paste your <b>Dhan Access Token</b>, select an <b>Index</b> and <b>Strike</b>,
            then click <b>Run Backtest</b>.
        </p>
        <br>
        <table style="margin:auto; font-size:0.9rem; border-collapse:collapse; line-height:2.2;">
            <tr>
                <td style="padding:4px 20px;">✅ NIFTY · BANKNIFTY · FINNIFTY · MIDCPNIFTY</td>
                <td style="padding:4px 20px;">✅ SENSEX · BANKEX</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ Index Futures-style Backtest</td>
                <td style="padding:4px 20px;">✅ Options: ATM-3 → ATM → ATM+3</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ BSP + SMC Order Blocks + FVG</td>
                <td style="padding:4px 20px;">✅ Sharpe · Sortino · Calmar · CAGR</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
