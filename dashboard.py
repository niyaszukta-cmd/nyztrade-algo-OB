"""
NYZTrade · Engine Monitor Dashboard
Streamlit Web Service — Render.com
Open only when you want to check. Engine runs independently.
"""

import streamlit as st
import json, os, time, math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from zoneinfo import ZoneInfo

IST        = ZoneInfo("Asia/Kolkata")
STATE_FILE = "state.json"

st.set_page_config(
    page_title="NYZTrade Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { background-color: #060a12 !important; }
.main, section[data-testid="stSidebar"] { background-color: #060a12 !important; }

div[data-testid="metric-container"] {
    background: #0d1220 !important;
    border: 1px solid #1c2235 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}

.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: #ffffff; letter-spacing: -0.04em;
    margin-bottom: 2px;
}
.page-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #4a5568; margin-bottom: 0;
}

.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 7px 16px; border-radius: 30px;
}
.status-running  { background: rgba(0,229,160,0.1);  color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }
.status-closed   { background: rgba(100,116,139,0.1); color: #64748b; border: 1px solid rgba(100,116,139,0.25); }
.status-error    { background: rgba(255,75,110,0.1);  color: #ff4b6e; border: 1px solid rgba(255,75,110,0.3); }
.status-dot      { width: 7px; height: 7px; border-radius: 50%; }
.dot-green       { background: #00e5a0; box-shadow: 0 0 6px #00e5a0; animation: pulse 1.6s infinite; }
.dot-grey        { background: #64748b; }
.dot-red         { background: #ff4b6e; }
@keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:0.4} }

.pos-card {
    background: #0d1220; border: 1px solid #1c2235;
    border-left: 3px solid #00e5a0;
    border-radius: 10px; padding: 16px 20px; margin: 10px 0;
}
.pos-header {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; color: #fff; margin-bottom: 6px;
}
.pos-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #4a5568; line-height: 1.7;
}
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.08em;
    padding: 2px 9px; border-radius: 4px; margin-right: 5px;
}
.badge-carry   { background: rgba(245,200,66,0.1);  color: #f5c842; border: 1px solid rgba(245,200,66,0.25); }
.badge-intra   { background: rgba(77,159,255,0.1);  color: #4d9fff; border: 1px solid rgba(77,159,255,0.25); }
.badge-paper   { background: rgba(181,123,255,0.1); color: #b57bff; border: 1px solid rgba(181,123,255,0.25); }
.badge-live    { background: rgba(255,75,110,0.1);  color: #ff4b6e; border: 1px solid rgba(255,75,110,0.25); }

.log-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem; padding: 5px 0;
    border-bottom: 1px solid #0d1220; line-height: 1.5;
}
.log-BUY     { color: #00e5a0; }
.log-EXIT    { color: #ff4b6e; }
.log-INFO    { color: #4a5568; }
.log-NEUTRAL { color: #4a5568; }
.log-EOD     { color: #f5a623; }
.log-ERROR   { color: #ff4b6e; font-weight: 600; }

.sig-box {
    background: #0d1220; border: 1px solid #1c2235;
    border-radius: 10px; padding: 16px 20px; margin: 12px 0;
}
.sig-buy  { border-left: 4px solid #00e5a0; }
.sig-exit { border-left: 4px solid #ff4b6e; }
.sig-neut { border-left: 4px solid #1c2235; }
.sig-label {
    font-family: 'Syne', sans-serif; font-size: 1.3rem;
    font-weight: 800; letter-spacing: -0.02em;
}
.sig-reason {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #4a5568; margin-top: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: #0d1220 !important;
    border: 1px solid #1c2235 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    background: #162035 !important;
    color: #00e5a0 !important;
    border-color: rgba(0,229,160,0.3) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load state ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def get_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {}


# ── Header ────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown('<div class="page-title">📡 NYZTrade · Algo Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Engine runs 24/7 — open this only to check status</div>', unsafe_allow_html=True)
with hcol2:
    refresh = st.selectbox("Refresh", [10, 15, 30, 60, "Manual"], index=2, label_visibility="collapsed")

state = get_state()

if not state:
    st.warning("⏳ No state file found. Waiting for engine to start...")
    st.info("Make sure `engine.py` background worker is deployed and running on Render.")
    if refresh != "Manual":
        time.sleep(int(refresh))
        st.rerun()
    st.stop()

# ── Status Row ────────────────────────────────────────────────────────────────
eng    = state.get("engine_status", "UNKNOWN")
last_t = state.get("last_check_ts", "—")
ticks  = state.get("total_ticks", 0)

scol1, scol2, scol3 = st.columns([2, 2, 1])
with scol1:
    if eng == "RUNNING":
        st.markdown('<div class="status-pill status-running"><span class="status-dot dot-green"></span>ENGINE RUNNING</div>', unsafe_allow_html=True)
    elif eng == "MARKET_CLOSED":
        st.markdown('<div class="status-pill status-closed"><span class="status-dot dot-grey"></span>MARKET CLOSED</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-pill status-error"><span class="status-dot dot-red"></span>{eng}</div>', unsafe_allow_html=True)
with scol2:
    st.markdown(f'<div class="pos-meta">Last tick: <b style="color:#fff">{last_t}</b> IST</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pos-meta">Total ticks: <b style="color:#fff">{ticks:,}</b></div>', unsafe_allow_html=True)
with scol3:
    paper = state.get("positions", [{}])
    is_paper = any(p.get("paper", True) for p in state.get("trade_log", [{"paper": True}]))
    st.markdown(f'<span class="badge badge-paper">{"PAPER" if is_paper else "LIVE"}</span>', unsafe_allow_html=True)

st.markdown("---")

# ── Live Signal Box ────────────────────────────────────────────────────────────
sig_val    = state.get("last_signal", 0)
sig_reason = state.get("last_signal_reason", "—")
sig_bsp    = state.get("last_bsp",   0)
sig_close  = state.get("last_close", 0)
sig_e20    = state.get("last_ema20", 0)
sig_e50    = state.get("last_ema50", 0)

sig_cls = "sig-buy" if sig_val==1 else "sig-exit" if sig_val==-1 else "sig-neut"
sig_lbl = "🟢 BUY"  if sig_val==1 else "🔴 EXIT"  if sig_val==-1 else "⚪ NEUTRAL"
sig_col = "#00e5a0" if sig_val==1 else "#ff4b6e"  if sig_val==-1 else "#4a5568"

st.markdown(
    f'<div class="sig-box {sig_cls}">'
    f'<span class="sig-label" style="color:{sig_col}">{sig_lbl}</span>'
    f'<div class="sig-reason">{sig_reason}</div>'
    f'</div>',
    unsafe_allow_html=True
)

# ── KPI Metrics ───────────────────────────────────────────────────────────────
init_cap = float(state.get("initial_capital", 500000))
cur_cap  = float(state.get("capital",         init_cap))
net_pnl  = cur_cap - init_cap
ret_pct  = net_pnl / init_cap * 100 if init_cap else 0
trades   = state.get("trade_log", [])
open_pos = [p for p in state.get("positions", []) if p.get("status") == "OPEN"]
wins     = [t for t in trades if float(t.get("pnl", 0)) > 0]
wr       = round(len(wins)/len(trades)*100, 1) if trades else 0.0

m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
m1.metric("Capital",       f"₹{cur_cap:,.0f}")
m2.metric("Net P&L",       f"₹{net_pnl:,.0f}",     delta=f"{ret_pct:+.2f}%")
m3.metric("Open Positions",f"{len(open_pos)}")
m4.metric("Total Trades",  f"{len(trades)}")
m5.metric("Win Rate",      f"{wr}%")
m6.metric("BSP",           f"{sig_bsp:.4f}")
m7.metric("Close",         f"₹{sig_close:,.0f}")
m8.metric("EMA20/50",      f"{sig_e20:,.0f}/{sig_e50:,.0f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "📂 Positions", "📋 Trades", "📉 Equity", "📜 Signal Log", "⚠️ Errors"
])

# ── Positions Tab ─────────────────────────────────────────────────────────────
with t1:
    st.markdown("#### Open Positions")
    if not open_pos:
        st.info("No open positions right now.")
    for pos in open_pos:
        carry_b  = '<span class="badge badge-carry">CARRY/CNC</span>'  if pos.get("carry") else '<span class="badge badge-intra">INTRADAY</span>'
        paper_b  = '<span class="badge badge-paper">PAPER</span>'       if pos.get("entry_order",{}).get("_paper") else '<span class="badge badge-live">LIVE</span>'
        entry_p  = float(pos.get("entry_price", 0))
        cur_mtm  = sig_close
        pnl_mtm  = (cur_mtm - entry_p) * pos.get("lots", 1) * 75
        pnl_col  = "#00e5a0" if pnl_mtm >= 0 else "#ff4b6e"
        st.markdown(
            f'<div class="pos-card">'
            f'<div class="pos-header">{pos["id"]} {carry_b} {paper_b}</div>'
            f'<div class="pos-meta">'
            f'Entry: <b style="color:#fff">₹{entry_p:,.2f}</b> &nbsp;|&nbsp; '
            f'Lots: <b style="color:#fff">{pos.get("lots",1)}</b> &nbsp;|&nbsp; '
            f'Unrealised MTM: <b style="color:{pnl_col}">₹{pnl_mtm:,.0f}</b><br>'
            f'BSP at entry: {pos.get("bsp_at_entry",0):.4f} &nbsp;|&nbsp; '
            f'Entry time: {pos.get("entry_time","?")} &nbsp;|&nbsp; '
            f'{pos.get("signal_reason","")}'
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("#### All Positions")
    all_pos = state.get("positions", [])
    if all_pos:
        pos_rows = []
        for p in all_pos:
            pos_rows.append({
                "ID":         p.get("id",""),
                "Status":     p.get("status",""),
                "Entry":      p.get("entry_time",""),
                "Exit":       p.get("exit_time","—"),
                "Entry ₹":    float(p.get("entry_price",0)),
                "Exit ₹":     float(p.get("exit_price",0)) if p.get("exit_price") else 0,
                "P&L ₹":      float(p.get("realized_pnl",0)),
                "Product":    p.get("product",""),
                "Carry":      "✓" if p.get("carry") else "—",
                "Reason":     p.get("exit_reason","—"),
            })
        pdf = pd.DataFrame(pos_rows)
        st.dataframe(
            pdf.style.applymap(
                lambda v: "color:#00e5a0" if isinstance(v,(int,float)) and v>0
                     else "color:#ff4b6e" if isinstance(v,(int,float)) and v<0 else "",
                subset=["P&L ₹"]
            ),
            use_container_width=True, height=300
        )

# ── Trades Tab ────────────────────────────────────────────────────────────────
with t2:
    st.markdown("#### Closed Trade Log")
    if not trades:
        st.info("No closed trades yet.")
    else:
        tdf = pd.DataFrame(trades)
        for col in ["entry_price","exit_price","pnl"]:
            if col in tdf.columns:
                tdf[col] = pd.to_numeric(tdf[col], errors="coerce").round(2)
        tdf["cum_pnl"] = tdf["pnl"].cumsum().round(2)

        show = [c for c in ["entry_time","exit_time","entry_price","exit_price",
                             "lots","pnl","cum_pnl","reason","carry","paper"] if c in tdf.columns]
        styled = tdf[show].style.format({
            "entry_price":"{:.2f}", "exit_price":"{:.2f}",
            "pnl":"{:.2f}",        "cum_pnl":"{:.2f}"
        }).applymap(
            lambda v: "color:#00e5a0" if isinstance(v,(int,float)) and v>0
                 else "color:#ff4b6e" if isinstance(v,(int,float)) and v<0 else "",
            subset=["pnl","cum_pnl"]
        )
        st.dataframe(styled, use_container_width=True, height=400)

        tc1,tc2,tc3,tc4 = st.columns(4)
        tc1.metric("Total P&L",  f"₹{tdf['pnl'].sum():,.2f}")
        tc2.metric("Avg P&L",    f"₹{tdf['pnl'].mean():,.2f}")
        tc3.metric("Best",       f"₹{tdf['pnl'].max():,.2f}")
        tc4.metric("Worst",      f"₹{tdf['pnl'].min():,.2f}")
        st.download_button("⬇ Export CSV", tdf.to_csv(index=False),
                           "nyztrade_trades.csv", "text/csv")

# ── Equity Tab ────────────────────────────────────────────────────────────────
with t3:
    eq = state.get("equity_curve", [])
    if not eq:
        st.info("Equity curve will appear as trades close.")
    else:
        edf = pd.DataFrame(eq)
        edf["ts"]     = pd.to_datetime(edf["ts"])
        edf["equity"] = pd.to_numeric(edf["equity"], errors="coerce")
        edf["peak"]   = edf["equity"].cummax()
        edf["dd"]     = (edf["equity"] - edf["peak"]) / edf["peak"] * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65,0.35], vertical_spacing=0.04,
                            subplot_titles=["Equity Curve ₹","Drawdown %"])
        fig.add_trace(go.Scatter(
            x=edf["ts"], y=edf["equity"],
            fill="tozeroy", name="Equity",
            line=dict(color="#00e5a0", width=2),
            fillcolor="rgba(0,229,160,0.06)"
        ), row=1, col=1)
        fig.add_hline(y=init_cap, line_color="#2d3748", line_dash="dash",
                      line_width=1, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=edf["ts"], y=edf["dd"],
            fill="tozeroy", name="Drawdown",
            line=dict(color="#ff4b6e", width=1.5),
            fillcolor="rgba(255,75,110,0.08)"
        ), row=2, col=1)
        fig.update_layout(
            height=420, template="plotly_dark",
            paper_bgcolor="#060a12", plot_bgcolor="#060a12",
            showlegend=False, margin=dict(t=30, b=10)
        )
        fig.update_xaxes(gridcolor="#0d1220", showgrid=True)
        fig.update_yaxes(gridcolor="#0d1220", showgrid=True)
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        if len(edf) > 1:
            try:
                daily = edf.set_index("ts")["equity"].resample("D").last().dropna()
                dr    = daily.pct_change().dropna()
                sharpe = round(float(dr.mean()/dr.std()*math.sqrt(252)),2) if len(dr)>1 and dr.std()>0 else 0
                max_dd = round(edf["dd"].min(), 2)
                days   = max((edf["ts"].iloc[-1]-edf["ts"].iloc[0]).days, 1)
                cagr   = round((max(edf["equity"].iloc[-1]/init_cap,1e-9)**(365/days)-1)*100,2) if days>0 else 0
                sc1,sc2,sc3 = st.columns(3)
                sc1.metric("CAGR",        f"{cagr:.1f}%")
                sc2.metric("Sharpe",      f"{sharpe:.2f}")
                sc3.metric("Max Drawdown",f"{max_dd:.1f}%")
            except:
                pass

# ── Signal Log Tab ────────────────────────────────────────────────────────────
with t4:
    st.markdown("#### Signal & Event Log")
    slog = list(reversed(state.get("signal_log", [])))
    if not slog:
        st.info("No events logged yet.")
    else:
        for row in slog[:150]:
            lvl = row.get("level","INFO")
            ts  = row.get("ts","")
            msg = row.get("msg","")
            st.markdown(
                f'<div class="log-row">'
                f'<span style="color:#2d3748">[{ts}]</span> '
                f'<span class="log-{lvl}">{msg}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ── Errors Tab ────────────────────────────────────────────────────────────────
with t5:
    elog = list(reversed(state.get("error_log", [])))
    if not elog:
        st.success("✅ No errors logged.")
    else:
        st.warning(f"⚠️ {len(elog)} error(s) logged — most recent first")
        for row in elog[:50]:
            st.markdown(
                f'<div class="log-row log-ERROR">[{row.get("ts","")}] {row.get("msg","")}</div>',
                unsafe_allow_html=True
            )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if refresh != "Manual":
    time.sleep(int(refresh))
    st.cache_data.clear()
    st.rerun()
