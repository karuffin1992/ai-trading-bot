import streamlit as st
import pandas as pd
from datetime import date
from sqlalchemy import create_engine, text
from app.config import settings
from app.util.clock import now_utc
import httpx

st.set_page_config(page_title="AI Trading Bot", layout="wide")

@st.cache_resource
def _engine():
    return create_engine(settings.database_url)

def _q(sql, **params):
    with _engine().connect() as c:
        rows = c.execute(text(sql), params).fetchall()
    return [dict(r._mapping) for r in rows]

def _q1(sql, **params):
    with _engine().connect() as c:
        row = c.execute(text(sql), params).fetchone()
    return dict(row._mapping) if row else {}

# Header
summary = _q1("SELECT * FROM daily_summary WHERE date=:d", d=date.today().isoformat())
ks      = _q1("SELECT * FROM kill_switch_state WHERE id=1")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Mode",   settings.trading_mode.upper())
c2.metric("P&L",    f"${summary.get('total_pnl', 0.0):.2f}")
c3.metric("Kill",   "🔴 ON" if ks.get("active") else "🟢 OFF")
c4.markdown(f"**Updated:** {now_utc().strftime('%H:%M:%S UTC')}")
st.divider()

# Pending approval (live_manual only)
if settings.trading_mode == "live_manual":
    pending = _q("SELECT * FROM pending_trades WHERE status='PENDING_APPROVAL'")
    if pending:
        st.subheader("⏳ Pending Approval")
        for pt in pending:
            sig = pt.get("signal_json", {})
            with st.container(border=True):
                a, b, c = st.columns([3,1,1])
                a.markdown(
                    f"**{sig.get('symbol')} {sig.get('direction','').upper()}** "
                    f"@ ${sig.get('entry_price',0):.2f}  \n"
                    f"SL: ${sig.get('stop_loss',0):.2f} | TP: ${sig.get('take_profit',0):.2f}  \n"
                    f"conf: {sig.get('strategy_confidence',0):.2f}"
                )
                if b.button("✅ Approve", key=f"ap_{pt['id']}"):
                    try:
                        r = httpx.post(f"http://localhost:8000/approve/{pt['id']}", timeout=10)
                        if r.status_code == 200: st.rerun()
                        else: st.error(r.json().get("detail"))
                    except Exception as e:
                        st.error(str(e))
                if c.button("❌ Reject", key=f"re_{pt['id']}"):
                    try:
                        r = httpx.post(f"http://localhost:8000/reject/{pt['id']}", timeout=10)
                        if r.status_code == 200: st.rerun()
                        else: st.error(r.json().get("detail"))
                    except Exception as e:
                        st.error(str(e))

# Latest cycle
cycle = _q1("SELECT * FROM cycles ORDER BY started_at DESC LIMIT 1")
if cycle:
    st.subheader("📊 Latest Cycle")
    sig  = cycle.get("signal_json") or {}
    ai   = cycle.get("ai_analysis_json") or {}
    risk = cycle.get("risk_json") or {}
    if sig.get("type") == "SIGNAL":
        sig_label = f"{sig.get('direction','').upper()} @ ${sig.get('entry_price',0):.2f}"
    elif sig.get("type") == "REJECTION":
        sig_label = f"NO TRADE — {', '.join(sig.get('reasons', []))}"
    else:
        sig_label = "N/A"
    st.markdown(f"**Signal:** {sig_label} | conf: {sig.get('strategy_confidence',0):.2f}")
    st.markdown(f"**AI:** {ai.get('decision','N/A')} | ai_conf: {ai.get('ai_confidence',0):.2f} | {ai.get('regime','N/A')}")
    with st.expander("AI Reasoning"):
        st.write(ai.get("reasoning", "None"))
    st.markdown(f"**Risk:** {risk.get('outcome','N/A')} | score: {risk.get('risk_score',0):.3f}")

# Trade history
st.subheader("📋 Trade History")
trades = _q("SELECT * FROM trade_executions ORDER BY created_at DESC LIMIT :n",
            n=settings.trade_history_page_size)
if trades:
    df = pd.DataFrame(trades)[["symbol","direction","entry_price","fill_price",
                                "slippage","broker_state","created_at"]]
    st.dataframe(df, use_container_width=True)
else:
    st.info("No trades yet.")

# Performance
st.subheader("📈 Performance (today)")
if summary:
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Win Rate",    f"{summary.get('win_rate',0)*100:.0f}%")
    p2.metric("Profit Factor", f"{summary.get('profit_factor',0):.2f}")
    p3.metric("Drawdown",    f"{summary.get('drawdown_pct',0)*100:.1f}%")
    p4.metric("Consec Losses", summary.get("consecutive_losses", 0))

import time
time.sleep(settings.dashboard_refresh_seconds)
st.rerun()
