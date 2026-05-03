import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Statistical Jump Model", layout="wide")
st.title("📊 Statistical Jump Model (SJM)")
st.caption("Regime detection with persistence | Convex relaxation | Nystrup et al. (2020)")

@st.cache_data(ttl=3600)
def load_latest():
    fs = HfFileSystem(token=config.HF_TOKEN)
    repo = config.HF_OUTPUT_REPO
    try:
        files = fs.ls(f"datasets/{repo}")
        json_files = []
        for f in files:
            name = f['name'] if isinstance(f, dict) else f
            if name.endswith('.json'):
                json_files.append(name)
        if not json_files:
            return None
        latest = max(json_files)
        with fs.open(latest, "r") as fp:
            return json.load(fp)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

data = load_latest()
if not data:
    st.warning("No results. Run trainer.py first.")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")

universes = data['universes']

# For each universe, compute a recommendation score
# Score = current_regime * (1 + log(current_duration / avg_duration))
# This favours high‑regime states that have lasted longer than average
recommendations = {}
for universe_name, uni_data in universes.items():
    best_ticker = None
    best_score = -np.inf
    best_info = None
    for ticker, info in uni_data.items():
        # Avoid division by zero
        avg_dur = max(info["average_duration_days"], 1)
        score = info["current_regime"] * (1 + np.log(info["current_duration_days"] / avg_dur))
        if score > best_score:
            best_score = score
            best_ticker = ticker
            best_info = info
    if best_ticker:
        recommendations[universe_name] = (best_ticker, best_info, best_score)

st.header("🎯 Top ETF Recommendation for Next Trading Day")
for universe, (ticker, info, score) in recommendations.items():
    st.markdown(f"### {universe}")
    col1, col2, col3 = st.columns(3)
    col1.metric("ETF", ticker)
    col2.metric("Current Regime", info["current_regime"])
    col3.metric("Regime Duration (days)", info["current_duration_days"])
    st.caption(f"Regime strength score: {score:.2f} (higher = stronger buy signal)")

    # Show timeline plot for this ticker
    regime_seq = info["regime_sequence"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=regime_seq,
        mode='lines+markers',
        name='Regime',
        line=dict(color='firebrick', width=2)
    ))
    fig.update_layout(
        title=f"Regime timeline – {ticker}",
        xaxis_title="Time (past to present)",
        yaxis_title="Regime (0=low, 1=mid, 2=high)",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# Optional: show full table for debugging
with st.expander("📋 Full Table (All Tickers)"):
    for universe_name, uni_data in universes.items():
        st.subheader(universe_name)
        rows = []
        for ticker, info in uni_data.items():
            rows.append({
                "Ticker": ticker,
                "Current Regime": info["current_regime"],
                "Current Duration": info["current_duration_days"],
                "Avg Duration": info["average_duration_days"],
                "Total Regimes": info["total_regimes"]
            })
        df = pd.DataFrame(rows).sort_values("Current Regime", ascending=False)
        st.dataframe(df, use_container_width=True)

st.caption("Method: Convex trend filtering with persistence penalty. Recommendation = highest current regime × duration score.")
