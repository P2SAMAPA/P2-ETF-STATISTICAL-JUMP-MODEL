import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Statistical Jump Model", layout="wide")
st.title("📊 Statistical Jump Model (SJM)")
st.caption("Regime detection with persistence | Returns‑chasing recommendation")

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

# For each universe, pick ticker with highest expected return from its current regime
recommendations = {}
for universe_name, uni_data in universes.items():
    if not uni_data:
        continue
    best_ticker = None
    best_expected_return = -np.inf
    best_info = None
    for ticker, info in uni_data.items():
        curr_reg = info["current_regime"]
        exp_return = info.get("regime_returns", {}).get(curr_reg, -np.inf)
        if exp_return > best_expected_return:
            best_expected_return = exp_return
            best_ticker = ticker
            best_info = info
    if best_ticker:
        recommendations[universe_name] = (best_ticker, best_info, best_expected_return)

st.header("🎯 Top ETF Recommendation for Next Trading Day")
for universe, (ticker, info, exp_return) in recommendations.items():
    st.markdown(f"### {universe}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ETF", ticker)
    col2.metric("Current Regime", info["current_regime"])
    col3.metric("Regime Duration (days)", info["current_duration_days"])
    # Display percentage with 2 decimal places (e.g., 0.05%)
    col4.metric("Expected Annual Return", f"{exp_return*100:.2f}%")
    st.caption("Annualised simple return (mean daily × 252) for the current regime.")

    # Timeline plot with dates
    dates = pd.to_datetime(info["dates"])
    regime_seq = info["regime_sequence"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=regime_seq,
        mode='lines+markers',
        name='Regime',
        line=dict(color='firebrick', width=2),
        marker=dict(size=3)
    ))
    fig.update_layout(
        title=f"Regime timeline – {ticker}",
        xaxis_title="Date",
        yaxis_title="Regime (0=low, 1=mid, 2=high)",
        height=400,
        xaxis=dict(tickformat="%Y", dtick="M12")
    )
    st.plotly_chart(fig, use_container_width=True, key=f"regime_plot_{universe}")
    st.divider()

# Optional full table
with st.expander("📋 Full Table (All Tickers)"):
    for universe_name, uni_data in universes.items():
        if not uni_data:
            continue
        st.subheader(universe_name)
        rows = []
        for ticker, info in uni_data.items():
            curr_reg = info["current_regime"]
            exp_return = info.get("regime_returns", {}).get(curr_reg, 0.0)
            rows.append({
                "Ticker": ticker,
                "Current Regime": curr_reg,
                "Current Duration": info["current_duration_days"],
                "Exp. Annual Return": f"{exp_return*100:.2f}%",
                "Total Regimes": info["total_regimes"]
            })
        df = pd.DataFrame(rows).sort_values("Exp. Annual Return", ascending=False)
        st.dataframe(df, use_container_width=True)

st.caption("Method: Convex trend filtering. Recommendation = ticker with highest historical return for its current regime.")
