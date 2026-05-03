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
selected_universe = st.selectbox("Select Universe", list(universes.keys()))
uni_data = universes[selected_universe]

if uni_data:
    # Convert to DataFrame for easy display
    rows = []
    for ticker, info in uni_data.items():
        rows.append({
            "Ticker": ticker,
            "Current Regime": info["current_regime"],
            "Current Duration (days)": info["current_duration_days"],
            "Avg Regime Duration": info["average_duration_days"],
            "Total Regimes": info["total_regimes"]
        })
    df = pd.DataFrame(rows).sort_values("Current Regime", ascending=False)

    st.subheader("📈 Current Regime Status for Next Trading Day")
    st.dataframe(df, use_container_width=True)

    # Detailed view for selected ticker
    selected_ticker = st.selectbox("View details for ticker", df["Ticker"].tolist())
    if selected_ticker:
        info = uni_data[selected_ticker]
        st.subheader(f"Regime Timeline – {selected_ticker}")
        # Plot regime sequence
        fig = go.Figure()
        regime_seq = info["regime_sequence"]
        fig.add_trace(go.Scatter(
            y=regime_seq,
            mode='lines+markers',
            name='Regime',
            line=dict(color='firebrick', width=2)
        ))
        fig.update_layout(
            title=f"Regime labels over time",
            xaxis_title="Time step (past to present)",
            yaxis_title="Regime (0=low,1=mid,2=high)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔄 Transition Points")
        trans_df = pd.DataFrame(info["transition_points"])
        if not trans_df.empty:
            st.dataframe(trans_df)
        else:
            st.write("No regime changes detected.")
else:
    st.info("No data for this universe.")

st.caption("Method: Convex relaxation of piecewise constant mean + persistence penalty. See Nystrup et al. (2020)")
