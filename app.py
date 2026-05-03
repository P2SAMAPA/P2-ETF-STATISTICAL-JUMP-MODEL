"""app.py — Statistical Jump Model Dashboard."""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="Statistical Jump Model · P2Quant",
    layout="wide",
    page_icon="📊",
)

# ── Colours ───────────────────────────────────────────────────────────────────
REGIME_COLOURS = {0: "#E74C3C", 1: "#F39C12", 2: "#27AE60"}
REGIME_NAMES = {0: "Low", 1: "Mid", 2: "High"}
SIGNAL_COLOURS = {"Low": "#E74C3C", "Mid": "#F39C12", "High": "#27AE60"}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading latest SJM results…")
def load_latest() -> dict | None:
    # HF public dataset — no auth needed for reading
    base_api = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
    base_raw = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
    headers = {"Authorization": f"Bearer {config.HF_TOKEN}"} if config.HF_TOKEN else {}

    try:
        resp = requests.get(base_api, headers=headers, timeout=30)
        if resp.status_code == 404:
            return None  # repo doesn't exist yet
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    try:
        all_files = sorted(
            [f["path"] for f in resp.json() if f["path"].endswith(".json")]
        )
    except Exception:
        return None

    if not all_files:
        return None

    # Latest file per universe slug
    universe_files: dict[str, str] = {}
    for path in all_files:
        name = path.split("/")[-1]
        parts = name.replace(".json", "").split("_")
        slug = parts[-1] if len(parts) >= 3 else "all"
        universe_files[slug] = path  # keeps latest (sorted)

    merged: dict = {}
    run_date = "unknown"
    for slug, path in universe_files.items():
        try:
            r = requests.get(f"{base_raw}/{path}", headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            run_date = data.get("run_date", run_date)
            merged.update(data.get("universes", {}))
        except Exception:
            continue  # skip bad files, keep loading others

    if not merged:
        return None

    return {"run_date": run_date, "universes": merged}


def regime_badge(regime_id: int) -> str:
    name = REGIME_NAMES.get(regime_id, str(regime_id))
    colour = REGIME_COLOURS.get(regime_id, "#888")
    return (
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;font-size:13px">{name}</span>'
    )


def fmt_return(val: float) -> str:
    if abs(val) < 1e-6:
        return "0.00%"
    return f"{val * 100:+.2f}%"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Statistical Jump Model")
st.caption(
    "Convex trend filtering detects persistent volatility regimes · "
    "Regimes labelled Low / Mid / High by geometrically compounded annualised return · "
    "Returns are historical medians within each regime — not forecasts · "
    "Rankings by current regime historical return"
)

data = load_latest()
if not data:
    st.warning("⚠️ No results found. Run `trainer.py` first.")
    st.stop()

run_date = data.get("run_date", "unknown")
next_trade = next_trading_day()

h1, h2, h3 = st.columns(3)
h1.metric("Run Date", run_date)
h2.metric("Next Trading Day", str(next_trade))
h3.metric("Universes", len(data.get("universes", {})))
st.divider()

universes = data.get("universes", {})
if not universes:
    st.warning("No universe data in results.")
    st.stop()

# ── Universe tabs ─────────────────────────────────────────────────────────────
tab_icons = {"FI_COMMODITIES": "🏦", "EQUITY_SECTORS": "📊", "COMBINED": "🌐"}
tab_labels = [f"{tab_icons.get(u, '📈')} {u}" for u in universes]
tabs = st.tabs(tab_labels)

for tab, universe_name in zip(tabs, universes):
    with tab:
        uni = universes[universe_name]
        tickers_data = uni.get("tickers", {})
        rankings = uni.get("rankings", [])

        if not tickers_data:
            st.info("No data for this universe.")
            continue

        sorted_tickers = sorted(
            tickers_data.items(), key=lambda x: x[1].get("rank", 99)
        )
        top_ticker, top_info = sorted_tickers[0]
        curr_regime = top_info.get("current_regime", 0)
        curr_name = top_info.get("current_regime_name", "?")
        curr_return = top_info.get("current_annual_return", 0.0)

        # ── Hero card ─────────────────────────────────────────────────────────
        st.markdown(f"## 🎯 Top Pick for {next_trade}")
        hero_l, hero_r = st.columns([3, 2])

        with hero_l:
            st.markdown(
                f"### {top_ticker} &nbsp; {regime_badge(curr_regime)}",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Regime", curr_name)
            m2.metric("Expected Annual Return", fmt_return(curr_return))
            m3.metric(
                "Regime Duration",
                f"{top_info.get('current_duration_days', 0)} days",
            )
            m4.metric("# Changepoints", top_info.get("n_changepoints", 0))

            # Regime return table for top ticker
            reg_returns = top_info.get("regime_returns", {})
            reg_names = top_info.get("regime_names", {})
            if reg_returns:
                st.markdown("**Return by Regime**")
                reg_rows = [
                    {
                        "Regime": f"{reg_names.get(int(k), k)} {'◄ current' if int(k) == curr_regime else ''}",
                        "Annualised Return": fmt_return(v),
                    }
                    for k, v in sorted(reg_returns.items())
                ]
                st.dataframe(
                    pd.DataFrame(reg_rows), use_container_width=True, hide_index=True
                )

        with hero_r:
            # Regime timeline for top ticker
            dates = pd.to_datetime(top_info.get("dates", []))
            regime_seq = top_info.get("regime_sequence", [])
            if len(dates) > 0 and len(regime_seq) > 0:
                fig_hero = go.Figure()
                # Colour-coded background bands per regime
                regime_arr = np.array(regime_seq)
                for reg_id, colour in REGIME_COLOURS.items():
                    mask = regime_arr == reg_id
                    if not mask.any():
                        continue
                    fig_hero.add_trace(
                        go.Scatter(
                            x=dates[mask],
                            y=np.where(mask, reg_id, np.nan)[mask],
                            mode="markers",
                            marker=dict(color=colour, size=3),
                            name=REGIME_NAMES.get(reg_id, str(reg_id)),
                            showlegend=True,
                        )
                    )
                fig_hero.update_layout(
                    title=f"{top_ticker} — Regime Timeline",
                    yaxis=dict(
                        tickvals=[0, 1, 2],
                        ticktext=["Low", "Mid", "High"],
                        title="Regime",
                    ),
                    xaxis_title="Date",
                    height=300,
                    margin=dict(t=40, b=40, l=50, r=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(
                    fig_hero, use_container_width=True, key=f"hero_{universe_name}"
                )

        st.divider()

        # ── Full rankings ─────────────────────────────────────────────────────
        st.subheader(f"📋 Full Rankings — {universe_name}")

        rank_rows = []
        for ticker, info in sorted_tickers:
            reg = info.get("current_regime", 0)
            rank_rows.append(
                {
                    "Rank": info.get("rank", "-"),
                    "Ticker": ticker,
                    "Regime": info.get("current_regime_name", "?"),
                    "Expected Return": fmt_return(info.get("current_annual_return", 0)),
                    "Duration (days)": info.get("current_duration_days", 0),
                    "Avg Duration": info.get("average_duration_days", 0),
                    "Changepoints": info.get("n_changepoints", 0),
                    "Low Ret": fmt_return(info.get("regime_returns", {}).get(0, 0)),
                    "Mid Ret": fmt_return(info.get("regime_returns", {}).get(1, 0)),
                    "High Ret": fmt_return(info.get("regime_returns", {}).get(2, 0)),
                }
            )
        ranks_df = pd.DataFrame(rank_rows)

        chart_col, table_col = st.columns([1, 1])

        with chart_col:
            # Horizontal bar — expected return ranked
            tickers_ord = [r["Ticker"] for r in rank_rows]
            returns_ord = [
                info.get("current_annual_return", 0) for _, info in sorted_tickers
            ]
            bar_colours = [
                REGIME_COLOURS.get(info.get("current_regime", 0), "#888")
                for _, info in sorted_tickers
            ]
            fig_bar = go.Figure(
                go.Bar(
                    y=tickers_ord,
                    x=[r * 100 for r in returns_ord],
                    orientation="h",
                    marker_color=bar_colours,
                    text=[fmt_return(r) for r in returns_ord],
                    textposition="outside",
                )
            )
            fig_bar.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_bar.update_layout(
                title="Expected Annual Return by Current Regime",
                xaxis_title="Annualised Return (%)",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(tickers_ord) * 32),
                margin=dict(t=40, b=30, l=60, r=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(
                fig_bar, use_container_width=True, key=f"bar_{universe_name}"
            )

        with table_col:
            st.dataframe(
                ranks_df,
                use_container_width=True,
                hide_index=True,
                height=max(300, len(ranks_df) * 35 + 40),
            )

        # ── Regime heatmap: all tickers × current regime returns ──────────────
        st.subheader("🌡️ Regime Return Heatmap")
        heat_tickers = [t for t, _ in sorted_tickers]
        heat_data = []
        for _, info in sorted_tickers:
            rr = info.get("regime_returns", {})
            heat_data.append(
                [rr.get(0, 0) * 100, rr.get(1, 0) * 100, rr.get(2, 0) * 100]
            )

        fig_heat = go.Figure(
            go.Heatmap(
                z=heat_data,
                x=["Low Regime", "Mid Regime", "High Regime"],
                y=heat_tickers,
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Ann. Return %"),
                text=[[f"{v:.1f}%" for v in row] for row in heat_data],
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig_heat.update_layout(
            height=max(300, len(heat_tickers) * 28 + 80),
            margin=dict(t=20, b=40, l=60, r=20),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{universe_name}")

        # ── Per-ticker detail ─────────────────────────────────────────────────
        st.subheader("🔬 Per-Ticker Regime Timeline")
        for ticker, info in sorted_tickers:
            curr = info.get("current_regime", 0)
            curr_n = info.get("current_regime_name", "?")
            ann_r = info.get("current_annual_return", 0.0)
            with st.expander(
                f"#{info.get('rank','-')}  {ticker}  —  "
                f"Regime: {curr_n}  |  Expected: {fmt_return(ann_r)}"
            ):
                dates = pd.to_datetime(info.get("dates", []))
                regime_seq = info.get("regime_sequence", [])

                if len(dates) > 0 and len(regime_seq) > 0:
                    regime_arr = np.array(regime_seq)
                    fig_t = go.Figure()
                    for reg_id, colour in REGIME_COLOURS.items():
                        mask = regime_arr == reg_id
                        if not mask.any():
                            continue
                        fig_t.add_trace(
                            go.Scatter(
                                x=dates[mask],
                                y=np.where(mask, reg_id, np.nan)[mask],
                                mode="markers",
                                marker=dict(color=colour, size=4),
                                name=REGIME_NAMES.get(reg_id, str(reg_id)),
                            )
                        )
                    fig_t.update_layout(
                        yaxis=dict(
                            tickvals=[0, 1, 2],
                            ticktext=["Low", "Mid", "High"],
                        ),
                        height=220,
                        margin=dict(t=10, b=30, l=50, r=10),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        xaxis=dict(tickformat="%Y"),
                    )
                    st.plotly_chart(
                        fig_t,
                        use_container_width=True,
                        key=f"detail_{universe_name}_{ticker}",
                    )

                d1, d2, d3 = st.columns(3)
                d1.metric("Duration", f"{info.get('current_duration_days', 0)} days")
                d2.metric(
                    "Avg Duration", f"{info.get('average_duration_days', 0)} days"
                )
                d3.metric("Changepoints", info.get("n_changepoints", 0))

                rr = info.get("regime_returns", {})
                rn = info.get("regime_names", {})
                if rr:
                    ret_rows = [
                        {
                            "Regime": f"{rn.get(int(k), k)}{'  ◄ current' if int(k) == curr else ''}",
                            "Ann. Return": fmt_return(v),
                        }
                        for k, v in sorted(rr.items())
                    ]
                    st.dataframe(
                        pd.DataFrame(ret_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

st.divider()
st.caption(
    f"P2Quant SJM Engine · Run: {run_date} · "
    "Data: P2SAMAPA/fi-etf-macro-signal-master-data · "
    "Results: P2SAMAPA/p2-etf-sjm-results"
)
