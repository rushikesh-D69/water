"""
dashboard/app.py — Stage 2 Streamlit Dashboard
=================================================
5 Interactive Panels:

  Panel 1 — Telescope Queue        : Current round targets + priority/uncertainty
  Panel 2 — Priority Evolution     : Top-20 planet scores across rounds
  Panel 3 — Uncertainty Heatmap    : Which planets still need data
  Panel 4 — Campaign Timeline      : Gantt-style observation schedule
  Panel 5 — Metrics Dashboard      : Live NDCG, gain, regret, diversity, utilization

Usage (from project root):
  streamlit run dashboard/app.py

Or in Colab:
  !streamlit run dashboard/app.py &
  from google.colab.output import eval_js
  print(eval_js('google.colab.kernel.proxyPort(8501)'))
"""

import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "ExoPlanet Scheduler — Stage 2",
    page_icon   = "🔭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "teal":    "#22b5a0",
    "gold":    "#f0a500",
    "pink":    "#c9ada7",
    "blue":    "#7bccf6",
    "white":   "#e6edf3",
    "muted":   "#8b949e",
    "green":   "#3fb950",
    "purple":  "#bc8cff",
}

SCHED_COLORS = {
    "Static Priority":      COLORS["gold"],
    "Detectability Greedy": COLORS["pink"],
    "Uncertainty Greedy":   COLORS["blue"],
    "Adaptive Scheduler":   COLORS["teal"],
    "Oracle":               COLORS["white"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["panel"],
    font          = dict(color=COLORS["white"], family="Inter, sans-serif"),
    xaxis         = dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis         = dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    margin        = dict(l=40, r=20, t=40, b=40),
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 4px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #22b5a0;
    margin: 0;
}
.metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 4px;
}
.metric-delta {
    font-size: 0.75rem;
    color: #3fb950;
    margin-top: 2px;
}

.panel-header {
    background: linear-gradient(90deg, #22b5a0 0%, #7bccf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.planet-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #22b5a0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.planet-name {
    font-weight: 600;
    color: #e6edf3;
    font-size: 0.95rem;
}
.planet-meta {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 2px;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-high   { background: #22b5a022; color: #22b5a0; border: 1px solid #22b5a066; }
.badge-medium { background: #f0a50022; color: #f0a500; border: 1px solid #f0a50066; }
.badge-low    { background: #c9ada722; color: #c9ada7; border: 1px solid #c9ada766; }

stMetricLabel { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading planet data...")
def load_planet_data():
    path = ROOT / "data" / "exoplanets_processed.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner="Loading Stage 2 results...")
def load_stage2_results():
    data = {}
    data_dir = ROOT / "data"

    schedulers = [
        "static_priority", "detectability_greedy",
        "uncertainty_greedy", "adaptive_scheduler", "oracle"
    ]
    name_map = {
        "static_priority":      "Static Priority",
        "detectability_greedy": "Detectability Greedy",
        "uncertainty_greedy":   "Uncertainty Greedy",
        "adaptive_scheduler":   "Adaptive Scheduler",
        "oracle":               "Oracle",
    }

    for key in schedulers:
        log_path = data_dir / f"s2_{key}_logs.csv"
        if log_path.exists():
            data[name_map[key]] = pd.read_csv(log_path)

    obs_path = data_dir / "s2_adaptive_obs_history.csv"
    obs_df   = pd.read_csv(obs_path) if obs_path.exists() else pd.DataFrame()

    comp_path = data_dir / "stage2_comparison.csv"
    comp_df   = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()

    final_path = data_dir / "final_priority_ranking.csv"
    final_df   = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()

    return data, obs_df, comp_df, final_df


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔭 ExoPlanet Scheduler")
        st.markdown("**Stage 2: Adaptive Observation Scheduling**")
        st.markdown("---")

        panel = st.radio(
            "Navigate to Panel",
            options=[
                "Telescope Queue",
                "Priority Evolution",
                "Uncertainty Heatmap",
                "Campaign Timeline",
                "Metrics Dashboard",
            ],
            index = 0,
        )

        st.markdown("---")

        # Simulation config display
        st.markdown("**Simulation Config**")
        st.caption(f"Rounds: `30`")
        st.caption(f"Planets/round: `10`")
        st.caption(f"Budget: `8 hrs/night`")
        st.caption(f"Weather: `AR(1) ρ=0.65`")
        st.caption(f"Exploration τ: `15 rounds`")

        st.markdown("---")

        # Scheduler selector for comparison
        st.markdown("**Active Schedulers**")
        selected_schedulers = st.multiselect(
            "Show in plots",
            options=["Static Priority", "Detectability Greedy",
                     "Uncertainty Greedy", "Adaptive Scheduler", "Oracle"],
            default=["Static Priority", "Adaptive Scheduler", "Oracle"],
        )

        st.markdown("---")
        st.markdown(
            "**GitHub:** [rushikesh-D69/water](https://github.com/rushikesh-D69/water)",
            unsafe_allow_html=True
        )

    return panel, selected_schedulers


# =============================================================================
# PANEL 1 — Telescope Queue
# =============================================================================

def panel_telescope_queue(df_ml, obs_df, comp_df):
    st.markdown('<p class="panel-header">Panel 1: Telescope Queue</p>', unsafe_allow_html=True)
    st.caption("Current observation queue — top targets by adaptive scheduler utility")

    if df_ml.empty:
        st.warning("No planet data loaded. Run Stage 1 pipeline first.")
        return

    col1, col2, col3, col4 = st.columns(4)

    # Summary metrics
    n_planets = len(df_ml)
    if "priority_score" in df_ml.columns:
        mean_pri  = df_ml["priority_score"].mean()
        n_high    = int((df_ml["priority_score"] >= 0.5).sum())
    else:
        mean_pri, n_high = 0, 0

    if not obs_df.empty:
        n_obs    = obs_df["planet_name"].nunique()
        mean_unc = obs_df["sigma_after"].mean() if "sigma_after" in obs_df.columns else 0
    else:
        n_obs, mean_unc = 0, 0

    with col1:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{n_planets:,}</p>
            <p class="metric-label">Total Exoplanets</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{mean_pri:.3f}</p>
            <p class="metric-label">Mean Priority Score</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{n_high:,}</p>
            <p class="metric-label">High-Priority (>0.5)</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{n_obs:,}</p>
            <p class="metric-label">Planets Observed</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Top targets table
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Current Telescope Queue (Top 20 Targets)")
        if "priority_score" in df_ml.columns and "detectability" in df_ml.columns:
            top_targets = df_ml.nlargest(20, "priority_score")[
                [c for c in ["pl_name", "hostname", "priority_score", "detectability",
                              "pl_rade", "pl_eqt", "sy_dist", "st_teff"] if c in df_ml.columns]
            ].reset_index(drop=True)
            top_targets.index += 1

            # Color the priority column
            def color_priority(val):
                if val >= 0.7: return "background-color: #22b5a022; color: #22b5a0"
                if val >= 0.5: return "background-color: #f0a50022; color: #f0a500"
                return "background-color: #c9ada722; color: #c9ada7"

            styled = top_targets.style.map(
                color_priority, subset=["priority_score"]
            ).format({
                "priority_score": "{:.4f}",
                "detectability":  "{:.4f}",
                "pl_rade":        "{:.2f}",
                "pl_eqt":         "{:.0f}",
                "sy_dist":        "{:.1f}",
                "st_teff":        "{:.0f}",
            })
            st.dataframe(styled, use_container_width=True, height=500)

    with col_right:
        st.markdown("#### Priority vs Detectability")
        if "priority_score" in df_ml.columns and "detectability" in df_ml.columns:
            sample = df_ml.sample(min(1000, len(df_ml)), random_state=42)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x    = sample["detectability"],
                y    = sample["priority_score"],
                mode = "markers",
                marker = dict(
                    color     = sample["priority_score"],
                    colorscale = "plasma",
                    size      = 4,
                    opacity   = 0.7,
                    showscale = True,
                    colorbar  = dict(title="Priority", thickness=10),
                ),
                text = sample.get("pl_name", pd.Series([""] * len(sample))),
                hovertemplate = "<b>%{text}</b><br>Priority: %{y:.3f}<br>Detectability: %{x:.3f}",
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                title = "Scheduling Domain: Detectability vs Priority",
                xaxis_title = "Detectability D",
                yaxis_title = "Priority Score",
                height = 500,
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PANEL 2 — Priority Evolution
# =============================================================================

def panel_priority_evolution(df_ml, obs_df, selected_schedulers, all_logs):
    st.markdown('<p class="panel-header">Panel 2: Priority Evolution</p>', unsafe_allow_html=True)
    st.caption("How mean priority and uncertainty of selected targets evolve over 30 rounds")

    if not all_logs:
        st.warning("Run Stage 2 notebook in Colab first to generate scheduler logs.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Mean Priority of Selected Targets per Round",
            "Mean Prediction Uncertainty per Round",
            "Cumulative Scientific Gain",
            "Exploration Weight Decay (Adaptive Scheduler)",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for name in selected_schedulers:
        if name not in all_logs:
            continue
        logs  = all_logs[name]
        color = SCHED_COLORS.get(name, COLORS["white"])
        lw    = 3 if name == "Adaptive Scheduler" else 1.5
        dash  = "solid" if name in ("Adaptive Scheduler", "Oracle") else "dash"

        if "mean_priority" in logs.columns:
            fig.add_trace(go.Scatter(x=logs["round"], y=logs["mean_priority"],
                name=name, line=dict(color=color, width=lw, dash=dash),
                showlegend=True), row=1, col=1)

        if "mean_sigma_before" in logs.columns:
            fig.add_trace(go.Scatter(x=logs["round"], y=logs["mean_sigma_before"],
                name=name, line=dict(color=color, width=lw, dash=dash),
                showlegend=False), row=1, col=2)

        if "cum_sci_gain" in logs.columns:
            fig.add_trace(go.Scatter(x=logs["round"], y=logs["cum_sci_gain"],
                name=name, line=dict(color=color, width=lw, dash=dash),
                showlegend=False), row=2, col=1)

    # Weight decay curve
    rounds  = np.arange(1, 31)
    beta_t  = 0.30 * np.exp(-rounds / 15.0)
    alpha_t = np.full_like(rounds, 0.50, dtype=float)
    gamma   = np.full_like(rounds, 0.20, dtype=float)
    total   = alpha_t + beta_t + gamma
    fig.add_trace(go.Scatter(x=rounds, y=alpha_t/total, name="α (uncertainty)",
        line=dict(color=COLORS["teal"], width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=rounds, y=beta_t/total, name="β_t (priority, decays)",
        line=dict(color=COLORS["gold"], width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=rounds, y=gamma/total, name="γ (detectability)",
        line=dict(color=COLORS["pink"], width=2)), row=2, col=2)
    fig.add_vline(x=15, line_dash="dot", line_color=COLORS["muted"], row=2, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height    = 700,
        title     = "Stage 2: Scheduler Performance Evolution (30 Rounds)",
        hovermode = "x unified",
    )
    for i in range(1, 5):
        row, col = (1 if i <= 2 else 2), (i if i <= 2 else i - 2)
        fig.update_xaxes(gridcolor="#30363d", row=row, col=col)
        fig.update_yaxes(gridcolor="#30363d", row=row, col=col)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PANEL 3 — Uncertainty Heatmap
# =============================================================================

def panel_uncertainty_heatmap(df_ml, obs_df):
    st.markdown('<p class="panel-header">Panel 3: Uncertainty Heatmap</p>', unsafe_allow_html=True)
    st.caption("Which planets still need observation — darker = higher uncertainty (more valuable to observe)")

    if df_ml.empty:
        st.warning("No planet data loaded.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        # Scatter: T_eq vs Radius, coloured by priority, sized by uncertainty
        plot_df = df_ml[
            df_ml["pl_eqt"].notna() & df_ml["pl_rade"].notna()
            & df_ml["priority_score"].notna()
        ].copy()

        # Add observed flag
        if not obs_df.empty and "planet_name" in obs_df.columns:
            observed_names = set(obs_df["planet_name"].unique())
            plot_df["observed"] = plot_df["pl_name"].isin(observed_names).astype(int)
        else:
            plot_df["observed"] = 0

        # Clip for readability
        plot_df = plot_df[
            (plot_df["pl_eqt"] >= 100) & (plot_df["pl_eqt"] <= 2000) &
            (plot_df["pl_rade"] >= 0.3) & (plot_df["pl_rade"] <= 10)
        ]

        sample = plot_df.sample(min(2000, len(plot_df)), random_state=42)

        fig = go.Figure()

        # Unobserved
        unobs = sample[sample["observed"] == 0]
        fig.add_trace(go.Scatter(
            x    = unobs["pl_eqt"],
            y    = unobs["pl_rade"],
            mode = "markers",
            name = "Unobserved",
            marker = dict(
                color      = unobs["priority_score"],
                colorscale = "plasma",
                size       = 5,
                opacity    = 0.65,
                showscale  = True,
                colorbar   = dict(title="Priority", thickness=10),
            ),
            text = unobs.get("pl_name", pd.Series([""] * len(unobs))),
            hovertemplate = "<b>%{text}</b><br>T_eq: %{x:.0f}K<br>R: %{y:.2f}R⊕<br>Priority: %{marker.color:.3f}",
        ))

        # Observed (highlighted)
        obs_sample = sample[sample["observed"] == 1]
        if len(obs_sample) > 0:
            fig.add_trace(go.Scatter(
                x    = obs_sample["pl_eqt"],
                y    = obs_sample["pl_rade"],
                mode = "markers",
                name = "Observed",
                marker = dict(
                    color   = COLORS["teal"],
                    size    = 9,
                    symbol  = "star",
                    opacity = 0.9,
                    line    = dict(color=COLORS["white"], width=0.5),
                ),
                text = obs_sample.get("pl_name", pd.Series([""] * len(obs_sample))),
                hovertemplate = "<b>%{text}</b> [OBSERVED]<br>T_eq: %{x:.0f}K<br>R: %{y:.2f}R⊕",
            ))

        # Reference lines
        fig.add_vline(x=288, line_dash="dot", line_color=COLORS["gold"],
                      annotation_text="288 K (Earth)", annotation_font_color=COLORS["gold"])
        fig.add_hline(y=2.0, line_dash="dot", line_color=COLORS["pink"],
                      annotation_text="2.0 R⊕", annotation_font_color=COLORS["pink"])

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title       = "Uncertainty Heatmap: T_eq vs Radius (stars = already observed)",
            xaxis_title = "Equilibrium Temperature T_eq [K]",
            yaxis_title = "Planet Radius [R⊕]",
            height      = 550,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Observation Stats")
        if not obs_df.empty:
            n_unique = obs_df["planet_name"].nunique() if "planet_name" in obs_df.columns else 0
            n_rounds = obs_df["round"].nunique() if "round" in obs_df.columns else 0

            st.metric("Planets Observed", f"{n_unique:,}")
            st.metric("Rounds Completed", f"{n_rounds}")

            if "sigma_reduction" in obs_df.columns:
                total_red = obs_df["sigma_reduction"].sum()
                mean_red  = obs_df["sigma_reduction"].mean()
                st.metric("Total σ Reduced", f"{total_red:.3f}")
                st.metric("Mean σ/observation", f"{mean_red:.4f}")

            if "weather" in obs_df.columns:
                good_weather = (obs_df["weather"] >= 0.65).mean()
                st.metric("Good Weather Rounds", f"{good_weather*100:.1f}%")
        else:
            st.info("Run Stage 2 notebook to populate observation history.")


# =============================================================================
# PANEL 4 — Campaign Timeline
# =============================================================================

def panel_campaign_timeline(obs_df):
    st.markdown('<p class="panel-header">Panel 4: Campaign Timeline</p>', unsafe_allow_html=True)
    st.caption("Gantt-style observation schedule — each row is one observation")

    if obs_df.empty:
        st.warning("No observation history. Run Stage 2 notebook in Colab first.")
        return

    # Filter selector
    max_rounds = int(obs_df["round"].max()) if "round" in obs_df.columns else 30
    round_range = st.slider("Round range", 1, max_rounds, (1, min(15, max_rounds)))
    filtered    = obs_df[obs_df["round"].between(*round_range)]

    if filtered.empty:
        st.info("No observations in selected round range.")
        return

    # Gantt chart via horizontal bar
    fig = go.Figure()

    unique_planets = filtered["planet_name"].unique() if "planet_name" in filtered.columns else []
    planet_idx_map = {p: i for i, p in enumerate(unique_planets)}

    for _, row in filtered.iterrows():
        planet = row.get("planet_name", "Unknown")
        rnd    = int(row.get("round", 1))
        cost   = float(row.get("cost_hrs", 1.0))
        mu_a   = float(row.get("mu_after", 0.5))
        sigma_b = float(row.get("sigma_before", 0.15))
        weather = float(row.get("weather", 0.7))

        color = (COLORS["teal"] if mu_a >= 0.7 else
                 COLORS["gold"] if mu_a >= 0.5 else COLORS["pink"])
        opacity = 0.5 + 0.5 * weather

        p_idx = planet_idx_map.get(planet, 0)
        fig.add_trace(go.Bar(
            x           = [cost],
            y           = [planet],
            orientation = "h",
            base        = [(rnd - 1) * 8],
            marker      = dict(color=color, opacity=opacity, line=dict(width=0)),
            name        = f"Round {rnd}",
            showlegend  = False,
            hovertemplate = (
                f"<b>{planet}</b><br>"
                f"Round: {rnd}<br>"
                f"Cost: {cost:.2f} hrs<br>"
                f"Priority after: {mu_a:.3f}<br>"
                f"σ before: {sigma_b:.3f}<br>"
                f"Weather: {weather:.2f}"
                "<extra></extra>"
            ),
        ))

    # Color legend
    for label, color in [("High priority (>0.7)", COLORS["teal"]),
                          ("Medium (0.5-0.7)",     COLORS["gold"]),
                          ("Lower (<0.5)",          COLORS["pink"])]:
        fig.add_trace(go.Bar(x=[0], y=[""], orientation="h",
            marker_color=color, name=label, showlegend=True))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title       = f"Observation Timeline: Rounds {round_range[0]}–{round_range[1]}",
        xaxis_title = "Cumulative Telescope Time [hrs]",
        yaxis_title = "Planet",
        barmode     = "stack",
        height      = max(400, min(len(unique_planets) * 20 + 100, 800)),
        legend      = dict(x=1.01, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table for selected range
    st.markdown("#### Observation Log")
    cols_show = [c for c in ["round", "planet_name", "mu_before", "sigma_before",
                               "mu_after", "sigma_after", "weather",
                               "snr_effective", "cost_hrs"] if c in filtered.columns]
    st.dataframe(filtered[cols_show].round(4), use_container_width=True, height=300)


# =============================================================================
# PANEL 5 — Metrics Dashboard
# =============================================================================

def panel_metrics_dashboard(comp_df, all_logs, selected_schedulers):
    st.markdown('<p class="panel-header">Panel 5: Metrics Dashboard</p>', unsafe_allow_html=True)
    st.caption("Full scheduler comparison across all 7 metrics")

    if comp_df.empty and not all_logs:
        st.warning("No Stage 2 results. Run stage2_pipeline.ipynb in Colab first.")
        return

    # Comparison table
    if not comp_df.empty:
        st.markdown("#### Scheduler Comparison Table")
        def highlight_best(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            if "Composite Score" in df.columns:
                best_idx = df["Composite Score"].astype(float).idxmax()
                styles.loc[best_idx, "Composite Score"] = "background-color: #22b5a033; color: #22b5a0; font-weight: bold"
            if "Cum. Sci. Gain" in df.columns:
                best_idx = df["Cum. Sci. Gain"].astype(float).idxmax()
                styles.loc[best_idx, "Cum. Sci. Gain"] = "background-color: #22b5a033; color: #22b5a0; font-weight: bold"
            if "Regret vs Oracle" in df.columns:
                best_idx = df["Regret vs Oracle"].astype(float).idxmin()
                styles.loc[best_idx, "Regret vs Oracle"] = "background-color: #22b5a033; color: #22b5a0; font-weight: bold"
            if "Diversity Score" in df.columns:
                best_idx = df["Diversity Score"].astype(float).idxmax()
                styles.loc[best_idx, "Diversity Score"] = "background-color: #f0a50033; color: #f0a500; font-weight: bold"
            if "Priority Coverage" in df.columns:
                best_idx = df["Priority Coverage"].astype(float).idxmax()
                styles.loc[best_idx, "Priority Coverage"] = "background-color: #f0a50033; color: #f0a500; font-weight: bold"
            return styles
        st.dataframe(comp_df.style.apply(highlight_best, axis=None), use_container_width=True)

    if not all_logs:
        return

    st.markdown("---")

    # Cumulative gain comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cumulative Scientific Gain")
        fig = go.Figure()
        for name in selected_schedulers:
            if name not in all_logs:
                continue
            logs  = all_logs[name]
            color = SCHED_COLORS.get(name, COLORS["white"])
            lw    = 3 if name == "Adaptive Scheduler" else 1.5
            dash  = "solid" if name in ("Adaptive Scheduler", "Oracle") else "dash"
            if "cum_sci_gain" in logs.columns:
                final_val = logs["cum_sci_gain"].iloc[-1]
                fig.add_trace(go.Scatter(
                    x=logs["round"], y=logs["cum_sci_gain"],
                    name=f"{name} ({final_val:.3f})",
                    line=dict(color=color, width=lw, dash=dash),
                ))
        fig.update_layout(**PLOTLY_LAYOUT, title="", xaxis_title="Round",
                          yaxis_title="Cum. Sci. Gain", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Regret vs Oracle")
        if "Oracle" in all_logs and "cum_sci_gain" in all_logs["Oracle"].columns:
            oracle_gain = all_logs["Oracle"]["cum_sci_gain"].iloc[-1]
            fig = go.Figure()
            for name in selected_schedulers:
                if name == "Oracle" or name not in all_logs:
                    continue
                logs  = all_logs[name]
                color = SCHED_COLORS.get(name, COLORS["white"])
                lw    = 3 if name == "Adaptive Scheduler" else 1.5
                dash  = "solid" if name == "Adaptive Scheduler" else "dash"
                if "cum_sci_gain" in logs.columns:
                    regret = (oracle_gain - logs["cum_sci_gain"].clip(upper=oracle_gain)) / (oracle_gain + 1e-8)
                    fig.add_trace(go.Scatter(
                        x=logs["round"], y=regret.clip(0, 1),
                        name=name,
                        line=dict(color=color, width=lw, dash=dash),
                    ))
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS["white"],
                          annotation_text="Oracle (0 regret)")
            fig.update_layout(**PLOTLY_LAYOUT, title="", xaxis_title="Round",
                              yaxis_title="Regret vs Oracle", yaxis_range=[-0.05, 1.0],
                              height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Oracle data not available.")

    # Plots from disk
    st.markdown("---")
    st.markdown("#### Generated Analysis Plots")

    plots_dir = ROOT / "plots"
    s2_plots  = sorted(plots_dir.glob("s2_*.png")) if plots_dir.exists() else []

    if s2_plots:
        cols = st.columns(2)
        for i, p in enumerate(s2_plots):
            cols[i % 2].image(str(p), caption=p.stem.replace("s2_", "").replace("_", " ").title(),
                              use_column_width=True)
    else:
        st.info("No Stage 2 plots yet. Run the Stage 2 notebook to generate them.")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size:2rem; font-weight:700; background:linear-gradient(90deg,#22b5a0,#7bccf6);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Adaptive Exoplanet Observation Scheduler
        </h1>
        <p style="color:#8b949e; margin-top:-0.5rem;">
            Stage 2 · 5 Schedulers · 30 Rounds · Oracle Upper Bound · Campaign Diversity
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df_ml              = load_planet_data()
    all_logs, obs_df, comp_df, final_df = load_stage2_results()

    # Sidebar
    panel, selected_schedulers = render_sidebar()

    # Route to panel
    if panel == "Telescope Queue":
        panel_telescope_queue(df_ml, obs_df, comp_df)

    elif panel == "Priority Evolution":
        panel_priority_evolution(df_ml, obs_df, selected_schedulers, all_logs)

    elif panel == "Uncertainty Heatmap":
        panel_uncertainty_heatmap(df_ml, obs_df)

    elif panel == "Campaign Timeline":
        panel_campaign_timeline(obs_df)

    elif panel == "Metrics Dashboard":
        panel_metrics_dashboard(comp_df, all_logs, selected_schedulers)


if __name__ == "__main__":
    main()
