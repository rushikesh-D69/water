"""
rl_evaluation.py — Stage 3
============================
RL-Specific Evaluation Metrics + Visualization Suite

7 Publication-Quality Plots:
  1. s3_training_reward_curve.png  — episode reward + running mean vs timesteps
  2. s3_policy_heatmap.png         — T_eq vs radius, colored by RL selection freq
  3. s3_reward_decomposition.png   — stacked area G/D/E/P components per round
  4. s3_exploration_timeline.png   — new vs revisit targets per round (stacked bar)
  5. s3_pareto_extended.png        — Stage 2 Pareto + RL Agent + LinUCB nodes
  6. s3_generalization.png         — robustness bars across 4 weather seeds
  7. s3_state_tsne.png             — t-SNE of state embeddings by reward quartile

All plots use the established dark-mode theme from evaluation.py.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Theme (consistent with evaluation.py) ────────────────────────────────────
DARK_BG = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#22b5a0"
GOLD    = "#f0a500"
PINK    = "#c9ada7"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
BLUE    = "#7bccf6"
PURPLE  = "#bb86fc"
GREEN   = "#39d353"

SCHEDULER_COLORS = {
    "Static Priority":       GOLD,
    "Detectability Greedy":  PINK,
    "Uncertainty Greedy":    BLUE,
    "Adaptive Scheduler":    ACCENT,
    "Oracle":                "#ffffff",
    "PPO RL Agent":          PURPLE,
    "LinUCB Bandit":         GREEN,
}

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _fig_setup(figsize=(12, 6), title=""):
    fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=14, fontweight="bold", y=0.98)
    return fig


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
    if title:  ax.set_title(title,  color=TEXT, fontsize=11, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT, fontsize=10)


def _running_mean(arr: np.ndarray, window: int = 20) -> np.ndarray:
    out = np.convolve(arr, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, out])


# =============================================================================
# 1. Training Reward Curve
# =============================================================================

def plot_training_reward_curve(
    episode_rewards: List[float],
    window:          int  = 20,
    save:            bool = True,
) -> plt.Figure:
    """
    Episode reward vs training episode, with running mean overlay.
    Clearly shows convergence and stability of learning.
    """
    fig = _fig_setup((12, 5), "Stage 3: PPO Training Reward Curve")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "Episode Reward vs Training Episode",
              "Episode", "Total Campaign Reward")

    eps = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards, dtype=float)
    rm      = _running_mean(rewards, window)

    ax.fill_between(eps, rewards, alpha=0.15, color=PURPLE)
    ax.plot(eps, rewards, color=PURPLE, alpha=0.4, linewidth=0.8, label="Episode reward")
    ax.plot(eps, rm, color=ACCENT, linewidth=2.0, label=f"Running mean (w={window})")

    # Annotate final mean
    final_mean = float(np.nanmean(rm[-window:]))
    ax.axhline(final_mean, color=GOLD, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(len(eps) * 0.02, final_mean * 1.02, f"Final mean: {final_mean:.3f}",
            color=GOLD, fontsize=9)

    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9, loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_training_reward_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 2. Policy Heatmap
# =============================================================================

def plot_policy_heatmap(
    env,
    model,
    df:      pd.DataFrame,
    n_eval:  int  = 20,
    save:    bool = True,
) -> plt.Figure:
    """
    2D heatmap of equilibrium temperature (T_eq) vs planetary radius (R_p),
    colored by how frequently the RL policy selects each planet.

    Reveals what parameter-space regions the policy prefers and whether
    it avoids the easy gas-giant dominated mini-Neptune trap.
    """
    from src.rl_environment import ExoplanetSchedulingEnv
    try:
        from sb3_contrib import MaskablePPO as _MaskPPO
        _masking = True
    except ImportError:
        _masking = False

    sl     = env._shortlist
    sl_df  = df.iloc[sl].reset_index(drop=True)
    n_sl   = len(sl)
    counts = np.zeros(n_sl, dtype=float)

    print(f"[Heatmap] Sampling policy over {n_eval} episodes …")
    for ep in range(n_eval):
        obs, _ = env.reset()
        done   = False
        while not done:
            mask = env.action_masks()
            if _masking:
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=False)
            action = int(action)
            if action < n_sl:
                counts[action] += 1
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                break

    freq = counts / (counts.sum() + 1e-8)

    # Extract T_eq and radius for shortlist
    teq    = sl_df["pl_eqt"].fillna(300.0).values    if "pl_eqt"   in sl_df.columns else np.random.uniform(200, 3000, n_sl)
    radius = sl_df["pl_rade"].fillna(1.0).values     if "pl_rade"  in sl_df.columns else np.random.uniform(0.5, 15, n_sl)

    fig = _fig_setup((11, 7), "Stage 3: RL Policy Selection Heatmap")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "RL Policy Preference Space (T_eq vs Radius)",
              "Equilibrium Temperature [K]", "Planet Radius [R⊕]")

    sc = ax.scatter(teq, radius, c=freq, cmap="plasma",
                    s=60 + 400 * freq / (freq.max() + 1e-8),
                    alpha=0.85, edgecolors="none", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Selection Frequency", color=TEXT, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

    # HZ band
    ax.axvspan(200, 400, alpha=0.07, color=GREEN, label="HZ range (approx)")
    ax.axhline(1.6, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.6, label="Rocky/gas boundary")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)
    ax.set_xlim(0, max(teq.max() * 1.05, 3500))
    ax.set_ylim(0, min(radius.max() * 1.05, 25))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_policy_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 3. Reward Decomposition Plot
# =============================================================================

def plot_reward_decomposition(
    reward_log: List[dict],
    save:       bool = True,
) -> plt.Figure:
    """
    Stacked area chart showing per-round G / D / E / P reward components.
    Reveals what the RL agent is optimizing and how balance shifts over rounds.
    """
    if not reward_log:
        print("[Decomp] No reward log available — skipping plot.")
        return None

    df_log = pd.DataFrame(reward_log)
    rounds = sorted(df_log["round"].unique())

    g_vals = [df_log[df_log["round"] == r]["r_g"].mean() for r in rounds]
    d_vals = [df_log[df_log["round"] == r]["r_d"].mean() for r in rounds]
    e_vals = [df_log[df_log["round"] == r]["r_e"].mean() for r in rounds]
    p_vals = [df_log[df_log["round"] == r]["r_p"].mean() for r in rounds]

    rounds  = np.array(rounds)
    g_arr   = np.clip(g_vals, 0, None)
    d_arr   = np.clip(d_vals, 0, None)
    e_arr   = np.clip(e_vals, 0, None)
    p_arr   = np.clip(p_vals, 0, None)

    fig = _fig_setup((12, 6), "Stage 3: RL Reward Component Decomposition")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "Mean Reward Components per Round (G / D / E / P)",
              "Round", "Mean Reward Component")

    ax.stackplot(rounds, g_arr, d_arr, e_arr, p_arr,
                 labels=["Gain (G, 35%)", "Diversity (D, 25%)",
                         "Efficiency (E, 20%)", "Priority (P, 20%)"],
                 colors=[ACCENT, GREEN, GOLD, PINK], alpha=0.75)

    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_reward_decomposition.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 4. Exploration vs Exploitation Timeline
# =============================================================================

def plot_exploration_exploitation_timeline(
    env,
    model,
    n_eval:  int  = 5,
    save:    bool = True,
) -> plt.Figure:
    """
    Stacked bar chart per round: new planets (exploration) vs revisits (exploitation).
    Directly reveals how the RL policy's exploration-exploitation balance evolves.
    """
    try:
        from sb3_contrib import MaskablePPO as _MaskPPO
        _masking = True
    except ImportError:
        _masking = False

    n_rounds       = env._n_rounds
    explore_counts = np.zeros(n_rounds, dtype=float)
    exploit_counts = np.zeros(n_rounds, dtype=float)
    sl             = env._shortlist

    for ep in range(n_eval):
        obs, _        = env.reset()
        done          = False
        observed_this = set()

        while not done:
            mask = env.action_masks()
            if _masking:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            round_idx = min(env._current_round - 1, n_rounds - 1)

            global_i = int(sl[action]) if action < len(sl) else -1
            if global_i in observed_this:
                exploit_counts[round_idx] += 1
            else:
                explore_counts[round_idx] += 1
                observed_this.add(global_i)

            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                break

    explore_counts /= n_eval
    exploit_counts /= n_eval
    rounds = np.arange(1, n_rounds + 1)

    fig = _fig_setup((14, 6), "Stage 3: RL Exploration vs Exploitation Timeline")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "Policy Behavior: Exploration (new) vs Exploitation (revisit)",
              "Round", "Mean Targets Selected")

    ax.bar(rounds, explore_counts, color=ACCENT,  alpha=0.85, label="Exploration (new planets)")
    ax.bar(rounds, exploit_counts, color=GOLD,    alpha=0.75, bottom=explore_counts,
           label="Exploitation (revisit)")

    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)
    ax.set_xticks(rounds[::3])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_exploration_timeline.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 5. Extended Pareto Frontier (+ RL nodes)
# =============================================================================

def plot_extended_pareto(
    results_dict: Dict,      # name → {"cum_gain", "diversity", "efficiency"}
    save: bool = True,
) -> plt.Figure:
    """
    Re-plots the Stage 2 Pareto frontier in Gain × Diversity space and adds
    the PPO RL Agent and LinUCB Bandit nodes. Shows whether RL pushes closer
    to or beyond the existing Pareto frontier.
    """
    names     = list(results_dict.keys())
    gains     = np.array([results_dict[n]["cum_gain"]   for n in names])
    divs      = np.array([results_dict[n]["diversity"]  for n in names])
    effs      = np.array([results_dict[n]["efficiency"] for n in names])
    effs_norm = (effs - effs.min()) / (effs.max() - effs.min() + 1e-8)

    # Identify Pareto frontier (maximize both gain and diversity)
    pareto_mask = np.zeros(len(names), dtype=bool)
    for i in range(len(names)):
        dominated = False
        for j in range(len(names)):
            if i != j and gains[j] >= gains[i] and divs[j] >= divs[i]:
                if gains[j] > gains[i] or divs[j] > divs[i]:
                    dominated = True
                    break
        pareto_mask[i] = not dominated
    pareto_order = np.argsort(gains[pareto_mask])
    pareto_g     = gains[pareto_mask][pareto_order]
    pareto_d     = divs[pareto_mask][pareto_order]

    fig = _fig_setup((11, 8), "Stage 3: Extended Pareto Frontier — All Schedulers")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "Gain vs. Diversity — Pareto Frontier Analysis",
              "Cumulative Scientific Gain", "Campaign Diversity Score")

    # Frontier curve
    ax.plot(pareto_g, pareto_d, color=ACCENT, linewidth=1.5, linestyle="--",
            alpha=0.6, label="Pareto frontier", zorder=1)

    # Plot all schedulers
    for i, name in enumerate(names):
        c    = SCHEDULER_COLORS.get(name, MUTED)
        size = 150 + 300 * effs_norm[i]
        marker = "*" if "RL" in name or "Bandit" in name else "o"
        zorder = 5 if "RL" in name or "Bandit" in name else 3
        ax.scatter(gains[i], divs[i], c=c, s=size, marker=marker,
                   edgecolors="white", linewidths=0.8,
                   zorder=zorder, label=name)
        ax.annotate(name, (gains[i], divs[i]),
                    textcoords="offset points", xytext=(6, 5),
                    color=c, fontsize=8)

    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MUTED,
               markersize=8, label="Stage 2 heuristics"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=PURPLE,
               markersize=12, label="RL Agents (Stage 3)"),
    ]
    ax.legend(handles=size_legend + [
        Line2D([0], [0], color=ACCENT, linewidth=1.5, linestyle="--", label="Pareto frontier")
    ], facecolor=PANEL, labelcolor=TEXT, fontsize=8, loc="lower right")

    ax.text(0.02, 0.98, "Marker size ∝ Obs. Efficiency",
            transform=ax.transAxes, color=MUTED, fontsize=8, va="top")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_pareto_extended.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 6. Generalization / Robustness
# =============================================================================

def plot_generalization(
    gen_df: pd.DataFrame,     # from evaluate_generalization()
    save:   bool = True,
) -> plt.Figure:
    """
    Bar chart of mean policy reward across different weather seeds.
    Demonstrates that the policy generalizes across environmental conditions.
    """
    fig = _fig_setup((10, 5), "Stage 3: RL Policy Generalization Across Weather Conditions")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "Robustness Test — Mean Reward vs Weather Seed",
              "Weather Seed", "Mean Episode Reward")

    seeds = gen_df["weather_seed"].astype(str).tolist()
    means = gen_df["mean_reward"].values
    stds  = gen_df["std_reward"].values

    bars = ax.bar(seeds, means, color=PURPLE, alpha=0.85,
                  width=0.5, edgecolor=PANEL, linewidth=0.5)
    ax.errorbar(seeds, means, yerr=stds, fmt="none",
                ecolor=TEXT, elinewidth=1.5, capsize=5)

    overall_mean = float(np.mean(means))
    ax.axhline(overall_mean, color=ACCENT, linewidth=1.5, linestyle="--",
               label=f"Overall mean: {overall_mean:.3f}")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[list(means).index(m)] * 0.1,
                f"{m:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9)

    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_generalization.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 7. State t-SNE Embedding
# =============================================================================

def plot_state_tsne(
    state_buffer:  np.ndarray,    # (N_steps, obs_dim)
    reward_buffer: np.ndarray,    # (N_steps,)
    n_components:  int  = 2,
    perplexity:    int  = 30,
    max_samples:   int  = 2000,
    save:          bool = True,
) -> plt.Figure:
    """
    t-SNE visualization of RL state embeddings, colored by reward quartile.
    Reveals distinct behavioral clusters: high/low reward states, exploration
    vs exploitation regions in observation space.
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[t-SNE] sklearn not found — skipping. Install: pip install scikit-learn")
        return None

    if len(state_buffer) > max_samples:
        idx           = np.random.choice(len(state_buffer), max_samples, replace=False)
        state_buffer  = state_buffer[idx]
        reward_buffer = reward_buffer[idx]

    print(f"[t-SNE] Fitting t-SNE on {len(state_buffer)} state vectors …")
    scaler     = StandardScaler()
    states_sc  = scaler.fit_transform(state_buffer)

    tsne       = TSNE(n_components=n_components, perplexity=perplexity,
                      random_state=42, n_iter=500, verbose=0)
    emb        = tsne.fit_transform(states_sc)

    # Color by reward quartile
    q25, q50, q75 = np.percentile(reward_buffer, [25, 50, 75])
    colors_arr    = np.where(reward_buffer >= q75, 0,
                    np.where(reward_buffer >= q50, 1,
                    np.where(reward_buffer >= q25, 2, 3)))
    quartile_colors = [GREEN, ACCENT, GOLD, PINK]
    quartile_labels = ["Top 25%", "50–75%", "25–50%", "Bottom 25%"]

    fig = _fig_setup((11, 8), "Stage 3: RL State Space t-SNE — Reward Clusters")
    ax  = fig.add_subplot(111)
    _ax_style(ax, "t-SNE State Embedding (colored by reward quartile)",
              "t-SNE Dim 1", "t-SNE Dim 2")

    for q, (c, lbl) in enumerate(zip(quartile_colors, quartile_labels)):
        mask = colors_arr == q
        ax.scatter(emb[mask, 0], emb[mask, 1], c=c, s=8, alpha=0.55,
                   label=lbl, linewidths=0)

    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9, markerscale=3,
              loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = PLOTS_DIR / "s3_state_tsne.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Plot] Saved → {path}")
    return fig


# =============================================================================
# 8. Policy Explainability Report
# =============================================================================

def print_policy_explanation(explanations: List[dict]):
    """
    Pretty-print the perturbation-based policy explanations for top-k targets.
    Meant for notebook display or terminal output.
    """
    print("\n" + "═" * 60)
    print("  RL POLICY DECISION EXPLANATION")
    print("═" * 60)
    for rank, exp in enumerate(explanations, 1):
        print(f"\n  #{rank}  Target: {exp['target_name']}")
        print(f"       Selection probability: {exp['action_prob']:.4f}")
        print("       Reasoning:")
        for line in exp["reasoning"][:5]:
            print(f"         {line}")
    print("\n" + "═" * 60)


# =============================================================================
# 9. Full Stage 3 Comparison Table
# =============================================================================

def build_stage3_comparison_table(
    stage2_results: dict,   # from stage2_comparison.csv or evaluation.py
    rl_results:     dict,   # {"PPO RL Agent": {...}, "LinUCB Bandit": {...}}
    oracle_gain:    float,
    oracle_div:     float,
    oracle_eff:     float,
    oracle_pri:     float,
) -> pd.DataFrame:
    """
    Combine Stage 2 heuristic results with Stage 3 RL results into a unified
    comparison table using the same Composite Score formula.
    """
    all_results = {**stage2_results, **rl_results}
    rows = []
    for name, res in all_results.items():
        g_n = min(res.get("cum_gain",  0) / (oracle_gain + 1e-8), 1.0)
        d_n = min(res.get("diversity", 0) / (oracle_div  + 1e-8), 1.0)
        e_n = min(res.get("efficiency",0) / (oracle_eff  + 1e-8), 1.0)
        p_n = min(res.get("priority",  0) / (oracle_pri  + 1e-8), 1.0)
        composite = 0.35 * g_n + 0.25 * d_n + 0.20 * e_n + 0.20 * p_n

        rows.append({
            "Scheduler":         name,
            "Composite Score %": round(composite * 100, 2),
            "Cum. Gain":         round(res.get("cum_gain",   0.0), 4),
            "Diversity":         round(res.get("diversity",  0.0), 4),
            "Efficiency":        round(res.get("efficiency", 0.0), 4),
            "Priority Coverage": round(res.get("priority",   0.0), 4),
            "n_observed":        res.get("n_observed", 0),
        })

    df = pd.DataFrame(rows).sort_values("Composite Score %", ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df


def save_stage3_comparison(df: pd.DataFrame):
    """Save the comparison table to CSV."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    path     = data_dir / "stage3_comparison.csv"
    df.to_csv(path, index=True, index_label="Rank")
    print(f"[Eval] Saved → {path}")
    return path


# =============================================================================
# 10. Training Curve from SB3 Monitor Logs
# =============================================================================

def plot_training_from_monitor(
    log_dir: Path,
    window:  int  = 20,
    save:    bool = True,
) -> plt.Figure:
    """
    Parse SB3 Monitor CSV logs and plot episode reward + length curves.
    Useful when training runs asynchronously in Colab.
    """
    monitor_files = list(log_dir.glob("*.monitor.csv"))
    if not monitor_files:
        print("[Monitor] No monitor CSV found — skipping.")
        return None

    dfs = []
    for f in monitor_files:
        df = pd.read_csv(f, skiprows=1)
        dfs.append(df)
    log = pd.concat(dfs).sort_values("t").reset_index(drop=True)

    rewards = log["r"].values if "r" in log.columns else log.iloc[:, 0].values

    return plot_training_reward_curve(rewards, window=window, save=save)
