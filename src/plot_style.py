"""
IEEE-print plotting theme.

White background, black text, colorblind-safe hues plus distinct
linestyles/markers so figures remain readable in grayscale print.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PAPER_BG = "white"
PANEL = "white"
TEXT = "#111111"
MUTED = "#444444"
SPINE = "#333333"
GRID = "#cccccc"
DPI = 300

# Wong 2011 colorblind-safe palette
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GOLD = "#E69F00"
SKY = "#56B4E9"
BLACK = "#000000"

ACCENT = BLUE
PINK = VERM
COLORS = [BLUE, VERM, GREEN, PURPLE]

SCHEDULER_STYLES = {
    "Adaptive Scheduler":   dict(color=BLUE,   ls="-",  marker="o", lw=2.4, ms=4.5),
    "Detectability Greedy": dict(color=VERM,   ls="--", marker="s", lw=1.7, ms=4.0),
    "Static Priority":      dict(color=GREEN,  ls="-.", marker="^", lw=1.7, ms=4.5),
    "Uncertainty Greedy":   dict(color=PURPLE, ls=":",  marker="D", lw=1.7, ms=4.0),
    "Oracle":               dict(color=BLACK,  ls="-",  marker="*", lw=2.0, ms=7.0),
}
SCHEDULER_COLORS = {k: v["color"] for k, v in SCHEDULER_STYLES.items()}

MODEL_STYLES = {
    "Random Forest":      dict(color=BLUE,   marker="o"),
    "XGBoost":            dict(color=VERM,   marker="s"),
    "Gradient Boosting":  dict(color=GREEN,  marker="^"),
    "LightGBM":           dict(color=PURPLE, marker="D"),
}


def apply_rc():
    plt.rcParams.update({
        "figure.facecolor": PAPER_BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linestyle": ":",
        "legend.facecolor": PAPER_BG,
        "legend.edgecolor": SPINE,
        "legend.labelcolor": TEXT,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": DPI,
        "savefig.facecolor": PAPER_BG,
        "savefig.edgecolor": "none",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": PANEL,
        "figure.facecolor": PAPER_BG,
        "grid.color": GRID,
        "axes.edgecolor": SPINE,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    })
    plt.rcParams.update({
        "savefig.dpi": DPI,
        "savefig.facecolor": PAPER_BG,
        "savefig.edgecolor": "none",
        "figure.facecolor": PAPER_BG,
        "axes.facecolor": PANEL,
    })


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=TEXT)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    if title:
        ax.set_title(title, color=TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT)


def legend(ax, **kwargs):
    kwargs.setdefault("facecolor", PAPER_BG)
    kwargs.setdefault("edgecolor", SPINE)
    kwargs.setdefault("framealpha", 1.0)
    kwargs.setdefault("labelcolor", TEXT)
    return ax.legend(**kwargs)


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=PAPER_BG, edgecolor="none")
    plt.close(fig)
    print(f"[Plot] Saved -> {path}")


IEEE_FIGURES = (
    "predicted_vs_actual.png",
    "feature_importance.png",
    "s2_weather_sequence.png",
    "s2_weight_decay.png",
    "s2_cumulative_gain.png",
    "s2_pareto_frontier.png",
)


def export_ieee_figures(plots_dir=None, ieee_dir=None, names=None):
    """Copy paper figures from plots/ into IEEE/ for pdfLaTeX."""
    import shutil
    root = Path(__file__).resolve().parent.parent
    plots_dir = Path(plots_dir or root / "plots")
    ieee_dir = Path(ieee_dir or root / "IEEE")
    if not ieee_dir.exists():
        return
    for fn in (names or IEEE_FIGURES):
        src = plots_dir / fn
        if src.exists():
            shutil.copy2(src, ieee_dir / fn)
            print(f"[Plot] IEEE copy -> {ieee_dir / fn}")


apply_rc()
