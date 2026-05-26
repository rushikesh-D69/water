"""
data_acquisition.py  — v2.0
============================
Stage 1: Exoplanet Probabilistic Prioritization Pipeline

KEY DESIGN DECISIONS (v2.0):
  - No data leakage: HZ boundaries and ESI are used ONLY for constructing the
    target label (priority_score) via weak supervision. They are NEVER passed
    as ML features.
  - ML features contain ONLY raw astrophysical + observational parameters.
  - Target is a CONTINUOUS priority score [0,1], not binary. This is a ranking
    problem, not a classification problem.
  - Observation feasibility is modelled explicitly (transit depth, SNR proxy,
    distance penalty) and included as ML features.
  - Uncertainty is estimated post-prediction via ensemble variance.

Scientific formulations:
  - Kopparapu et al. (2013, 2014) Habitable Zone boundaries  [LABEL ONLY]
  - Chen & Kipping (2017) empirical mass-radius imputation
  - Earth Similarity Index (Schulze-Makuch et al. 2011)      [LABEL ONLY]
  - Scientific Gain = uncertainty * detectability
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---- Paths -------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

RAW_CSV       = DATA_DIR / "exoplanets_raw.csv"
PROCESSED_CSV = DATA_DIR / "exoplanets_processed.csv"

# ---- NASA TAP config ---------------------------------------------------------
NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

COLUMNS = [
    "pl_name", "hostname", "sy_snum", "sy_pnum",
    "disc_year", "discoverymethod",
    # Planetary physics
    "pl_rade", "pl_bmasse", "pl_dens",
    "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_orbincl",
    "pl_eqt", "pl_insol",
    # Transit observability
    "pl_trandep", "pl_trandur",
    # Stellar physics
    "st_teff", "st_rad", "st_mass", "st_lum",
    "st_met", "st_logg", "st_age",
    # Distance / brightness
    "sy_dist", "sy_vmag", "sy_jmag",
]

ADQL_QUERY = f"""
SELECT {', '.join(COLUMNS)}
FROM pscomppars
WHERE pl_rade IS NOT NULL OR pl_bmasse IS NOT NULL
ORDER BY pl_name
"""

# Theme constants for plots
DARK_BG = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#22b5a0"
GOLD    = "#f0a500"
PINK    = "#c9ada7"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"


# =============================================================================
# 1. Data Download
# =============================================================================

def fetch_nasa_exoplanets(force_refresh=False):
    """Download confirmed exoplanet composite data from NASA TAP API."""
    if RAW_CSV.exists() and not force_refresh:
        print(f"[Data] Loading cached dataset from: {RAW_CSV}")
        return pd.read_csv(RAW_CSV, comment="#")

    print("[Data] Fetching data from NASA Exoplanet Archive TAP ...")
    params = {"query": ADQL_QUERY, "format": "csv", "lang": "ADQL"}
    t0 = time.time()
    resp = requests.get(NASA_TAP_URL, params=params, timeout=120)
    resp.raise_for_status()
    elapsed = time.time() - t0

    with open(RAW_CSV, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(RAW_CSV, comment="#")
    print(f"[Data] Downloaded {len(df):,} records in {elapsed:.1f}s -> {RAW_CSV}")
    return df


# =============================================================================
# 2. Imputation: Mass-Radius Relations (Chen & Kipping 2017)
# =============================================================================

def impute_mass_radius(df):
    """Fill missing planet mass or radius with empirical power-law relations."""
    df = df.copy()

    # Radius -> Mass
    mask_no_mass = df["pl_bmasse"].isna() & df["pl_rade"].notna()
    rp = df.loc[mask_no_mass, "pl_rade"]
    df.loc[mask_no_mass, "pl_bmasse"] = np.where(
        rp <= 1.23, 0.9718 * rp ** 3.58,
        np.where(rp <= 14.26, 1.436 * rp ** 1.70, np.nan)
    )
    print(f"[Impute] Filled {mask_no_mass.sum():,} missing planet masses from radius.")

    # Mass -> Radius (inverse relations)
    mask_no_rad = df["pl_rade"].isna() & df["pl_bmasse"].notna()
    mp = df.loc[mask_no_rad, "pl_bmasse"]
    boundary_mass = 0.9718 * (1.23 ** 3.58)
    df.loc[mask_no_rad, "pl_rade"] = np.where(
        mp <= boundary_mass,
        (mp / 0.9718) ** (1 / 3.58),
        np.where(mp <= 1.436 * (14.26 ** 1.70), (mp / 1.436) ** (1 / 1.70), np.nan)
    )
    print(f"[Impute] Filled {mask_no_rad.sum():,} missing planet radii from mass.")
    return df


# =============================================================================
# 3. Derived Physical Features (for ML input)
# =============================================================================

def compute_equilibrium_temperature(df):
    """Derive T_eq from Stefan-Boltzmann where missing. Bond albedo A_B = 0.3."""
    df = df.copy()
    A_B = 0.3
    mask = (
        df["pl_eqt"].isna() & df["st_teff"].notna()
        & df["st_rad"].notna() & df["pl_orbsmax"].notna()
    )
    st_r_m  = df.loc[mask, "st_rad"] * 6.957e8     # Solar radii -> metres
    sma_m   = df.loc[mask, "pl_orbsmax"] * 1.496e11  # AU -> metres
    df.loc[mask, "pl_eqt"] = (
        df.loc[mask, "st_teff"]
        * np.sqrt(st_r_m / (2 * sma_m))
        * (1 - A_B) ** 0.25
    )
    print(f"[Derive] Computed {mask.sum():,} missing equilibrium temperatures.")
    return df


def compute_planet_density(df):
    """Compute planet density [g/cm^3] from mass and radius where missing."""
    df = df.copy()
    mask = df["pl_dens"].isna() & df["pl_bmasse"].notna() & df["pl_rade"].notna()
    M_e, R_e = 5.972e27, 6.371e8  # g, cm
    m_g  = df.loc[mask, "pl_bmasse"] * M_e
    r_cm = df.loc[mask, "pl_rade"] * R_e
    df.loc[mask, "pl_dens"] = m_g / ((4 / 3) * np.pi * r_cm ** 3)
    print(f"[Derive] Computed {mask.sum():,} missing planet densities.")
    return df


def compute_stellar_luminosity(df):
    """Derive linear stellar luminosity [L_sun] from log luminosity or T/R."""
    df = df.copy()
    df["st_lum_linear"] = np.where(
        df["st_lum"].notna(),
        10 ** df["st_lum"],
        np.where(
            df["st_teff"].notna() & df["st_rad"].notna(),
            (df["st_rad"] ** 2) * ((df["st_teff"] / 5778) ** 4),
            np.nan,
        ),
    )
    return df


# =============================================================================
# 4. Observation Feasibility Features (for ML input)
# =============================================================================

def compute_observation_features(df):
    """
    Build telescope-side observability features.
    These are REAL ML inputs — not leakage — because they relate to
    telescope scheduling cost/benefit, not to the label construction.

    Features added:
      snr_proxy       : proxy for Signal-to-Noise Ratio of atmospheric detection
      distance_penalty: penalty for far targets (harder spectroscopy)
      transit_depth_norm: normalised transit depth (detectability)
      obs_duration_norm : normalised transit duration
      detectability   : composite observability score [0,1]
    """
    df = df.copy()

    # ---- SNR proxy -----------------------------------------------------------
    # SNR ~ transit_depth * sqrt(T_transit) / V_mag_factor
    # Using J-band for cooler stars where water signatures appear
    mag = df["sy_jmag"].fillna(df["sy_vmag"].fillna(12.0))
    td  = df["pl_trandep"].fillna(0.0).clip(lower=0)   # transit depth [%]
    dur = df["pl_trandur"].fillna(1.0).clip(lower=0.1)  # duration [hours]

    # Flux factor (relative to mag=10 star)
    flux_factor = 10 ** ((10.0 - mag) / 2.5)
    df["snr_proxy"] = td * np.sqrt(dur) * flux_factor

    # ---- Distance penalty ----------------------------------------------------
    # Spectroscopic follow-up becomes exponentially harder beyond ~100 pc
    dist = df["sy_dist"].fillna(df["sy_dist"].median()).clip(lower=0.1)
    df["distance_penalty"] = np.exp(-dist / 100.0)  # 1.0 = nearby, -> 0 = far

    # ---- Normalised transit observables --------------------------------------
    df["transit_depth_norm"] = td / (td.max() + 1e-10)
    df["obs_duration_norm"]  = dur / (dur.max() + 1e-10)

    # ---- Composite detectability score [0,1] ---------------------------------
    snr_norm = df["snr_proxy"] / (df["snr_proxy"].quantile(0.99) + 1e-10)
    snr_norm = snr_norm.clip(0, 1)
    df["detectability"] = 0.5 * snr_norm + 0.3 * df["distance_penalty"] + \
                          0.2 * df["transit_depth_norm"]
    df["detectability"] = df["detectability"].clip(0, 1)

    print(f"[Obs]   Observation feasibility features computed.")
    return df


# =============================================================================
# 5. Spectral Class Encoding (for ML input)
# =============================================================================

def encode_stellar_features(df):
    """Derive spectral class from T_eff and one-hot encode it."""
    df = df.copy()

    def teff_to_spectral(t):
        if pd.isna(t):   return "Unknown"
        if t >= 30000:   return "O"
        elif t >= 10000: return "B"
        elif t >= 7500:  return "A"
        elif t >= 6000:  return "F"
        elif t >= 5200:  return "G"
        elif t >= 3700:  return "K"
        else:            return "M"

    df["spectral_class"] = df["st_teff"].apply(teff_to_spectral)
    for cls in ["O", "B", "A", "F", "G", "K", "M", "Unknown"]:
        df[f"star_{cls}"] = (df["spectral_class"] == cls).astype(int)

    print("[Encode] Spectral class one-hot encoded.")
    return df


# =============================================================================
# 6. HZ Boundaries & ESI  —  LABEL CONSTRUCTION ONLY (never ML features)
# =============================================================================

# Kopparapu et al. 2014 coefficients
HZ_COEFFICIENTS = {
    "RV": (1.7763,  1.4335e-4,  3.3954e-9, -7.6364e-12, -1.1950e-15),
    "RG": (1.0385,  1.2456e-4,  1.4612e-8, -7.6345e-12, -1.7511e-15),
    "MG": (0.3507,  5.9578e-5,  1.6707e-9, -3.0058e-12, -5.1925e-16),
    "EM": (0.3207,  5.4471e-5,  1.5275e-9, -2.1709e-12, -3.8282e-16),
}


def _kopparapu_flux(T_star, limit):
    S0, a, b, c, d = HZ_COEFFICIENTS[limit]
    T = np.clip(T_star, 2600, 7200) - 5780.0
    return S0 + a*T + b*T**2 + c*T**3 + d*T**4


def _hz_au(L_star, S_eff):
    return np.sqrt(np.maximum(L_star, 1e-10) / np.maximum(S_eff, 1e-10))


def _compute_hz_factor(df):
    """
    Internal helper: compute continuous HZ factor [0,1] for label building.
    NOT exposed as an ML feature.
    """
    mask = (
        df["st_teff"].notna() & df["st_lum_linear"].notna()
        & df["pl_orbsmax"].notna()
    )
    hz_factor = np.zeros(len(df))

    T_arr = df["st_teff"].values
    L_arr = df["st_lum_linear"].values
    a_arr = df["pl_orbsmax"].values

    for i in range(len(df)):
        if not mask.iloc[i]:
            continue
        T, L, a = T_arr[i], L_arr[i], a_arr[i]
        if np.isnan(T) or np.isnan(L) or np.isnan(a):
            continue
        T_c = np.clip(T, 2600, 7200)
        d_rv = _hz_au(L, _kopparapu_flux(T_c, "RV"))
        d_rg = _hz_au(L, _kopparapu_flux(T_c, "RG"))
        d_mg = _hz_au(L, _kopparapu_flux(T_c, "MG"))
        d_em = _hz_au(L, _kopparapu_flux(T_c, "EM"))

        in_chz = (a >= d_rg) and (a <= d_mg)
        in_ohz = (a >= d_rv) and (a <= d_em)

        if in_chz:
            chz_center   = (d_rg + d_mg) / 2
            chz_half     = max((d_mg - d_rg) / 2, 1e-10)
            hz_factor[i] = 1.0 - 0.5 * abs(a - chz_center) / chz_half
        elif in_ohz:
            ohz_center   = (d_rv + d_em) / 2
            ohz_half     = max((d_em - d_rv) / 2, 1e-10)
            hz_factor[i] = 0.5 * (1.0 - abs(a - ohz_center) / ohz_half)
        else:
            hz_factor[i] = 0.0

    return np.clip(hz_factor, 0.0, 1.0)


def _compute_esi(df):
    """
    Internal helper: compute Earth Similarity Index.
    NOT exposed as an ML feature.
    """
    EARTH_DENSITY = 5.515  # g/cm^3
    rp   = df["pl_rade"].values.astype(float)
    rho  = (df["pl_dens"].fillna(EARTH_DENSITY) / EARTH_DENSITY).values
    teq  = df["pl_eqt"].values.astype(float)

    n = 3
    params = [
        (rp,   1.0,   0.57),
        (rho,  1.0,   1.07),
        (teq,  288.0, 0.70),
    ]
    esi = np.ones(len(df))
    for x, x0, w in params:
        denom = np.where(np.abs(x + x0) < 1e-10, 1e-10, x + x0)
        term  = (1.0 - np.abs((x - x0) / denom)) ** (w / n)
        esi   = np.where(np.isnan(x), np.nan, esi * term)
    return np.clip(esi, 0.0, 1.0)


# =============================================================================
# 7. Priority Score Construction (Weak Supervision Target)
# =============================================================================

def compute_priority_score(df):
    """
    Build the continuous priority score target using weak supervision.

    The score integrates three orthogonal signals via a weighted product
    to ensure all components must be non-zero for a high score:

      priority_score = f_thermal^w1 * f_rocky^w2 * f_esi^w3

    where all factors are derived from KNOWN astrophysical theory and
    are EXCLUDED from the ML feature set (no leakage).

    Also computes:
      detectability   : observation feasibility [0,1]  (IS an ML feature)
      scientific_gain : uncertainty * detectability     (post-prediction)
    """
    df = df.copy()

    # ---- Thermal habitability score (Kopparapu HZ) --------------------------
    f_thermal = _compute_hz_factor(df)
    print(f"[Label] CHZ/OHZ calculation complete.")

    # ---- Physical habitability score (rocky planet proxy) -------------------
    rp  = df["pl_rade"].fillna(np.inf).values
    rho = df["pl_dens"].fillna(0.0).values
    f_rocky = np.where(
        (rp <= 1.8) & (rho >= 3.0), 1.0,
        np.where((rp <= 2.0) & (rho >= 2.0), 0.85,
        np.where(rp <= 2.5, 0.5, 0.1))
    )

    # ---- Earth Similarity Index (Schulze-Makuch 2011) -----------------------
    f_esi = _compute_esi(df)
    f_esi = np.where(np.isnan(f_esi), 0.0, f_esi)

    # ---- Weighted product (multiplicative — all must be high) ---------------
    # Weights chosen to reflect scientific importance:
    #   thermal (HZ): dominant constraint (w=0.5)
    #   rocky:        necessary for surface liquid water (w=0.3)
    #   ESI:          multi-parameter similarity proxy (w=0.2)
    priority_score = (f_thermal ** 0.5) * (f_rocky ** 0.3) * (f_esi ** 0.2)
    df["priority_score"] = np.clip(priority_score, 0.0, 1.0)

    # Store internal components for diagnostics ONLY (not ML features)
    df["_hz_factor_diag"] = f_thermal
    df["_esi_diag"]       = f_esi
    df["_rocky_diag"]     = f_rocky

    n_high = int((df["priority_score"] >= 0.3).sum())
    print(f"[Label] Priority score computed. {n_high:,} planets with score >= 0.3")
    return df


# =============================================================================
# 8. ML Feature Set  (leakage-free)
# =============================================================================

# RAW astrophysical + observability features only.
# NO hz_factor, NO esi, NO hz_rv/rg/mg/em — those are label components.
ML_FEATURES = [
    # ---- Planet physics (raw) ----
    "pl_rade",          # Planet radius [R_earth]
    "pl_bmasse",        # Planet mass [M_earth]
    "pl_dens",          # Planet density [g/cm^3]
    "pl_orbper",        # Orbital period [days]
    "pl_orbsmax",       # Semi-major axis [AU]
    "pl_orbeccen",      # Orbital eccentricity
    "pl_orbincl",       # Orbital inclination [deg]
    "pl_eqt",           # Equilibrium temperature [K]
    "pl_insol",         # Insolation flux [Earth flux]
    # ---- Stellar physics (raw) ----
    "st_teff",          # Stellar effective temperature [K]
    "st_rad",           # Stellar radius [R_sun]
    "st_mass",          # Stellar mass [M_sun]
    "st_lum_linear",    # Stellar luminosity [L_sun]
    "st_met",           # Stellar metallicity [dex]
    "st_logg",          # Surface gravity [log cm/s^2]
    "st_age",           # Stellar age [Gyr]
    # ---- Distance / brightness (raw) ----
    "sy_dist",          # System distance [pc]
    "sy_vmag",          # V-band magnitude
    "sy_jmag",          # J-band magnitude
    # ---- Observation feasibility (engineered but NOT label components) ----
    "snr_proxy",        # Transit SNR proxy
    "distance_penalty", # Spectroscopic accessibility
    "transit_depth_norm",
    "obs_duration_norm",
    "detectability",    # Composite observability [0,1]
    # ---- Stellar type encoding ----
    "star_O", "star_B", "star_A", "star_F", "star_G", "star_K", "star_M",
]

TARGET = "priority_score"   # Continuous [0,1] — regression / ranking task


def build_ml_dataset(df):
    """Filter to ML-ready rows, median-fill remaining NaNs, return clean df."""
    required = ["pl_rade", "st_teff", "pl_orbsmax", "pl_eqt"]
    df_clean = df.dropna(subset=required).copy()

    available = [c for c in ML_FEATURES if c in df_clean.columns]
    missing   = [c for c in ML_FEATURES if c not in df_clean.columns]
    if missing:
        print(f"[Clean] Warning: {len(missing)} features not in dataset: {missing}")

    num_cols = df_clean[available].select_dtypes(include=np.number).columns.tolist()
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

    print(f"[Clean] ML-ready: {len(df_clean):,} planets | {len(available)} features | target: {TARGET}")
    return df_clean, available


# =============================================================================
# 9. Exploratory Visualisations
# =============================================================================

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED)
    if title:  ax.set_title(title, color=TEXT, fontsize=12)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT)


def plot_priority_distribution(df):
    """Plot distribution of the priority score target."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(DARK_BG)

    # Priority score histogram
    ax = axes[0]
    _style_ax(ax, "Priority Score Distribution", "Priority Score (target)", "Count")
    ps = df["priority_score"].dropna()
    n, bins, patches = ax.hist(ps, bins=60, color=ACCENT, alpha=0.85, edgecolor=DARK_BG)
    ax.axvline(0.3, color=GOLD, lw=1.5, ls="--", label="Score >= 0.3 threshold")
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)

    # Radius vs Equilibrium Temperature, coloured by priority score
    ax = axes[1]
    _style_ax(ax, "Radius vs T_eq  (coloured by priority score)", "T_eq [K]", "Radius [R_earth]")
    sub = df[df["pl_eqt"].notna() & df["pl_rade"].notna() & df["priority_score"].notna()]
    sc  = ax.scatter(sub["pl_eqt"], sub["pl_rade"],
                     c=sub["priority_score"], cmap="plasma",
                     s=10, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Priority Score").ax.yaxis.label.set_color(TEXT)
    ax.axvline(288, color=GOLD, lw=1, ls=":", alpha=0.7)
    ax.axhline(2.0, color=PINK, lw=1, ls=":", alpha=0.7)
    ax.set_xlim(0, 2500); ax.set_ylim(0, 20)

    fig.suptitle("Exoplanet Probabilistic Prioritization — Score Overview",
                 fontsize=14, color=TEXT, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "priority_score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


def plot_feature_correlations(df, feature_names):
    """Correlation heatmap of ML features with priority score."""
    cols = feature_names[:18] + ["priority_score"]  # top 18 + target
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                square=True, linewidths=0.4, ax=ax, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Matrix (vs Priority Score)", color=TEXT, fontsize=13, pad=12)
    ax.tick_params(colors=MUTED, labelsize=8)

    out = PLOTS_DIR / "feature_correlations.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


def plot_observability_analysis(df):
    """Plot observation feasibility landscape."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)

    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors=MUTED)

    # SNR proxy distribution
    axes[0].hist(df["snr_proxy"].clip(0, df["snr_proxy"].quantile(0.99)),
                 bins=50, color=ACCENT, alpha=0.85, edgecolor=DARK_BG)
    axes[0].set_title("SNR Proxy Distribution", color=TEXT)
    axes[0].set_xlabel("SNR Proxy", color=TEXT); axes[0].set_ylabel("Count", color=TEXT)

    # Distance penalty vs priority score
    sc = axes[1].scatter(df["sy_dist"].clip(0, 2000), df["priority_score"],
                         c=df["detectability"], cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[1], label="Detectability").ax.yaxis.label.set_color(TEXT)
    axes[1].set_title("Distance vs Priority Score\n(coloured by detectability)", color=TEXT)
    axes[1].set_xlabel("Distance [pc]", color=TEXT); axes[1].set_ylabel("Priority Score", color=TEXT)

    # Detectability vs priority (the ideal target for observation scheduling)
    sc2 = axes[2].scatter(df["detectability"], df["priority_score"],
                          c=df["pl_rade"].clip(0, 20), cmap="plasma", s=10, alpha=0.6)
    plt.colorbar(sc2, ax=axes[2], label="Radius [R_earth]").ax.yaxis.label.set_color(TEXT)
    axes[2].set_xlabel("Detectability", color=TEXT)
    axes[2].set_ylabel("Priority Score", color=TEXT)
    axes[2].set_title("Detectability vs Priority\n(Stage 2 scheduling domain)", color=TEXT)

    plt.tight_layout()
    out = PLOTS_DIR / "observability_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


# =============================================================================
# 10. Main Pipeline
# =============================================================================

def run_pipeline(force_refresh=False):
    """Execute full data acquisition and feature engineering pipeline."""
    print("=" * 60)
    print("  Stage 1 -- Data Acquisition & Feature Engineering v2.0")
    print("=" * 60)

    df = fetch_nasa_exoplanets(force_refresh=force_refresh)
    print(f"\n[Info]  Raw dataset shape: {df.shape}")

    # Step 1: Physical imputation
    df = impute_mass_radius(df)
    df = compute_equilibrium_temperature(df)
    df = compute_planet_density(df)
    df = compute_stellar_luminosity(df)

    # Step 2: Observation feasibility features (ML input)
    df = compute_observation_features(df)

    # Step 3: Spectral encoding (ML input)
    df = encode_stellar_features(df)

    # Step 4: Construct target label via weak supervision (NOT ML features)
    df = compute_priority_score(df)

    # Step 5: Build clean ML-ready dataset
    df_ml, feature_names = build_ml_dataset(df)

    df_ml.to_csv(PROCESSED_CSV, index=False)
    print(f"\n[Save]  Processed dataset -> {PROCESSED_CSV}")

    print("\n[Plots] Generating exploratory visualisations ...")
    plot_priority_distribution(df_ml)
    plot_feature_correlations(df_ml, feature_names)
    plot_observability_analysis(df_ml)

    print("\n[Done]  Pipeline complete.")
    print("=" * 60)
    return df_ml, feature_names


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Force re-download from NASA")
    args = p.parse_args()
    run_pipeline(force_refresh=args.refresh)
