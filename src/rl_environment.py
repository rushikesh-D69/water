"""
rl_environment.py — Stage 3
============================
Gymnasium Environment for Exoplanet Observation Scheduling

ExoplanetSchedulingEnv wraps the existing Stage 2 simulation stack
(ObservationSimulator + ObservationConstraintEngine) into a standard
gymnasium.Env interface for use with Stable-Baselines3.

Design decisions (incorporating review feedback):
  - Action Masking:      EARLY implementation via action_masks() for MaskablePPO.
                         Prevents impossible/useless actions; dramatically improves
                         convergence and exploration quality.
  - Reward Normalization: Each G/D/E/P component normalized INDEPENDENTLY via
                         RunningMeanStd. Prevents any single term from dominating.
  - Smooth Diversity:    EMA-smoothed stellar-type entropy increment rather than
                         noisy raw diff. Stable across early training.
  - Candidate Shortlist: top-N planets by composite score (priority × σ × D).
                         Reduces action space from 5,522 → ≤100.
  - Curriculum Support:  n_candidates parameter scales difficulty:
                           Level 1: 10 planets  (fast convergence / debug)
                           Level 2: 50 planets  (intermediate)
                           Level 3: 100 planets (full)
  - Episode Structure:   300 steps = 30 rounds × up to 10 slots per round.
                         Each step = one planet selection by the agent.
  - Discount:            γ = 0.98 (long-horizon campaign planning).

Reward decomposition:
  R_t = 0.35·G_norm + 0.25·D_norm + 0.20·E_norm + 0.20·P_norm
        - 0.50 (over-budget)
        - 0.20 (redundant re-obs)
        - 0.10 (very low detectability, D < 0.1)
        - 0.30 (action not visible / truly invalid)
"""

import copy
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Tuple, Dict, List

import gymnasium as gym
from gymnasium import spaces

from src.observation_simulator import ObservationSimulator
from src.constraint_engine import ObservationConstraintEngine

# ── Shortlist scoring ─────────────────────────────────────────────────────────
_SHORTLIST_ALPHA = 0.40   # weight for priority score
_SHORTLIST_SIGMA = 0.35   # weight for uncertainty
_SHORTLIST_DET   = 0.25   # weight for detectability


def make_shortlist(
    df:          pd.DataFrame,
    mu_pred:     np.ndarray,
    sigma_pred:  np.ndarray,
    n_candidates: int = 100,
) -> np.ndarray:
    """
    Pre-filter the full planet catalogue to the top-N candidates.

    Shortlist score = 0.40 × priority + 0.35 × uncertainty + 0.25 × detectability

    This single call is done ONCE at environment construction. The RL agent
    only ever sees and acts on this reduced candidate set.

    Parameters
    ----------
    df           : full processed planet DataFrame
    mu_pred      : predicted priority scores (n_planets,)
    sigma_pred   : prediction uncertainties  (n_planets,)
    n_candidates : desired shortlist size

    Returns
    -------
    shortlist_indices : np.ndarray of shape (n_candidates,)
        Global indices into df / mu_pred / sigma_pred.
    """
    mu   = np.asarray(mu_pred,    dtype=np.float64).ravel()
    sig  = np.asarray(sigma_pred, dtype=np.float64).ravel()
    det  = np.asarray(df["detectability"].fillna(0.1).values, dtype=np.float64)

    # Normalise each dimension to [0, 1] before scoring
    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-8)

    score = (
        _SHORTLIST_ALPHA * _norm(mu)
        + _SHORTLIST_SIGMA * _norm(sig)
        + _SHORTLIST_DET  * _norm(det)
    )
    n = min(n_candidates, len(df))
    return np.argsort(score)[::-1][:n].astype(np.int32)


# ── Running Mean / Std for per-component reward normalisation ─────────────────
class RunningMeanStd:
    """
    Welford's online algorithm for running mean and variance.
    Used to normalise each reward component independently so no single
    dimension can dominate PPO's value function learning.
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean  = x.mean(axis=0) if x.ndim > 0 else x
        batch_var   = x.var(axis=0)  if x.ndim > 0 else 0.0
        batch_count = x.shape[0]     if x.ndim > 0 else 1

        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot_count
        m_a         = self.var   * self.count
        m_b         = batch_var  * batch_count
        m_2         = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var    = m_2 / tot_count
        self.count  = tot_count

    def normalize(self, x: float, clip: float = 5.0) -> float:
        z = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return float(np.clip(z, -clip, clip))


# ── Stellar-type diversity tracker ────────────────────────────────────────────
_STELLAR_TYPES = ["O", "B", "A", "F", "G", "K", "M"]
_EMA_ALPHA     = 0.15   # EMA smoothing for entropy increment


class DiversityTracker:
    """
    Tracks the stellar-type distribution of observed planets and computes
    a smoothed diversity reward using EMA to avoid noisy spikes in early
    training (addresses review feedback Issue 4).
    """

    def __init__(self, df: pd.DataFrame, shortlist_idx: np.ndarray):
        # One-hot spectral type columns in df
        self._stype_cols = [c for c in df.columns if c.startswith("st_spectype_")]
        self._n_types    = len(self._stype_cols)
        # Pre-extract spectral type probabilities for shortlist
        if self._stype_cols:
            self._stype_mat = df.iloc[shortlist_idx][self._stype_cols].values.astype(np.float64)
        else:
            # Fallback: uniform distribution if no spectral type columns
            self._stype_mat = np.ones((len(shortlist_idx), 1), dtype=np.float64)
            self._n_types   = 1

        self._counts  = np.zeros(self._n_types, dtype=np.float64)
        self._ema_ent = 0.0   # EMA-smoothed entropy

    def reset(self):
        self._counts[:]  = 0.0
        self._ema_ent    = 0.0

    def _entropy(self) -> float:
        total = self._counts.sum()
        if total < 1e-8:
            return 0.0
        p = self._counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-9)))

    def update(self, local_idx: int) -> float:
        """
        Add planet `local_idx` (index into shortlist) to observed set,
        and return the EMA-smoothed diversity reward.
        """
        ent_before = self._entropy()

        # Add stellar type contribution
        if local_idx < len(self._stype_mat):
            self._counts += self._stype_mat[local_idx]
        else:
            self._counts[0] += 1.0

        ent_after  = self._entropy()
        raw_delta  = max(ent_after - ent_before, 0.0)   # clipped to non-negative

        # EMA smoothing: reduces noise without losing the signal
        self._ema_ent = (1 - _EMA_ALPHA) * self._ema_ent + _EMA_ALPHA * raw_delta
        return self._ema_ent

    def get_coverage_vector(self) -> np.ndarray:
        """Return normalized stellar-type count vector (diversity state)."""
        total = self._counts.sum()
        if total > 0:
            return self._counts / total
        return self._counts.copy()


# ── Main Gymnasium Environment ────────────────────────────────────────────────
class ExoplanetSchedulingEnv(gym.Env):
    """
    Gymnasium environment for adaptive exoplanet observation scheduling.

    State (flat Box):
        For each of the N_CANDS candidates (6 features each):
            - priority_mu        : current estimated priority
            - uncertainty_sigma  : current prediction uncertainty
            - detectability      : physical detectability score
            - cost_norm          : normalised observation cost
            - observed_flag      : 1.0 if already observed this episode
            - visibility_flag    : 1.0 if currently visible this round
        Global scalars (10):
            - weather            : current weather quality [0, 1]
            - budget_remaining   : remaining hours / max_hours [0, 1]
            - round_frac         : round / n_rounds [0, 1]
            - slot_frac          : slot_in_round / k_per_round [0, 1]
            - diversity[0..5]    : normalized stellar-type coverage (6 dims)
        Total: 6 × N_CANDS + 10

    Action: Discrete(N_CANDS) — index into shortlist.

    Action Masking (via action_masks()):
        Invalid if: not visible | budget insufficient | already observed.

    Reward:
        R_t = 0.35·G_norm + 0.25·D_norm + 0.20·E_norm + 0.20·P_norm + penalties

    Episode terminates when all 30 rounds complete.
    """

    metadata = {"render_modes": []}

    # Penalty constants
    _PENALTY_OVER_BUDGET   = -0.50
    _PENALTY_REDUNDANT     = -0.20
    _PENALTY_LOW_DET       = -0.10
    _PENALTY_INVALID       = -0.30

    # Reward weights (mirror Stage 2 Composite Score)
    _W_GAIN      = 0.35
    _W_DIVERSITY = 0.25
    _W_EFFICIENCY= 0.20
    _W_PRIORITY  = 0.20

    def __init__(
        self,
        df:              pd.DataFrame,
        mu_pred:         np.ndarray,
        sigma_pred:      np.ndarray,
        true_priorities: np.ndarray,
        n_candidates:    int  = 100,
        n_rounds:        int  = 30,
        k_per_round:     int  = 10,
        seed:            int  = 42,
    ):
        """
        Parameters
        ----------
        df              : full processed planet DataFrame
        mu_pred         : Stage 1 predicted priority scores (n_planets,)
        sigma_pred      : Stage 1 prediction uncertainties  (n_planets,)
        true_priorities : ground-truth priority scores      (n_planets,)
        n_candidates    : shortlist size (curriculum: 10 / 50 / 100)
        n_rounds        : number of observation rounds per campaign
        k_per_round     : max planet slots per round
        seed            : random seed
        """
        super().__init__()

        self._df              = df.reset_index(drop=True)
        self._mu_pred         = np.asarray(mu_pred,         dtype=np.float64).ravel()
        self._sigma_pred      = np.asarray(sigma_pred,      dtype=np.float64).ravel()
        self._true_priorities = np.asarray(true_priorities, dtype=np.float64).ravel()
        self._n_rounds        = n_rounds
        self._k_per_round     = k_per_round
        self._seed            = seed
        self._n_candidates    = n_candidates

        # Build shortlist (done ONCE at construction)
        self._shortlist = make_shortlist(df, mu_pred, sigma_pred, n_candidates)
        N = len(self._shortlist)

        # Precompute static shortlist features
        det_full  = df["detectability"].fillna(0.1).values.astype(np.float64)
        cost_full = self._precompute_costs()
        self._det_sl   = det_full[self._shortlist]
        self._cost_sl  = cost_full[self._shortlist]
        cost_max       = self._cost_sl.max() + 1e-8
        self._cost_norm_sl = self._cost_sl / cost_max

        # ── Spaces ────────────────────────────────────────────────────────────
        obs_dim = 6 * N + 10   # 6 features per candidate + 10 global scalars
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N)

        # ── Running normalizers (one per reward component) ───────────────────
        self._rms_g = RunningMeanStd()
        self._rms_d = RunningMeanStd()
        self._rms_e = RunningMeanStd()
        self._rms_p = RunningMeanStd()

        # ── Diversity tracker ─────────────────────────────────────────────────
        self._diversity = DiversityTracker(self._df, self._shortlist)

        # ── Internal episode state (initialised in reset()) ──────────────────
        self._sim:          Optional[ObservationSimulator]      = None
        self._ce:           Optional[ObservationConstraintEngine] = None
        self._observed_set: set = set()
        self._current_round: int = 0
        self._slot_in_round: int = 0
        self._episode_done:  bool = False

        # Reward decomposition log (accessible after each episode)
        self.reward_log: List[dict] = []

        self._np_random = np.random.default_rng(seed)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        ep_seed = int(self._np_random.integers(0, 2**31))

        self._sim = ObservationSimulator(
            df             = self._df,
            initial_means  = self._mu_pred.copy(),
            initial_sigmas = self._sigma_pred.copy(),
            seed           = ep_seed,
        )
        self._ce = ObservationConstraintEngine(self._df, seed=ep_seed + 1)
        self._ce.new_round()   # advance to round 1

        self._observed_set  = set()
        self._current_round = 1
        self._slot_in_round = 0
        self._episode_done  = False
        self._diversity.reset()
        self.reward_log     = []

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one planet-selection step.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        N       = len(self._shortlist)
        action  = int(action)
        reward  = 0.0
        info    = {}

        global_idx = int(self._shortlist[action]) if action < N else -1
        mask       = self.action_masks()

        # ── Validity guard ────────────────────────────────────────────────────
        if not mask[action]:
            reward += self._PENALTY_INVALID
            info["invalid"] = True
        else:
            # ── Redundant observation penalty ──────────────────────────────
            if global_idx in self._observed_set:
                reward += self._PENALTY_REDUNDANT
                info["redundant"] = True

            # ── Low detectability penalty ──────────────────────────────────
            det_i = float(self._det_sl[action])
            if det_i < 0.1:
                reward += self._PENALTY_LOW_DET

            # ── Observe planet ─────────────────────────────────────────────
            can_obs, cost = self._ce.can_observe(
                global_idx, n_already_scheduled=self._slot_in_round
            )
            if not can_obs:
                reward += self._PENALTY_OVER_BUDGET
                info["over_budget"] = True
            else:
                sigma_before = float(self._sim.sigma[global_idx])

                self._ce.consume_budget(global_idx, self._slot_in_round)
                rec = self._sim.observe(
                    planet_idx   = global_idx,
                    round_number = self._current_round,
                    weather      = self._ce.weather,
                    cost_hrs     = cost,
                )

                sigma_after  = float(self._sim.sigma[global_idx])
                self._observed_set.add(global_idx)
                self._slot_in_round += 1

                # ── Reward computation ─────────────────────────────────────
                r_g, r_d, r_e, r_p = self._compute_reward_components(
                    action, global_idx, sigma_before, sigma_after, cost
                )
                # Normalise each component independently
                g_n = self._rms_g.normalize(r_g)
                d_n = self._rms_d.normalize(r_d)
                e_n = self._rms_e.normalize(r_e)
                p_n = self._rms_p.normalize(r_p)

                reward += (
                    self._W_GAIN       * g_n
                    + self._W_DIVERSITY  * d_n
                    + self._W_EFFICIENCY * e_n
                    + self._W_PRIORITY   * p_n
                )

                # Update running stats for future normalisation
                self._rms_g.update(np.array([r_g]))
                self._rms_d.update(np.array([r_d]))
                self._rms_e.update(np.array([r_e]))
                self._rms_p.update(np.array([r_p]))

                info.update({"r_g": r_g, "r_d": r_d, "r_e": r_e, "r_p": r_p,
                             "r_total": reward, "round": self._current_round})
                self.reward_log.append(info.copy())

        # ── Advance round when slot budget exhausted ──────────────────────────
        round_done = (
            self._slot_in_round >= self._k_per_round
            or self._ce.time_budget < self._cost_sl.min()
        )
        if round_done:
            self._current_round += 1
            self._slot_in_round  = 0
            if self._current_round <= self._n_rounds:
                self._ce.new_round()

        terminated = self._current_round > self._n_rounds
        return self._get_obs(), float(reward), terminated, False, info

    def action_masks(self) -> np.ndarray:
        """
        Return boolean array of shape (N_CANDS,) indicating valid actions.

        An action is VALID iff:
          1. The planet is currently visible (visibility > 0)
          2. Enough time budget remains to observe it
          3. (Soft) Not already observed — we allow re-obs but penalize it
        """
        N    = len(self._shortlist)
        mask = np.zeros(N, dtype=bool)

        # visibility for the full dataset
        vis = self._ce.visibility if hasattr(self._ce, "visibility") else np.ones(len(self._df))

        for local_i, global_i in enumerate(self._shortlist):
            if vis[global_i] <= 0:
                continue
            can_obs, _ = self._ce.can_observe(global_i, self._slot_in_round)
            if can_obs:
                mask[local_i] = True

        # If everything masked (rare edge case), unmask top-priority for stability
        if not mask.any():
            best = int(np.argmax(self._sim.mu[self._shortlist]))
            mask[best] = True

        return mask

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        N     = len(self._shortlist)
        sl    = self._shortlist
        vis   = self._ce.visibility if hasattr(self._ce, "visibility") else np.ones(len(self._df))

        # Per-candidate features (6 × N)
        priority    = np.clip(self._sim.mu[sl],    0, 1).astype(np.float32)
        uncertainty = np.clip(self._sim.sigma[sl], 0, 1).astype(np.float32)
        detect      = self._det_sl.astype(np.float32)
        cost_n      = self._cost_norm_sl.astype(np.float32)
        obs_flag    = np.array([1.0 if int(i) in self._observed_set else 0.0 for i in sl], dtype=np.float32)
        vis_flag    = np.clip(vis[sl], 0, 1).astype(np.float32)

        candidate_feats = np.concatenate([
            priority, uncertainty, detect, cost_n, obs_flag, vis_flag
        ])   # shape: (6*N,)

        # Global scalars (10)
        weather_val  = float(np.clip(self._ce.weather, 0, 1))
        budget_frac  = float(np.clip(self._ce.time_budget / 8.0, 0, 1))
        round_frac   = float((self._current_round - 1) / self._n_rounds)
        slot_frac    = float(self._slot_in_round / self._k_per_round)
        div_vec      = self._diversity.get_coverage_vector()[:6]   # up to 6 stellar types
        # Pad to exactly 6 values
        div_pad      = np.zeros(6, dtype=np.float32)
        div_pad[:len(div_vec)] = div_vec.astype(np.float32)

        global_feats = np.array([weather_val, budget_frac, round_frac, slot_frac], dtype=np.float32)
        obs = np.concatenate([candidate_feats, global_feats, div_pad])   # (6*N + 10,)
        return obs

    # ── Reward components ─────────────────────────────────────────────────────

    def _compute_reward_components(
        self,
        local_idx:    int,
        global_idx:   int,
        sigma_before: float,
        sigma_after:  float,
        cost:         float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute raw (un-normalized) reward components.
        Each will be independently normalized before weighting.

        Returns
        -------
        r_g : information gain  (sigma reduction × detectability)
        r_d : diversity reward  (EMA-smoothed entropy increment)
        r_e : efficiency reward (gain / cost)
        r_p : priority coverage (true priority of observed planet)
        """
        det = float(self._det_sl[local_idx])

        # G — information gain
        r_g = max(sigma_before - sigma_after, 0.0) * det

        # D — EMA-smoothed diversity
        r_d = self._diversity.update(local_idx)

        # E — observation efficiency
        r_e = r_g / (cost + 1e-6)

        # P — true priority coverage (ground truth, not noisy estimate)
        r_p = float(np.clip(self._true_priorities[global_idx], 0, 1))

        return r_g, r_d, r_e, r_p

    # ── Static cost precomputation ────────────────────────────────────────────

    def _precompute_costs(self) -> np.ndarray:
        """Mirror ObservationConstraintEngine._compute_static_costs logic."""
        df   = self._df
        jmag = df["sy_jmag"].fillna(df["sy_vmag"].fillna(11.0)).values
        dist = df["sy_dist"].fillna(100.0).values
        tdur = df["pl_trandur"].fillna(2.0).values.clip(0.5, 12.0)

        BASE = 0.5
        MAX  = 4.0
        mag_cost  = np.clip((jmag - 8.0) * 0.15, 0, 2.0)
        dist_cost = np.clip(np.log1p(dist / 50.0) * 0.3, 0, 1.5)
        tdur_cost = tdur / 24.0
        raw       = BASE + mag_cost + dist_cost + tdur_cost
        return np.clip(raw, BASE, MAX)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_shortlist_info(self) -> pd.DataFrame:
        """Return DataFrame describing the current shortlist (for debugging)."""
        sl = self._shortlist
        return pd.DataFrame({
            "global_idx":   sl,
            "pl_name":      self._df["pl_name"].iloc[sl].values if "pl_name" in self._df else sl,
            "priority_mu":  self._mu_pred[sl],
            "sigma":        self._sigma_pred[sl],
            "detectability": self._det_sl,
            "cost_hrs":     self._cost_sl,
        })

    def get_campaign_summary(self) -> dict:
        """Return episode summary statistics (call after episode ends)."""
        if self._sim is None:
            return {}
        unc_summary = self._sim.uncertainty_summary()
        total_gain  = sum(
            r.get("r_g", 0.0) for r in self.reward_log
        )
        return {
            "n_observed":      unc_summary["n_observed"],
            "total_gain":      total_gain,
            "n_rounds":        self._current_round - 1,
            "reward_log_len":  len(self.reward_log),
        }

    def render(self):
        pass   # no-op; visualization handled by rl_evaluation.py
