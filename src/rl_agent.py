"""
rl_agent.py — Stage 3
=======================
RL Agents: MaskablePPO + Behavior Cloning + LinUCB Contextual Bandit

Incorporates review feedback:
  - Behavior Cloning:    Pre-trains PPO on AdaptiveScheduler trajectories
                         (imitation → RL fine-tuning hybrid). Massively
                         stabilizes early learning.
  - Curriculum:          Phase 1 (10→10k steps) → Phase 2 (50→50k) → Phase 3 (100→100k+).
                         Start small for debugging, scale when stable.
  - Action Masking:      MaskablePPO from sb3_contrib; env provides action_masks().
  - Policy Explainability: Perturbation-based feature attribution reveals WHY
                            the RL agent selected each target.
  - LinUCB Bandit:       Interpretable contextual bandit for academic comparison.
  - RLScheduler:         Wraps trained PPO inside BaseScheduler for drop-in
                         comparison against all Stage 2 heuristics.
"""

import copy
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Stable-Baselines3 imports (with graceful fallback for missing sb3-contrib) ─
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    _HAS_MASKING = True
except ImportError:
    warnings.warn(
        "sb3-contrib not found. Install with: pip install sb3-contrib\n"
        "Falling back to standard PPO without action masking.",
        ImportWarning
    )
    from stable_baselines3 import PPO as MaskablePPO
    _HAS_MASKING = False

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.rl_environment import ExoplanetSchedulingEnv, make_shortlist
from src.scheduler import AdaptiveScheduler, run_campaign
from src.observation_simulator import ObservationSimulator
from src.constraint_engine import ObservationConstraintEngine

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. Behavior Cloning — Trajectory Collection & Supervised Pretraining
# =============================================================================

def collect_bc_trajectories(
    df:              pd.DataFrame,
    mu_pred:         np.ndarray,
    sigma_pred:      np.ndarray,
    true_priorities: np.ndarray,
    n_candidates:    int  = 100,
    n_episodes:      int  = 50,
    n_rounds:        int  = 30,
    k_per_round:     int  = 10,
    seed:            int  = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (obs, action) pairs by rolling out the AdaptiveScheduler inside
    the Gymnasium environment.

    The AdaptiveScheduler's selections are projected to the shortlist action
    space so they can supervise PPO's policy network.

    Parameters
    ----------
    n_episodes : number of campaigns to collect (more → better BC)

    Returns
    -------
    obs_buf : np.ndarray (total_steps, obs_dim)
    act_buf : np.ndarray (total_steps,)  — shortlist-local action indices
    """
    print(f"[BC] Collecting {n_episodes} AdaptiveScheduler campaigns …")

    obs_list: List[np.ndarray] = []
    act_list: List[int]        = []

    for ep in range(n_episodes):
        env = ExoplanetSchedulingEnv(
            df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
            true_priorities=true_priorities,
            n_candidates=n_candidates, n_rounds=n_rounds,
            k_per_round=k_per_round, seed=seed + ep,
        )
        obs, _ = env.reset()
        shortlist = env._shortlist
        # Build inverse map: global_idx → local action
        global_to_local = {int(g): l for l, g in enumerate(shortlist)}

        # Recreate simulator + CE with same seed for AdaptiveScheduler to drive
        sim = ObservationSimulator(
            df=df, initial_means=mu_pred.copy(),
            initial_sigmas=sigma_pred.copy(), seed=seed + ep,
        )
        ce  = ObservationConstraintEngine(df, seed=seed + ep + 1)
        scheduler = AdaptiveScheduler(df)
        observed_set = set()

        for rnd in range(1, n_rounds + 1):
            ce.new_round()
            selected, costs = scheduler.select(
                simulator=sim, constraint_engine=ce,
                round_number=rnd, k=k_per_round,
                observed_set=observed_set,
            )
            # Step the env for each scheduler-selected planet
            for global_idx in selected:
                # Record obs before step
                obs_list.append(obs.copy())

                # Map to local action (if in shortlist)
                if global_idx in global_to_local:
                    local_action = global_to_local[global_idx]
                else:
                    # If not in shortlist, pick highest-priority valid action
                    mask = env.action_masks()
                    local_action = int(np.argmax(env._sim.mu[shortlist] * mask))

                act_list.append(local_action)

                obs, _, done, _, _ = env.step(local_action)
                if done:
                    break

            # Sync env simulator state with scheduler's simulator
            if selected:
                sim.observe_batch(selected, rnd, ce.weather, np.array(costs))
                observed_set.update(selected)

            if done:
                break

        if (ep + 1) % 10 == 0:
            print(f"[BC]   Episode {ep+1}/{n_episodes} — steps collected: {len(act_list)}")

    obs_buf = np.array(obs_list, dtype=np.float32)
    act_buf = np.array(act_list, dtype=np.int64)
    print(f"[BC] Collected {len(act_buf)} (obs, action) pairs.")
    return obs_buf, act_buf


def pretrain_bc(
    model,
    obs_buf:        np.ndarray,
    act_buf:        np.ndarray,
    n_epochs:       int   = 10,
    batch_size:     int   = 256,
    lr:             float = 3e-4,
    device:         str   = "auto",
) -> List[float]:
    """
    Supervised behavior cloning pretraining for the PPO policy network.

    Loss = CrossEntropy(policy_logits, expert_actions)

    Parameters
    ----------
    model    : MaskablePPO or PPO instance (after calling model.learn(0) to init)
    obs_buf  : expert observations (N, obs_dim)
    act_buf  : expert actions      (N,)

    Returns
    -------
    losses : list of per-epoch mean loss values
    """
    if not _HAS_MASKING and not hasattr(model, "policy"):
        warnings.warn("[BC] Model has no policy — skipping pretraining.")
        return []

    print(f"[BC] Pretraining policy via Behavior Cloning ({n_epochs} epochs) …")

    policy = model.policy
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    policy.to(dev)

    obs_t = torch.FloatTensor(obs_buf).to(dev)
    act_t = torch.LongTensor(act_buf).to(dev)

    dataset    = TensorDataset(obs_t, act_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = optim.Adam(policy.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for obs_b, act_b in dataloader:
            optimizer.zero_grad()
            # Forward pass: get action distribution logits
            dist = policy.get_distribution(obs_b)
            logits = dist.distribution.logits   # (batch, n_actions)
            loss   = criterion(logits, act_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mean_loss = epoch_loss / len(dataloader)
        losses.append(mean_loss)
        print(f"[BC]   Epoch {epoch+1:2d}/{n_epochs} — loss: {mean_loss:.4f}")

    print("[BC] Pretraining complete.")
    return losses


# =============================================================================
# 2. Reward Logging Callback
# =============================================================================

class RewardDecompositionCallback(BaseCallback):
    """
    Logs per-step reward components (G, D, E, P) from env info dict.
    Accessible via `callback.reward_component_log` after training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reward_component_log: List[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "r_g" in info:
                self.reward_component_log.append({
                    "r_g": info["r_g"],
                    "r_d": info["r_d"],
                    "r_e": info["r_e"],
                    "r_p": info["r_p"],
                    "r_total": info["r_total"],
                    "round": info.get("round", 0),
                })
        return True


# =============================================================================
# 3. PPO Training — Curriculum Protocol
# =============================================================================

def _make_masked_env(env_kwargs: dict) -> gym.Env:
    """Factory to wrap env with ActionMasker if sb3-contrib is available."""
    env = ExoplanetSchedulingEnv(**env_kwargs)
    if _HAS_MASKING:
        env = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)


import gymnasium as gym


def train_ppo_curriculum(
    df:                pd.DataFrame,
    mu_pred:           np.ndarray,
    sigma_pred:        np.ndarray,
    true_priorities:   np.ndarray,
    # Curriculum timesteps (keep small for fast debugging — scale up when stable)
    phase1_steps:      int = 10_000,    # 10 planets, easy
    phase2_steps:      int = 30_000,    # 50 planets, intermediate
    phase3_steps:      int = 60_000,    # 100 planets, full
    # BC pretraining
    bc_episodes:       int = 20,
    bc_epochs:         int = 8,
    # PPO hyperparams
    n_steps:           int = 300,       # one full campaign per rollout
    batch_size:        int = 64,
    n_epochs:          int = 10,
    ent_coef:          float = 0.01,    # entropy regularization (adaptive exploration)
    gamma:             float = 0.98,    # long-horizon discount
    seed:              int = 42,
    save_dir:          Optional[Path] = None,
    verbose:           int = 1,
) -> Tuple[MaskablePPO, RewardDecompositionCallback]:
    """
    Three-phase curriculum training:
      Phase 1: 10-planet env  → quick exploration of reward structure
      Phase 2: 50-planet env  → moderate complexity
      Phase 3: 100-planet env → full campaign; BC-pretrained policy fine-tuned

    Behavior Cloning is applied to Phase 3 initial weights to warm-start
    the policy before RL fine-tuning.

    Parameters
    ----------
    phase{1,2,3}_steps : training timesteps per phase (start small, scale up)

    Returns
    -------
    model    : trained MaskablePPO model
    callback : RewardDecompositionCallback with full training log
    """
    save_dir = save_dir or MODELS_DIR
    algo     = MaskablePPO if _HAS_MASKING else PPO
    callback = RewardDecompositionCallback()

    ppo_kwargs = dict(
        policy     = "MlpPolicy",
        n_steps    = n_steps,
        batch_size = batch_size,
        n_epochs   = n_epochs,
        ent_coef   = ent_coef,
        gamma      = gamma,
        verbose    = verbose,
        seed       = seed,
        policy_kwargs = dict(net_arch=[256, 256]),
    )

    # ── Phase 1: 10-planet curriculum ────────────────────────────────────────
    if phase1_steps > 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 1 — 10-planet curriculum ({phase1_steps:,} steps)")
        print(f"{'='*60}")
        env1 = _make_masked_env(dict(
            df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
            true_priorities=true_priorities, n_candidates=10, seed=seed,
        ))
        model = algo(env=env1, **ppo_kwargs)
        model.learn(total_timesteps=phase1_steps, callback=callback)
        model.save(str(save_dir / "ppo_phase1"))
        print("[Phase 1] Saved → models/ppo_phase1.zip")
    else:
        model = None

    # ── Phase 2: 50-planet curriculum ────────────────────────────────────────
    if phase2_steps > 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 2 — 50-planet curriculum ({phase2_steps:,} steps)")
        print(f"{'='*60}")
        env2 = _make_masked_env(dict(
            df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
            true_priorities=true_priorities, n_candidates=50, seed=seed,
        ))
        if model is not None:
            model.set_env(env2)
        else:
            model = algo(env=env2, **ppo_kwargs)
        model.learn(total_timesteps=phase2_steps, reset_num_timesteps=False, callback=callback)
        model.save(str(save_dir / "ppo_phase2"))
        print("[Phase 2] Saved → models/ppo_phase2.zip")

    # ── Phase 3: 100-planet full env + BC pretraining ────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — 100-planet full env ({phase3_steps:,} steps)")
    print(f"{'='*60}")
    env3 = _make_masked_env(dict(
        df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
        true_priorities=true_priorities, n_candidates=100, seed=seed,
    ))
    if model is not None:
        model.set_env(env3)
    else:
        model = algo(env=env3, **ppo_kwargs)

    # Behavior Cloning warm-start on Phase 3
    if bc_episodes > 0:
        obs_buf, act_buf = collect_bc_trajectories(
            df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
            true_priorities=true_priorities,
            n_candidates=100, n_episodes=bc_episodes, seed=seed + 999,
        )
        # Must call learn(0) first to initialize policy weights
        model.learn(total_timesteps=1, reset_num_timesteps=False)
        bc_losses = pretrain_bc(model, obs_buf, act_buf, n_epochs=bc_epochs)

    model.learn(
        total_timesteps=phase3_steps, reset_num_timesteps=False, callback=callback
    )
    model.save(str(save_dir / "ppo_stage3_final"))
    print("[Phase 3] Saved → models/ppo_stage3_final.zip")

    return model, callback


# =============================================================================
# 4. Policy Evaluation
# =============================================================================

def evaluate_policy_campaigns(
    model,
    df:              pd.DataFrame,
    mu_pred:         np.ndarray,
    sigma_pred:      np.ndarray,
    true_priorities: np.ndarray,
    n_candidates:    int = 100,
    n_episodes:      int = 10,
    seed:            int = 999,
) -> Dict:
    """
    Evaluate trained policy over n_episodes held-out campaigns.

    Returns dict with per-episode composite scores and summary stats.
    """
    from src.evaluation import compute_campaign_diversity_score
    print(f"\n[Eval] Evaluating policy over {n_episodes} campaigns …")

    episode_rewards   = []
    composite_scores  = []
    state_buffer      = []
    reward_buffer     = []

    for ep in range(n_episodes):
        env = ExoplanetSchedulingEnv(
            df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
            true_priorities=true_priorities,
            n_candidates=n_candidates, seed=seed + ep,
        )
        obs, _ = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            state_buffer.append(obs.copy())

            # Use action masks if available
            if _HAS_MASKING:
                masks  = env.action_masks()
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, r, done, trunc, info = env.step(int(action))
            total_r += r
            reward_buffer.append(r)
            if done or trunc:
                break

        episode_rewards.append(total_r)
        summary = env.get_campaign_summary()
        composite_scores.append(summary.get("total_gain", 0.0))
        if (ep + 1) % 5 == 0:
            print(f"[Eval]   Episode {ep+1}/{n_episodes} — reward: {total_r:.4f}")

    result = {
        "mean_reward":       float(np.mean(episode_rewards)),
        "std_reward":        float(np.std(episode_rewards)),
        "episode_rewards":   episode_rewards,
        "composite_scores":  composite_scores,
        "state_buffer":      np.array(state_buffer, dtype=np.float32),
        "reward_buffer":     np.array(reward_buffer, dtype=np.float32),
    }
    print(f"[Eval] Mean reward: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
    return result


# =============================================================================
# 5. Generalization / Robustness Tests
# =============================================================================

def evaluate_generalization(
    model,
    df:              pd.DataFrame,
    mu_pred:         np.ndarray,
    sigma_pred:      np.ndarray,
    true_priorities: np.ndarray,
    weather_seeds:   List[int] = [0, 100, 200, 300, 400],
    n_candidates:    int = 100,
) -> pd.DataFrame:
    """
    Evaluate the policy on different weather seeds (generalization test).
    Compares mean reward across conditions to test robustness.
    """
    print("\n[Generalization] Evaluating across weather seeds …")
    rows = []
    for ws in weather_seeds:
        result = evaluate_policy_campaigns(
            model=model, df=df, mu_pred=mu_pred,
            sigma_pred=sigma_pred, true_priorities=true_priorities,
            n_candidates=n_candidates, n_episodes=5, seed=ws,
        )
        rows.append({
            "weather_seed":  ws,
            "mean_reward":   result["mean_reward"],
            "std_reward":    result["std_reward"],
        })
        print(f"  Seed {ws}: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")

    return pd.DataFrame(rows)


# =============================================================================
# 6. Policy Explainability
# =============================================================================

def explain_policy_decision(
    model,
    env:            ExoplanetSchedulingEnv,
    obs:            np.ndarray,
    top_k_targets:  int = 3,
) -> List[dict]:
    """
    Perturbation-based feature attribution: explains why the RL policy
    selected each target by measuring sensitivity of action probabilities
    to individual feature channels.

    For each top-k selected target:
      Attributions computed by zeroing each feature channel and measuring
      drop in selection probability.

    Returns
    -------
    List of dicts with 'target_name', 'action_prob', 'reasoning' (list of strings)
    """
    N    = len(env._shortlist)
    sl   = env._shortlist
    obs_t = torch.FloatTensor(obs).unsqueeze(0)

    feature_names = [
        "priority_mu", "uncertainty_sigma", "detectability",
        "cost_norm", "observed_flag", "visibility_flag",
    ]

    # Get base action probabilities
    with torch.no_grad():
        if _HAS_MASKING:
            masks = env.action_masks()
            dist  = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits[0].numpy()
            logits[~masks] = -1e9
        else:
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits[0].numpy()

    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    top_actions = np.argsort(probs)[::-1][:top_k_targets]

    results = []
    for local_a in top_actions:
        global_i = int(sl[local_a]) if local_a < len(sl) else -1
        name     = str(env._df["pl_name"].iloc[global_i]) if "pl_name" in env._df.columns else f"Planet_{global_i}"
        base_p   = float(probs[local_a])

        # Perturbation attribution
        attributions = {}
        for feat_j, feat_name in enumerate(feature_names):
            obs_perturbed = obs.copy()
            # Zero out channel feat_j for candidate local_a
            obs_perturbed[feat_j * N + local_a] = 0.0
            obs_pt = torch.FloatTensor(obs_perturbed).unsqueeze(0)
            with torch.no_grad():
                d2      = model.policy.get_distribution(obs_pt)
                lg2     = d2.distribution.logits[0].numpy()
                p2      = np.exp(lg2 - lg2.max()); p2 /= p2.sum()
                drop    = base_p - float(p2[local_a])
            attributions[feat_name] = drop   # positive = feature important

        # Build reasoning bullets
        reasoning = []
        for feat, attr in sorted(attributions.items(), key=lambda x: -abs(x[1])):
            val_str = f"{obs[feature_names.index(feat) * N + local_a]:.3f}"
            if attr > 0.01:
                reasoning.append(f"+ {feat} = {val_str} (↑ selection by {attr:.3f})")
            elif attr < -0.01:
                reasoning.append(f"- {feat} = {val_str} (↓ selection by {abs(attr):.3f})")

        results.append({
            "target_name":  name,
            "global_idx":   global_i,
            "action_prob":  base_p,
            "attributions": attributions,
            "reasoning":    reasoning,
        })

    return results


# =============================================================================
# 7. LinUCB Contextual Bandit
# =============================================================================

class LinUCBBandit:
    """
    LinUCB (Disjoint) Contextual Bandit for exoplanet scheduling.

    Context = observation feature vector (same as RL state).
    Arm     = shortlist planet index.
    Reward  = composite multi-objective reward (same as PPO env).

    Scientifically aligned: telescope scheduling is naturally a sequential
    bandit problem. Compared to PPO, LinUCB is:
      - Interpretable (linear model per arm)
      - Sample efficient (closed-form update)
      - Asymptotically optimal (theoretical regret bounds)

    Reference: Li et al. (2010) "A Contextual-Bandit Approach to Personalized
               News Article Recommendation"
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        """
        Parameters
        ----------
        n_arms      : number of planet candidates (shortlist size)
        context_dim : observation feature vector dimension
        alpha       : exploration bonus coefficient (higher = more exploration)
        """
        self.n_arms      = n_arms
        self.d           = context_dim
        self.alpha       = alpha

        # Per-arm ridge regression model: A_a × θ_a = b_a
        self.A = [np.eye(context_dim, dtype=np.float64) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim, dtype=np.float64) for _ in range(n_arms)]

    def select(self, context: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        """
        Select arm with highest UCB score.

        Parameters
        ----------
        context : feature vector (context_dim,)
        mask    : boolean array (n_arms,), True = valid action

        Returns
        -------
        arm_idx : int
        """
        x      = context.astype(np.float64).ravel()
        scores = np.full(self.n_arms, -np.inf)

        for a in range(self.n_arms):
            if mask is not None and not mask[a]:
                continue
            A_inv    = np.linalg.solve(self.A[a], np.eye(self.d))
            theta_a  = A_inv @ self.b[a]
            ucb      = theta_a @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            scores[a] = ucb

        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Online update for arm `arm` given observed reward."""
        x = context.astype(np.float64).ravel()
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def run_episode(
        self,
        env: ExoplanetSchedulingEnv,
    ) -> float:
        """Run one full campaign episode using LinUCB policy."""
        obs, _ = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            mask   = env.action_masks()
            arm    = self.select(obs, mask)
            obs_new, r, done, trunc, info = env.step(arm)
            self.update(arm, obs, r)
            obs     = obs_new
            total_r += r
            if done or trunc:
                break

        return total_r

    def train(
        self,
        df:              pd.DataFrame,
        mu_pred:         np.ndarray,
        sigma_pred:      np.ndarray,
        true_priorities: np.ndarray,
        n_candidates:    int = 100,
        n_episodes:      int = 200,
        seed:            int = 42,
    ) -> List[float]:
        """Train bandit for n_episodes and return per-episode rewards."""
        print(f"\n[LinUCB] Training for {n_episodes} episodes …")
        rewards = []
        for ep in range(n_episodes):
            env = ExoplanetSchedulingEnv(
                df=df, mu_pred=mu_pred, sigma_pred=sigma_pred,
                true_priorities=true_priorities,
                n_candidates=n_candidates, seed=seed + ep,
            )
            r = self.run_episode(env)
            rewards.append(r)
            if (ep + 1) % 50 == 0:
                print(f"[LinUCB]   Episode {ep+1:4d}/{n_episodes} — reward: {np.mean(rewards[-50:]):.4f}")
        print(f"[LinUCB] Final mean reward: {np.mean(rewards[-20:]):.4f}")
        return rewards


# =============================================================================
# 8. RLScheduler — Drop-in BaseScheduler Wrapper
# =============================================================================

from src.scheduler import BaseScheduler, RoundLog


class RLScheduler(BaseScheduler):
    """
    Wraps a trained MaskablePPO (or PPO) model inside the BaseScheduler
    interface so it can be evaluated via the EXISTING run_campaign() runner.

    This enables apple-to-apple comparison against all Stage 2 heuristics
    in a single table — zero extra evaluation code.
    """

    def __init__(
        self,
        df:              pd.DataFrame,
        model,           # trained MaskablePPO model
        mu_pred:         np.ndarray,
        sigma_pred:      np.ndarray,
        true_priorities: np.ndarray,
        n_candidates:    int = 100,
        seed:            int = 0,
    ):
        super().__init__("PPO RL Agent", df)
        self._model           = model
        self._mu_pred         = mu_pred
        self._sigma_pred      = sigma_pred
        self._true_priorities = true_priorities
        self._n_candidates    = n_candidates
        self._seed            = seed

        # Internal env for state computation
        self._rl_env: Optional[ExoplanetSchedulingEnv] = None

    def _init_rl_env(self, simulator, constraint_engine):
        """Lazy-initialize the RL env aligned with the external sim/CE state."""
        self._rl_env = ExoplanetSchedulingEnv(
            df=self.df,
            mu_pred=self._mu_pred,
            sigma_pred=self._sigma_pred,
            true_priorities=self._true_priorities,
            n_candidates=self._n_candidates,
            seed=self._seed,
        )
        # Sync simulator state
        self._rl_env._sim = simulator
        self._rl_env._ce  = constraint_engine

    def select(
        self,
        simulator,
        constraint_engine,
        round_number: int,
        k: int,
        observed_set: set,
    ):
        # Lazy init
        if self._rl_env is None:
            self._init_rl_env(simulator, constraint_engine)

        # Sync external state into RL env
        self._rl_env._sim           = simulator
        self._rl_env._ce            = constraint_engine
        self._rl_env._observed_set  = set(observed_set)
        self._rl_env._current_round = round_number
        self._rl_env._slot_in_round = 0

        selected  = []
        sel_costs = []

        # RL policy selects up to k targets
        for _ in range(k):
            obs  = self._rl_env._get_obs()
            mask = self._rl_env.action_masks()

            if not mask.any():
                break

            if _HAS_MASKING:
                action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
            else:
                action, _ = self._model.predict(obs, deterministic=True)
                action    = int(action)

            local_a    = int(action)
            sl         = self._rl_env._shortlist
            global_idx = int(sl[local_a]) if local_a < len(sl) else -1

            can_obs, cost = constraint_engine.can_observe(global_idx, len(selected))
            if not can_obs:
                break

            constraint_engine.consume_budget(global_idx, len(selected))
            selected.append(global_idx)
            sel_costs.append(cost)

            # Update RL env internal state
            self._rl_env._observed_set.add(global_idx)
            self._rl_env._slot_in_round += 1

        # Build log entry using parent class helper
        if selected:
            gains_arr = simulator.sigma[np.array(selected)] * simulator.detectability[np.array(selected)]
            feas_arr  = np.ones(len(selected))
            cand_idx  = np.array(selected)
            time_before = constraint_engine.time_budget + sum(sel_costs)
            log = self._build_log(
                round_number, selected, sel_costs, simulator,
                constraint_engine, gains_arr, feas_arr, cand_idx,
                0.5, 0.3 * np.exp(-round_number / 15.0), 0.2, time_before,
            )
            self.logs.append(log)

        return selected, sel_costs
