import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env

from src.constraint_engine import ObservationConstraintEngine
from src.observation_simulator import ObservationSimulator
from src.scheduler import run_campaign
from src.rl_env import ExoplanetSchedulingEnv
from src.rl_scheduler import RLScheduler

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

def main():
    print("Loading data...")
    df_ml = pd.read_csv("data/exoplanets_processed.csv")
    
    TARGET = "priority_score"
    # Fallback identical to stage2_pipeline.ipynb if models are missing
    mu_pred = np.clip(df_ml[TARGET].values, 0, 1)
    sigma_pred = np.clip(np.full(len(df_ml), 0.1), 0.01, 1.0)
    
    print("Initializing environment...")
    simulator = ObservationSimulator(
        df=df_ml, 
        initial_means=mu_pred.copy(), 
        initial_sigmas=sigma_pred.copy(), 
        seed=42
    )
    constraint_engine = ObservationConstraintEngine(df_ml, seed=42)
    
    base_env = ExoplanetSchedulingEnv(
        df=df_ml, 
        simulator=simulator, 
        constraint_engine=constraint_engine, 
        max_rounds=30, 
        top_k_targets=100
    )
    
    env = ActionMasker(base_env, mask_fn)
    
    print("Training MaskablePPO model (2,000,000 steps)...")
    model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, device="auto")
    model.learn(total_timesteps=2000000)
    print("Training complete.")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/maskable_ppo_scheduler")
    
    print("\nEvaluating RLScheduler in full campaign...")
    
    # We need fresh simulator and constraint engine for the campaign evaluation
    sim_eval = ObservationSimulator(
        df=df_ml, 
        initial_means=mu_pred.copy(), 
        initial_sigmas=sigma_pred.copy(), 
        seed=100
    )
    ce_eval = ObservationConstraintEngine(df_ml, seed=100)
    
    rl_scheduler = RLScheduler("RL Maskable PPO Scheduler", df_ml, model, top_k=100)
    
    results = run_campaign(
        scheduler=rl_scheduler, 
        simulator=sim_eval, 
        constraint_engine=ce_eval, 
        n_rounds=30, 
        k_per_round=10, 
        verbose=True
    )
    
    print("\n[RL Campaign] Cumulative Gain:", results["cumulative_gain"])
    print("[RL Campaign] Observed Planets:", results["n_observed"])
    print("[RL Campaign] Total Time Used:", results["total_time_used"])

if __name__ == "__main__":
    main()
