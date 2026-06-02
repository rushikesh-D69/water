import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib.ppo_mask import MaskablePPO
from src.observation_simulator import ObservationSimulator
from src.constraint_engine import ObservationConstraintEngine
from src.scheduler import run_campaign, AdaptiveScheduler
from src.rl_scheduler import RLScheduler

def evaluate():
    os.makedirs("plots", exist_ok=True)
    print("Loading data...")
    df_ml = pd.read_csv("data/exoplanets_processed.csv")
    
    TARGET = "priority_score"
    mu_pred = np.clip(df_ml[TARGET].values, 0, 1)
    sigma_pred = np.clip(np.full(len(df_ml), 0.1), 0.01, 1.0)
    
    # 1. Evaluate Baseline (Adaptive Scheduler)
    print("Running Baseline Campaign...")
    sim_baseline = ObservationSimulator(df=df_ml, initial_means=mu_pred.copy(), initial_sigmas=sigma_pred.copy(), seed=42)
    ce_baseline = ObservationConstraintEngine(df_ml, seed=42)
    baseline_scheduler = AdaptiveScheduler(df_ml, tau=15.0)
    run_campaign(baseline_scheduler, sim_baseline, ce_baseline, n_rounds=30, k_per_round=10, verbose=False)
    
    df_baseline = baseline_scheduler.get_logs_df()
    
    # 2. Evaluate RL Agent
    print("Loading RL Model and Running Campaign...")
    model_path = "models/maskable_ppo_scheduler.zip"
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did the 2M training finish successfully?")
        return
        
    model = MaskablePPO.load(model_path)
    
    sim_rl = ObservationSimulator(df=df_ml, initial_means=mu_pred.copy(), initial_sigmas=sigma_pred.copy(), seed=42)
    ce_rl = ObservationConstraintEngine(df_ml, seed=42)
    rl_scheduler = RLScheduler("RL PPO Scheduler", df_ml, model, top_k=100)
    run_campaign(rl_scheduler, sim_rl, ce_rl, n_rounds=30, k_per_round=10, verbose=False)
    
    df_rl = rl_scheduler.get_logs_df()
    
    # 3. Generate Plot
    print("Generating output graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(df_baseline["round"], df_baseline["cum_sci_gain"], marker='o', label='Baseline (Adaptive Heuristic)')
    plt.plot(df_rl["round"], df_rl["cum_sci_gain"], marker='s', label='Deep RL Agent (MaskablePPO)')
    
    plt.title("Cumulative Scientific Gain: RL vs Baseline")
    plt.xlabel("Observation Round (Night)")
    plt.ylabel("Cumulative Gain")
    plt.legend()
    plt.grid(True)
    
    plot_path = "plots/rl_vs_baseline.png"
    plt.savefig(plot_path)
    print(f"Graph saved to {plot_path}")

if __name__ == "__main__":
    evaluate()
