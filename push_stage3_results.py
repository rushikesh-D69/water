"""
push_stage3_results.py
=======================
Run this as the LAST cell in stage3_pipeline.ipynb (in Colab) to
automatically commit and push all Stage 3 outputs back to GitHub.

Usage in Colab (add as final cell):
    %run push_stage3_results.py

OR paste directly:
    import subprocess, os
    exec(open('push_stage3_results.py').read())
"""

import subprocess
import os

print("=" * 60)
print("  Stage 3 → GitHub Auto-Push")
print("=" * 60)

# ── Configure git identity ────────────────────────────────────────────────────
os.system('git config user.email "stage3-colab@water.ai"')
os.system('git config user.name  "Stage3 AutoPush"')

# ── Inject GitHub token from Colab Secrets ────────────────────────────────────
try:
    from google.colab import userdata
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    os.system(
        f'git remote set-url origin '
        f'https://{GITHUB_TOKEN}@github.com/rushikesh-D69/water.git'
    )
    print('[Auth] ✓ GitHub token injected from Colab Secrets.')
except Exception:
    print('[Auth] ⚠  No GITHUB_TOKEN in Colab Secrets.')
    print('       Add it: Colab sidebar → 🔑 icon → New secret → GITHUB_TOKEN')
    print('       Get token: github.com → Settings → Developer settings → Tokens')

# ── Files to push ─────────────────────────────────────────────────────────────
OUTPUT_FILES = [
    # 7 RL evaluation plots
    'plots/s3_training_reward_curve.png',
    'plots/s3_policy_heatmap.png',
    'plots/s3_reward_decomposition.png',
    'plots/s3_exploration_timeline.png',
    'plots/s3_pareto_extended.png',
    'plots/s3_generalization.png',
    'plots/s3_state_tsne.png',
    # Comparison data
    'data/stage3_comparison.csv',
    # Trained PPO model (final only — intermediate checkpoints ignored by .gitignore)
    'models/ppo_stage3_final.zip',
    # Notebook with outputs embedded
    'stage3_pipeline.ipynb',
]

print("\n[Git] Staging files...")
staged, missing = [], []
for f in OUTPUT_FILES:
    if os.path.exists(f):
        os.system(f'git add "{f}"')
        staged.append(f)
        print(f"  ✓  {f}")
    else:
        missing.append(f)
        print(f"  ✗  {f}  (not found — not generated yet)")

print(f"\n[Git] Staged {len(staged)}/{len(OUTPUT_FILES)} files.")
if missing:
    print(f"[Git] Skipped {len(missing)} files not yet generated.")

# ── Check if there's anything to commit ──────────────────────────────────────
diff = subprocess.run(
    ['git', 'diff', '--cached', '--quiet'],
    capture_output=True
)

if diff.returncode == 0:
    print("\n[Git] Nothing new to commit — all files already up to date on GitHub.")
else:
    # ── Commit ────────────────────────────────────────────────────────────────
    commit_msg = (
        "results(stage3): RL training outputs — "
        "7 plots + comparison table + PPO model\n\n"
        "- s3_training_reward_curve.png  : PPO episode reward + running mean\n"
        "- s3_policy_heatmap.png         : T_eq vs Radius RL selection frequency\n"
        "- s3_reward_decomposition.png   : G/D/E/P reward components per round\n"
        "- s3_exploration_timeline.png   : Exploration vs exploitation per round\n"
        "- s3_pareto_extended.png        : Pareto frontier + RL/Bandit nodes\n"
        "- s3_generalization.png         : Robustness across 5 weather seeds\n"
        "- s3_state_tsne.png             : State embedding reward clusters\n"
        "- data/stage3_comparison.csv    : 7-scheduler unified comparison table\n"
        "- models/ppo_stage3_final.zip   : Trained MaskablePPO final checkpoint"
    )
    c = subprocess.run(
        ['git', 'commit', '-m', commit_msg],
        capture_output=True, text=True
    )
    print("\n[Git] Commit output:")
    print(c.stdout or c.stderr)

    # ── Push ──────────────────────────────────────────────────────────────────
    print("[Git] Pushing to GitHub...")
    p = subprocess.run(
        ['git', 'push', 'origin', 'main'],
        capture_output=True, text=True
    )
    if p.returncode == 0:
        print("\n✅  Successfully pushed to GitHub!")
        print("    https://github.com/rushikesh-D69/water")
        print("\n    View plots in the README:")
        print("    https://github.com/rushikesh-D69/water#stage-3--reinforcement-learning-autonomous-scheduler")
    else:
        print("\n❌  Push failed:")
        print(p.stderr)
        print("\n── Troubleshooting ──────────────────────────────────────────")
        print("1. Add GITHUB_TOKEN to Colab Secrets (🔑 icon in sidebar)")
        print("2. Token needs 'repo' scope: github.com → Settings → Developer settings")
        print("3. If token expired, generate a new one")
