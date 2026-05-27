"""
push_stage3_results.py
=======================
Run as the LAST step in Colab after stage3_pipeline.ipynb completes.

Pushes ALL newly created/modified files to GitHub:
  - plots/s3_*.png          (7 RL evaluation plots)
  - plots/*.png             (any regenerated Stage 2 plots too)
  - data/stage3_comparison.csv
  - models/ppo_stage3_final.zip
  - stage3_pipeline.ipynb   (with embedded cell outputs)
  - Any other file created during the notebook run

Respects .gitignore automatically (intermediate checkpoints,
monitor logs, tb_logs etc. are excluded as defined there).

Usage in Colab — add this as the final cell:
    %run push_stage3_results.py
"""

import subprocess
import os
from pathlib import Path

print("=" * 60)
print("  Stage 3 → GitHub Auto-Push (ALL outputs)")
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
    print('       Get token: github.com → Settings → Developer settings → Tokens (classic)')
    print('       Token needs: repo scope')

# ── Show what will be staged ──────────────────────────────────────────────────
print("\n[Git] Scanning for new/modified files...")
status = subprocess.run(
    ['git', 'status', '--short'],
    capture_output=True, text=True
)
lines = [l for l in status.stdout.strip().splitlines() if l.strip()]
if lines:
    print(f"  Found {len(lines)} changed file(s):")
    for l in lines:
        print(f"    {l}")
else:
    print("  No changes detected — everything already pushed.")

# ── Stage ALL new and modified files (respects .gitignore) ───────────────────
print("\n[Git] Staging all files (git add -A) ...")
r = subprocess.run(['git', 'add', '-A'], capture_output=True, text=True)
if r.returncode != 0:
    print(f"  Warning: {r.stderr}")
else:
    print("  ✓ Staged.")

# ── Show exactly what's staged ───────────────────────────────────────────────
cached = subprocess.run(
    ['git', 'diff', '--cached', '--name-status'],
    capture_output=True, text=True
)
staged_files = [l for l in cached.stdout.strip().splitlines() if l.strip()]
if staged_files:
    print(f"\n[Git] {len(staged_files)} file(s) queued for commit:")
    for f in staged_files:
        print(f"    {f}")
else:
    print("\n[Git] Nothing new to commit — all files already on GitHub.")
    import sys; sys.exit(0)

# ── Commit ────────────────────────────────────────────────────────────────────
# Build dynamic commit message listing plot/data files
plot_files  = [f for f in staged_files if 'plots/' in f or '.png' in f]
data_files  = [f for f in staged_files if 'data/'  in f or '.csv' in f]
model_files = [f for f in staged_files if 'models/' in f or '.zip' in f]
other_files = [f for f in staged_files
               if f not in plot_files + data_files + model_files]

body_lines = []
if plot_files:  body_lines.append(f"- {len(plot_files)} plot(s): " + ", ".join(
    Path(f.split()[-1]).name for f in plot_files))
if data_files:  body_lines.append(f"- {len(data_files)} data file(s): " + ", ".join(
    Path(f.split()[-1]).name for f in data_files))
if model_files: body_lines.append(f"- {len(model_files)} model(s): " + ", ".join(
    Path(f.split()[-1]).name for f in model_files))
if other_files: body_lines.append(f"- {len(other_files)} other file(s)")

commit_msg = (
    f"results(stage3): Colab run outputs — "
    f"{len(staged_files)} files\n\n"
    + "\n".join(body_lines)
)

c = subprocess.run(
    ['git', 'commit', '-m', commit_msg],
    capture_output=True, text=True
)
print(f"\n[Git] Commit:\n  {c.stdout.strip() or c.stderr.strip()}")

# ── Push ──────────────────────────────────────────────────────────────────────
print("\n[Git] Pushing to GitHub ...")
p = subprocess.run(
    ['git', 'push', 'origin', 'main'],
    capture_output=True, text=True
)

if p.returncode == 0:
    print("\n" + "=" * 60)
    print("  ✅  All Stage 3 outputs pushed to GitHub!")
    print("=" * 60)
    print(f"  Repo: https://github.com/rushikesh-D69/water")
    print(f"  Plots:  https://github.com/rushikesh-D69/water/tree/main/plots")
    print(f"  Data:   https://github.com/rushikesh-D69/water/tree/main/data")
    print(f"  Models: https://github.com/rushikesh-D69/water/tree/main/models")
else:
    print("\n❌  Push failed:")
    print(p.stderr)
    print("\n── Troubleshooting ──────────────────────────────────────────")
    print("1. Add GITHUB_TOKEN to Colab Secrets (🔑 icon in sidebar)")
    print("2. Token needs 'repo' scope")
    print("   github.com → Settings → Developer settings → Tokens (classic)")
    print("3. If token expired, generate a new one and update the secret")
