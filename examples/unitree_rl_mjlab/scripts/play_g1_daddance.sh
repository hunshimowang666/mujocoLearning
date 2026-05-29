#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

unset PYTHONNOUSERSITE
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export WANDB_MODE=disabled
export MUJOCO_GL=egl

/home/administrator/mujocoLearning/venv/bin/python scripts/play.py Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint_file=logs/g1_moves/B_DadDance/policy/B_DadDance_policy.pt \
  --motion_file=logs/g1_moves/B_DadDance/training/B_DadDance.npz \
  --num-envs=1 \
  --device=cuda:0 \
  --viewer=viser \
  --hide-motion-reference=True
