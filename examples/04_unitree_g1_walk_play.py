"""Play the Unitree G1 walking policy trained by 03_unitree_g1_walk_train.py."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Literal


EXAMPLES_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = EXAMPLES_DIR / "03_unitree_g1_walk_train.py"
MJLAB_ROOT = EXAMPLES_DIR / "unitree_rl_mjlab"
MJLAB_SCRIPT_DIR = MJLAB_ROOT / "scripts"

sys.path.insert(0, str(MJLAB_SCRIPT_DIR))
sys.path.insert(0, str(MJLAB_ROOT))


def _load_train_module():
  spec = importlib.util.spec_from_file_location("unitree_g1_walk_train_cfg", TRAIN_SCRIPT)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load train script: {TRAIN_SCRIPT}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


TRAIN_CFG = _load_train_module()

from play import PlayConfig, run_play  # noqa: E402


##
# Direct-run settings.
##

DIRECT_PLAY_TASK = TRAIN_CFG.DIRECT_TASK
DIRECT_PLAY_EXPERIMENT_NAME = TRAIN_CFG.DIRECT_EXPERIMENT_NAME
DIRECT_PLAY_AGENT: Literal["zero", "random", "trained"] = "trained"
DIRECT_PLAY_LATEST_CHECKPOINT = True
DIRECT_PLAY_CHECKPOINT_FILE: str | None = None
DIRECT_PLAY_NUM_ENVS = 1
DIRECT_PLAY_DEVICE = "cuda:0"
DIRECT_PLAY_VIEWER: Literal["auto", "native", "viser"] = "viser"

# Keep rendering lightweight; this is smoother on WSL than trying to render every
# physics step.
DIRECT_RENDER_FRAME_RATE = 10.0
DIRECT_TARGET_STEPS_PER_SECOND: float | None = 10.0

DIRECT_SHOW_STRAIGHT_PATH = True


def configure_runtime_environment() -> None:
  TRAIN_CFG.configure_runtime_environment()


def _resolve_checkpoint() -> str | None:
  if DIRECT_PLAY_AGENT != "trained":
    return None
  if DIRECT_PLAY_CHECKPOINT_FILE is not None:
    return DIRECT_PLAY_CHECKPOINT_FILE
  if DIRECT_PLAY_LATEST_CHECKPOINT:
    return str(TRAIN_CFG.find_latest_checkpoint(DIRECT_PLAY_EXPERIMENT_NAME))
  return None


def direct_play_config() -> PlayConfig:
  return PlayConfig(
    agent=DIRECT_PLAY_AGENT,
    checkpoint_file=_resolve_checkpoint(),
    num_envs=DIRECT_PLAY_NUM_ENVS,
    device=DIRECT_PLAY_DEVICE,
    viewer=DIRECT_PLAY_VIEWER,
    render_frame_rate=DIRECT_RENDER_FRAME_RATE,
    target_steps_per_second=DIRECT_TARGET_STEPS_PER_SECOND,
    constant_lin_vel_x=TRAIN_CFG.DIRECT_LIN_VEL_X,
    constant_lin_vel_y=TRAIN_CFG.DIRECT_LIN_VEL_Y,
    constant_ang_vel_z=TRAIN_CFG.DIRECT_ANG_VEL_Z,
    show_straight_path=DIRECT_SHOW_STRAIGHT_PATH,
    straight_path_duration=TRAIN_CFG.DIRECT_EPISODE_LENGTH,
    no_terminations=False,
  )


def main() -> None:
  configure_runtime_environment()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  os.chdir(MJLAB_ROOT)
  cfg = direct_play_config()
  print("[INFO] Running 04_unitree_g1_walk_play.py with direct-run settings")
  run_play(DIRECT_PLAY_TASK, cfg)


if __name__ == "__main__":
  main()
