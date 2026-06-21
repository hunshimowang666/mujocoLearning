"""Play the Unitree G1 sine-path policy trained by 03_unitree_g1_walk_train.py."""

from __future__ import annotations

import importlib.util
import os
import site
import sys
from pathlib import Path
from typing import Literal


def configure_process_start_environment() -> None:
  env_changed = False

  if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    env_changed = True

  wsl_lib = "/usr/lib/wsl/lib"
  if Path(wsl_lib).exists():
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = [path for path in ld_library_path.split(":") if path]
    if wsl_lib not in ld_paths:
      os.environ["LD_LIBRARY_PATH"] = (
        wsl_lib if not ld_library_path else f"{wsl_lib}:{ld_library_path}"
      )
      env_changed = True

  if env_changed and os.environ.get("MJLAB_WSL_ENV_READY") != "1":
    os.environ["MJLAB_WSL_ENV_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


configure_process_start_environment()

USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
  sys.path.remove(USER_SITE)

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

DIRECT_SHOW_SINE_PATH = True


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
    sine_lin_vel_x=TRAIN_CFG.DIRECT_LIN_VEL_X,
    sine_lin_vel_y=TRAIN_CFG.DIRECT_LIN_VEL_Y,
    sine_ang_vel_z_amplitude=TRAIN_CFG.DIRECT_ANG_VEL_Z_AMPLITUDE,
    sine_period=TRAIN_CFG.DIRECT_PERIOD,
    sine_warmup_duration=TRAIN_CFG.DIRECT_SINE_WARMUP_DURATION,
    show_sine_path=DIRECT_SHOW_SINE_PATH,
    sine_path_duration=TRAIN_CFG.DIRECT_PATH_DURATION,
    sine_max_path_error=TRAIN_CFG.DIRECT_MAX_PATH_ERROR,
    sine_path_error_grace_duration=TRAIN_CFG.DIRECT_PATH_ERROR_GRACE_DURATION,
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
