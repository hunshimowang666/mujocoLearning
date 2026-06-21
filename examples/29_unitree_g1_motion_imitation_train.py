"""Train Unitree G1 motion imitation with MJLab GPU-parallel PPO.

This script uses an open reference motion file, not a pretrained policy. Edit
the direct-run settings below, then run this file directly from VS Code.
"""

from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import tyro


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
MJLAB_ROOT = EXAMPLES_DIR / "unitree_rl_mjlab"
MJLAB_SCRIPT_DIR = MJLAB_ROOT / "scripts"

sys.path.insert(0, str(MJLAB_SCRIPT_DIR))
sys.path.insert(0, str(MJLAB_ROOT))

from train import TrainConfig, launch_training  # noqa: E402


##
# Direct-run settings.
##

DIRECT_TASK = "Unitree-G1-Tracking-No-State-Estimation"
DIRECT_EXPERIMENT_NAME = "g1_motion_imitation_dance1_subject2_mjlab"

# This is reference motion data, not a neural-network policy.
DIRECT_MOTION_FILE = str(
  MJLAB_ROOT
  / "deploy"
  / "robots"
  / "g1"
  / "config"
  / "policy"
  / "mimic"
  / "dance1_subject2"
  / "params"
  / "dance1_subject2.npz"
)

DIRECT_NUM_ENVS = 4096
DIRECT_MAX_ITERATIONS = 30001
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]

# True: continue from the newest checkpoint if one exists. False: start fresh.
DIRECT_RESUME_LATEST = True

# Used only for fresh runs. Resume never deletes old networks.
DIRECT_DELETE_OLD_NETWORKS = False

REQUIRED_MOTION_KEYS = (
  "fps",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def configure_runtime_environment() -> None:
  os.environ.setdefault("PYTHONNOUSERSITE", "1")
  os.environ.setdefault("WANDB_MODE", "disabled")
  os.environ.setdefault("MUJOCO_GL", "egl")
  wsl_lib = "/usr/lib/wsl/lib"
  if Path(wsl_lib).exists():
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if wsl_lib not in ld_library_path.split(":"):
      os.environ["LD_LIBRARY_PATH"] = (
        wsl_lib if not ld_library_path else f"{wsl_lib}:{ld_library_path}"
      )


def _checkpoint_step(path: Path) -> int:
  try:
    return int(path.stem.split("_")[-1])
  except ValueError:
    return -1


def find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = MJLAB_ROOT / "logs" / "rsl_rl" / experiment_name
  checkpoints = sorted(
    log_root.glob("*/model_*.pt"),
    key=lambda path: (path.parent.name, _checkpoint_step(path)),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return checkpoints[-1]


def resolve_motion_file(motion_file: str) -> Path:
  path = Path(motion_file).expanduser()
  if path.is_absolute() and path.exists():
    return path

  candidates = (
    EXAMPLES_DIR / path,
    MJLAB_ROOT / path,
    Path.cwd() / path,
  )
  for candidate in candidates:
    if candidate.exists():
      return candidate.resolve()

  if path.is_absolute():
    return path
  return (EXAMPLES_DIR / path).resolve()


def validate_motion_file(path: Path) -> None:
  if not path.exists():
    raise FileNotFoundError(f"Motion file not found: {path}")
  with np.load(path) as data:
    missing = [key for key in REQUIRED_MOTION_KEYS if key not in data.files]
    if missing:
      raise ValueError(f"Motion file {path} is missing keys: {missing}")
    joint_pos_shape = data["joint_pos"].shape
    body_pos_shape = data["body_pos_w"].shape
    if len(joint_pos_shape) != 2 or joint_pos_shape[1] != 29:
      raise ValueError(
        f"Expected 29 G1 joint columns in joint_pos, got {joint_pos_shape}"
      )
    if len(body_pos_shape) != 3 or body_pos_shape[2] != 3:
      raise ValueError(f"Unexpected body_pos_w shape: {body_pos_shape}")
    print(
      "[INFO] Motion file OK: "
      f"{path} | frames={joint_pos_shape[0]}, joints={joint_pos_shape[1]}, "
      f"fps={float(data['fps'][0]):.1f}"
    )


@dataclass(frozen=True)
class UnitreeG1MotionImitationTrainConfig:
  task: str = DIRECT_TASK
  experiment_name: str = DIRECT_EXPERIMENT_NAME
  motion_file: str = DIRECT_MOTION_FILE
  num_envs: int | None = DIRECT_NUM_ENVS
  max_iterations: int | None = DIRECT_MAX_ITERATIONS
  gpu_ids: list[int] | Literal["all"] | None = field(
    default_factory=lambda: DIRECT_GPU_IDS
  )
  resume_latest: bool = DIRECT_RESUME_LATEST
  resume_run: str | None = None
  resume_checkpoint: str = "model_.*.pt"
  delete_old_networks: bool = DIRECT_DELETE_OLD_NETWORKS


def direct_train_config() -> UnitreeG1MotionImitationTrainConfig:
  return UnitreeG1MotionImitationTrainConfig()


def build_train_config(args: UnitreeG1MotionImitationTrainConfig) -> TrainConfig:
  motion_path = resolve_motion_file(args.motion_file)
  validate_motion_file(motion_path)

  cfg = TrainConfig.from_task(args.task)
  cfg.agent.experiment_name = args.experiment_name
  motion_cmd = cfg.env.commands.get("motion")
  if motion_cmd is not None and hasattr(motion_cmd, "motion_file"):
    motion_cmd.motion_file = str(motion_path)

  if args.max_iterations is not None:
    cfg.agent.max_iterations = args.max_iterations
  if args.num_envs is not None:
    cfg.env.scene.num_envs = args.num_envs

  if args.resume_latest and args.resume_run is not None:
    raise ValueError("Use either resume_latest or resume_run, not both.")

  resume = args.resume_latest or args.resume_run is not None
  if resume:
    cfg.agent.resume = True
    if args.resume_latest:
      try:
        checkpoint_path = find_latest_checkpoint(cfg.agent.experiment_name)
      except FileNotFoundError as exc:
        print(f"[INFO] {exc}; starting a fresh G1 imitation run instead.")
        resume = False
        cfg.agent.resume = False
      else:
        cfg.agent.load_run = checkpoint_path.parent.name
        cfg.agent.load_checkpoint = checkpoint_path.name
    else:
      cfg.agent.load_run = args.resume_run
      cfg.agent.load_checkpoint = args.resume_checkpoint
    if resume:
      print(
        "[INFO] Resuming G1 motion imitation from "
        f"run={cfg.agent.load_run}, checkpoint={cfg.agent.load_checkpoint}"
      )

  print(
    "[INFO] G1 motion imitation training: "
    f"task={args.task}, num_envs={args.num_envs}, "
    f"max_iterations={args.max_iterations}"
  )

  return replace(
    cfg,
    motion_file=str(motion_path),
    clear_old_logs=args.delete_old_networks and not resume,
    gpu_ids=args.gpu_ids,
  )


def main() -> None:
  configure_runtime_environment()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if len(sys.argv) == 1:
    args = direct_train_config()
    print("[INFO] Running 29_unitree_g1_motion_imitation_train.py")
  else:
    args = tyro.cli(UnitreeG1MotionImitationTrainConfig)

  os.chdir(MJLAB_ROOT)
  train_cfg = build_train_config(args)
  launch_training(task_id=args.task, args=train_cfg)


if __name__ == "__main__":
  main()
