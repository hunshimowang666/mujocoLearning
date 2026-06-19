"""Train Unitree G1 to walk forward with MJLab GPU-parallel PPO.

Edit the direct-run settings below, then run this file directly from the IDE.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import tyro


EXAMPLES_DIR = Path(__file__).resolve().parent
MJLAB_ROOT = EXAMPLES_DIR / "unitree_rl_mjlab"
MJLAB_SCRIPT_DIR = MJLAB_ROOT / "scripts"

sys.path.insert(0, str(MJLAB_SCRIPT_DIR))
sys.path.insert(0, str(MJLAB_ROOT))

from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # noqa: E402
from train import TrainConfig, launch_training  # noqa: E402

import src.tasks.velocity.mdp as velocity_mdp  # noqa: E402


##
# Direct-run settings.
##

# Use "Unitree-G1-23Dof-Flat" here if you explicitly want the 23-DOF variant.
DIRECT_TASK = "Unitree-G1-Flat"
DIRECT_EXPERIMENT_NAME = "g1_humanoid_velocity_mjlab"

# Body-frame command. Positive x is the robot's forward direction.
DIRECT_LIN_VEL_X = 0.5
DIRECT_LIN_VEL_Y = 0.0
DIRECT_ANG_VEL_Z = 0.0

DIRECT_EPISODE_LENGTH = 20.0
DIRECT_NUM_ENVS = 2048
DIRECT_MAX_ITERATIONS = 5000
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]

# True: continue from the newest checkpoint if one exists. False: start fresh.
DIRECT_RESUME_LATEST = True

# Used only for fresh runs. Resume never deletes old networks.
DIRECT_DELETE_OLD_NETWORKS = False


def configure_runtime_environment() -> None:
  os.environ.pop("PYTHONNOUSERSITE", None)
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


@dataclass(frozen=True)
class UnitreeG1WalkTrainConfig:
  task: str = DIRECT_TASK
  experiment_name: str = DIRECT_EXPERIMENT_NAME
  lin_vel_x: float = DIRECT_LIN_VEL_X
  lin_vel_y: float = DIRECT_LIN_VEL_Y
  ang_vel_z: float = DIRECT_ANG_VEL_Z
  episode_length: float = DIRECT_EPISODE_LENGTH
  num_envs: int | None = DIRECT_NUM_ENVS
  max_iterations: int | None = DIRECT_MAX_ITERATIONS
  gpu_ids: list[int] | Literal["all"] | None = field(
    default_factory=lambda: DIRECT_GPU_IDS
  )
  resume_latest: bool = DIRECT_RESUME_LATEST
  resume_run: str | None = None
  resume_checkpoint: str = "model_.*.pt"
  delete_old_networks: bool = DIRECT_DELETE_OLD_NETWORKS


def direct_train_config() -> UnitreeG1WalkTrainConfig:
  return UnitreeG1WalkTrainConfig()


def _configure_fixed_forward_command(
  cfg: TrainConfig,
  args: UnitreeG1WalkTrainConfig,
) -> None:
  twist_cmd = cfg.env.commands["twist"]
  assert isinstance(twist_cmd.ranges, UniformVelocityCommandCfg.Ranges)

  twist_cmd.ranges.lin_vel_x = (args.lin_vel_x, args.lin_vel_x)
  twist_cmd.ranges.lin_vel_y = (args.lin_vel_y, args.lin_vel_y)
  twist_cmd.ranges.ang_vel_z = (args.ang_vel_z, args.ang_vel_z)
  twist_cmd.ranges.heading = None
  twist_cmd.heading_command = False
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.resampling_time_range = (1.0e9, 1.0e9)


def build_train_config(args: UnitreeG1WalkTrainConfig) -> TrainConfig:
  cfg = TrainConfig.from_task(args.task)
  _configure_fixed_forward_command(cfg, args)

  cfg.env.episode_length_s = args.episode_length
  cfg.env.terminations["walk_episode_complete"] = TerminationTermCfg(
    func=velocity_mdp.sine_path_complete,
    time_out=True,
    params={"duration": args.episode_length},
  )

  cfg.agent.experiment_name = args.experiment_name
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
        print(f"[INFO] {exc}; starting a fresh G1 run instead.")
        resume = False
        cfg.agent.resume = False
      else:
        cfg.agent.load_run = checkpoint_path.parent.name
        cfg.agent.load_checkpoint = checkpoint_path.name
    else:
      cfg.agent.load_run = args.resume_run
      cfg.agent.load_checkpoint = args.resume_checkpoint

  if cfg.agent.resume:
    print(
      "[INFO] Resuming G1 walking training from "
      f"run={cfg.agent.load_run}, checkpoint={cfg.agent.load_checkpoint}"
    )

  return replace(
    cfg,
    clear_old_logs=args.delete_old_networks and not resume,
    gpu_ids=args.gpu_ids,
  )


def main() -> None:
  configure_runtime_environment()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if len(sys.argv) == 1:
    args = direct_train_config()
    print("[INFO] Running 03_unitree_g1_walk_train.py with direct-run settings")
  else:
    args = tyro.cli(UnitreeG1WalkTrainConfig)

  os.chdir(MJLAB_ROOT)
  train_cfg = build_train_config(args)
  launch_training(task_id=args.task, args=train_cfg)


if __name__ == "__main__":
  main()
