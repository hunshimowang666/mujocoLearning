"""Train Unitree Go2 to follow a gentle continuous curve.

This is intentionally not a circle.  It uses a small sinusoidal yaw-rate command
with a long period, producing a smooth low-curvature path that is easier for Go2
than tight turning.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import tyro

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from train import TrainConfig, launch_training

import src.tasks.velocity.mdp as velocity_mdp
from src.tasks.velocity.mdp.sine_velocity_command import SineVelocityCommandCfg


##
# Direct-run settings.
# Edit these values, then run this file directly from the IDE.
##

DIRECT_TASK = "Unitree-Go2-Flat"
DIRECT_EXPERIMENT_NAME = "go2_gentle_curve_velocity"
DIRECT_LIN_VEL_X = 0.5
DIRECT_LIN_VEL_Y = 0.0
DIRECT_ANG_VEL_Z_AMPLITUDE = 0.06
DIRECT_PERIOD = 20.0
DIRECT_RANDOMIZE_PHASE = True
DIRECT_PATH_DURATION = 20.0
DIRECT_NUM_ENVS = 4096
DIRECT_MAX_ITERATIONS = 1000
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]

# False: start a fresh run. True: continue from the newest checkpoint.
DIRECT_RESUME_LATEST = False

# Used only for fresh runs. Resume never deletes old networks.
DIRECT_DELETE_OLD_NETWORKS = True


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


def find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = Path("logs") / "rsl_rl" / experiment_name
  checkpoints = sorted(
    log_root.glob("*/model_*.pt"),
    key=lambda path: (path.parent.name, int(path.stem.split("_")[-1])),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return checkpoints[-1]


@dataclass(frozen=True)
class Go2CurveTrainConfig:
  task: str = "Unitree-Go2-Flat"
  lin_vel_x: float = 0.5
  lin_vel_y: float = 0.0
  ang_vel_z_amplitude: float = 0.06
  period: float = 20.0
  randomize_phase: bool = True
  path_duration: float = 20.0
  num_envs: int | None = None
  max_iterations: int | None = 1000
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  delete_old_networks: bool = True
  resume_latest: bool = False
  resume_run: str | None = None
  resume_checkpoint: str = "model_.*.pt"


def direct_train_config() -> Go2CurveTrainConfig:
  return Go2CurveTrainConfig(
    task=DIRECT_TASK,
    lin_vel_x=DIRECT_LIN_VEL_X,
    lin_vel_y=DIRECT_LIN_VEL_Y,
    ang_vel_z_amplitude=DIRECT_ANG_VEL_Z_AMPLITUDE,
    period=DIRECT_PERIOD,
    randomize_phase=DIRECT_RANDOMIZE_PHASE,
    path_duration=DIRECT_PATH_DURATION,
    num_envs=DIRECT_NUM_ENVS,
    max_iterations=DIRECT_MAX_ITERATIONS,
    gpu_ids=DIRECT_GPU_IDS,
    delete_old_networks=DIRECT_DELETE_OLD_NETWORKS,
    resume_latest=DIRECT_RESUME_LATEST,
  )


def build_train_config(args: Go2CurveTrainConfig) -> TrainConfig:
  cfg = TrainConfig.from_task(args.task)

  cfg.env.commands["twist"] = SineVelocityCommandCfg(
    entity_name="robot",
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(args.lin_vel_x, args.lin_vel_x),
      lin_vel_y=(args.lin_vel_y, args.lin_vel_y),
      ang_vel_z=(-args.ang_vel_z_amplitude, args.ang_vel_z_amplitude),
      heading=None,
    ),
    heading_command=False,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    resampling_time_range=(1.0e9, 1.0e9),
    lin_vel_x=args.lin_vel_x,
    lin_vel_y=args.lin_vel_y,
    ang_vel_z_amplitude=args.ang_vel_z_amplitude,
    period=args.period,
    randomize_phase=args.randomize_phase,
  )

  # The default linear velocity reward tracks horizontal speed magnitude.  For
  # this gentle-curve task we also require positive body-frame forward velocity,
  # otherwise the policy can learn to stand still while collecting stability
  # rewards.
  if "track_linear_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_linear_velocity"].weight = 2.0
  cfg.env.rewards["track_body_forward_velocity"] = RewardTermCfg(
    func=velocity_mdp.track_body_forward_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": 0.3},
  )

  cfg.env.episode_length_s = args.path_duration
  cfg.env.terminations["curve_path_complete"] = TerminationTermCfg(
    func=velocity_mdp.sine_path_complete,
    time_out=True,
    params={"duration": args.path_duration},
  )

  cfg.agent.experiment_name = DIRECT_EXPERIMENT_NAME
  if args.max_iterations is not None:
    cfg.agent.max_iterations = args.max_iterations
  if args.num_envs is not None:
    cfg.env.scene.num_envs = args.num_envs

  if args.resume_latest and args.resume_run is not None:
    raise ValueError("Use either --resume-latest or --resume-run, not both.")

  resume = args.resume_latest or args.resume_run is not None
  if resume:
    cfg.agent.resume = True
    if args.resume_latest:
      checkpoint_path = find_latest_checkpoint(cfg.agent.experiment_name)
      cfg.agent.load_run = checkpoint_path.parent.name
      cfg.agent.load_checkpoint = checkpoint_path.name
    else:
      cfg.agent.load_run = args.resume_run
      cfg.agent.load_checkpoint = args.resume_checkpoint
    print(
      "[INFO] Resuming gentle-curve training from "
      f"run={cfg.agent.load_run}, checkpoint={cfg.agent.load_checkpoint}"
    )

  min_radius = (
    float("inf")
    if abs(args.ang_vel_z_amplitude) < 1.0e-6
    else args.lin_vel_x / args.ang_vel_z_amplitude
  )
  print(
    "[INFO] Gentle curve command: "
    f"lin_vel_x={args.lin_vel_x}, "
    f"ang_vel_z_amplitude={args.ang_vel_z_amplitude}, "
    f"period={args.period}, min_radius≈{min_radius:.2f}m"
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
    print("[INFO] Running train_go2_curve.py with direct-run settings")
  else:
    args = tyro.cli(Go2CurveTrainConfig)
  os.chdir(PROJECT_ROOT)
  train_cfg = build_train_config(args)
  launch_training(task_id=args.task, args=train_cfg)


if __name__ == "__main__":
  main()
