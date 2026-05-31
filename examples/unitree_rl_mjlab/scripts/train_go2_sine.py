"""Train Unitree Go2 to follow a sine turning command.

The command keeps forward velocity constant and drives yaw velocity with a sine
wave, so the robot learns to walk an S-shaped path on flat ground.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import tyro

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import TrainConfig, launch_training

import src.tasks.velocity.mdp as velocity_mdp
from src.tasks.velocity.mdp.sine_velocity_command import SineVelocityCommandCfg


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
class Go2SineTrainConfig:
  task: str = "Unitree-Go2-Flat"
  lin_vel_x: float = 0.5
  lin_vel_y: float = 0.0
  ang_vel_z_amplitude: float = 0.6
  period: float = 5.0
  randomize_phase: bool = True
  path_reward_weight: float = 8.0
  path_reward_std: float = 0.4
  path_reward_late_weight: float = 6.0
  path_reward_ramp_duration: float = 15.0
  linear_velocity_reward_weight: float = 0.4
  angular_velocity_reward_weight: float = 0.4
  num_envs: int | None = None
  max_iterations: int | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  delete_old_networks: bool = True
  resume_latest: bool = False
  resume_run: str | None = None
  resume_checkpoint: str = "model_.*.pt"


def build_train_config(args: Go2SineTrainConfig) -> TrainConfig:
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
  cfg.env.rewards["sine_path_tracking"] = RewardTermCfg(
    func=velocity_mdp.sine_path_tracking,
    weight=args.path_reward_weight,
    params={
      "command_name": "twist",
      "std": args.path_reward_std,
      "late_weight": args.path_reward_late_weight,
      "ramp_duration": args.path_reward_ramp_duration,
    },
  )
  cfg.env.rewards["track_linear_velocity"].weight = args.linear_velocity_reward_weight
  cfg.env.rewards["track_angular_velocity"].weight = args.angular_velocity_reward_weight

  cfg.agent.experiment_name = "go2_sine_velocity"
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
      "[INFO] Resuming sine training from "
      f"run={cfg.agent.load_run}, checkpoint={cfg.agent.load_checkpoint}"
    )

  return replace(
    cfg,
    clear_old_logs=args.delete_old_networks and not resume,
    gpu_ids=args.gpu_ids,
  )


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  args = tyro.cli(Go2SineTrainConfig)
  project_root = Path(__file__).resolve().parents[1]
  os.chdir(project_root)
  train_cfg = build_train_config(args)
  launch_training(task_id=args.task, args=train_cfg)


if __name__ == "__main__":
  main()
