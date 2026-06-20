"""Train Unitree G1 to walk along a sine curve with MJLab GPU-parallel PPO.

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
from src.tasks.velocity.mdp.sine_velocity_command import SineVelocityCommandCfg  # noqa: E402


##
# Direct-run settings.
##

# Use "Unitree-G1-23Dof-Flat" here if you explicitly want the 23-DOF variant.
DIRECT_TASK = "Unitree-G1-Flat"
DIRECT_EXPERIMENT_NAME = "g1_humanoid_velocity_mjlab_sine_v1"

# Body-frame command. Positive x is the robot's forward direction.
DIRECT_LIN_VEL_X = 0.6
DIRECT_LIN_VEL_Y = 0.0
DIRECT_ANG_VEL_Z_AMPLITUDE = 0.08
DIRECT_PERIOD = 16.0
DIRECT_SINE_WARMUP_DURATION = 2.0
DIRECT_RANDOMIZE_PHASE = False

DIRECT_PATH_DURATION = 12.0
DIRECT_MAX_PATH_ERROR = 2.0
DIRECT_PATH_ERROR_GRACE_DURATION = 5.0
DIRECT_NUM_ENVS = 2048
DIRECT_MAX_ITERATIONS = 10000
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]

# True: continue from the newest checkpoint if one exists. False: start fresh.
DIRECT_RESUME_LATEST = True

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
  ang_vel_z_amplitude: float = DIRECT_ANG_VEL_Z_AMPLITUDE
  period: float = DIRECT_PERIOD
  sine_warmup_duration: float = DIRECT_SINE_WARMUP_DURATION
  randomize_phase: bool = DIRECT_RANDOMIZE_PHASE
  path_duration: float = DIRECT_PATH_DURATION
  max_path_error: float | None = DIRECT_MAX_PATH_ERROR
  path_error_grace_duration: float = DIRECT_PATH_ERROR_GRACE_DURATION
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


def _configure_sine_command(
  cfg: TrainConfig,
  args: UnitreeG1WalkTrainConfig,
) -> None:
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
    warmup_duration=args.sine_warmup_duration,
    randomize_phase=args.randomize_phase,
  )


def build_train_config(args: UnitreeG1WalkTrainConfig) -> TrainConfig:
  cfg = TrainConfig.from_task(args.task)
  _configure_sine_command(cfg, args)

  cfg.env.curriculum = {}
  cfg.env.events.pop("push_robot", None)
  for reward_name in (
    "track_world_forward_velocity",
    "track_world_lateral_velocity_zero",
    "lateral_position",
    "heading_zero",
    "sine_path_position",
    "sine_path_position_l2",
    "sine_path_tangent_velocity",
    "sine_path_heading",
    "sine_path_heading_l2",
    "sine_path_lateral_velocity_l2",
  ):
    cfg.env.rewards.pop(reward_name, None)
  if "track_linear_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_linear_velocity"].weight = 2.0
    cfg.env.rewards["track_linear_velocity"].params["std"] = 0.35
  if "track_body_forward_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_body_forward_velocity"].weight = 7.0
    cfg.env.rewards["track_body_forward_velocity"].params["std"] = 0.30
  if "track_angular_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_angular_velocity"].weight = 1.8
    cfg.env.rewards["track_angular_velocity"].params["std"] = 0.45
  if "body_orientation_l2" in cfg.env.rewards:
    cfg.env.rewards["body_orientation_l2"].weight = -1.5
  if "pose" in cfg.env.rewards:
    cfg.env.rewards["pose"].weight = 0.5
  if "foot_gait" in cfg.env.rewards:
    cfg.env.rewards["foot_gait"].weight = 1.5
  for termination_name in (
    "lateral_deviation",
    "heading_deviation",
    "insufficient_forward_progress",
  ):
    cfg.env.terminations.pop(termination_name, None)
  reset_base = cfg.env.events.get("reset_base")
  if reset_base is not None:
    pose_range = reset_base.params.get("pose_range")
    if pose_range is not None:
      pose_range["x"] = (0.0, 0.0)
      pose_range["y"] = (0.0, 0.0)
      pose_range["yaw"] = (0.0, 0.0)

  cfg.env.episode_length_s = args.path_duration
  cfg.env.terminations["sine_path_complete"] = TerminationTermCfg(
    func=velocity_mdp.sine_path_complete,
    time_out=True,
    params={"duration": args.path_duration},
  )
  if args.max_path_error is not None:
    cfg.env.terminations["sine_path_deviation"] = TerminationTermCfg(
      func=velocity_mdp.sine_path_deviation_over_limit,
      params={
        "command_name": "twist",
        "max_path_error": args.max_path_error,
        "grace_duration": args.path_error_grace_duration,
      },
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
  print(
    "[INFO] G1 sine-path training: "
    f"lin_vel_x={args.lin_vel_x}, "
    f"ang_vel_z_amplitude={args.ang_vel_z_amplitude}, "
    f"period={args.period}, curriculum=off, push_robot=off"
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
