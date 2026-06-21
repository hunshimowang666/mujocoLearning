"""Train Unitree G1 velocity locomotion with MJLab GPU-parallel PPO.

Edit the direct-run settings below, then run this file directly from the IDE.
"""

from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

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

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # noqa: E402
from train import TrainConfig, launch_training  # noqa: E402


##
# Direct-run settings.
##

# Use "Unitree-G1-23Dof-Flat" here if you explicitly want the 23-DOF variant.
DIRECT_TASK = "Unitree-G1-Flat"
DIRECT_EXPERIMENT_NAME = "g1_velocity_gait_mjlab_v1"

# Random body-frame velocity command ranges. Positive x is robot forward.
DIRECT_LIN_VEL_X_RANGE = (0.2, 0.8)
DIRECT_LIN_VEL_Y_RANGE = (-0.15, 0.15)
DIRECT_ANG_VEL_Z_RANGE = (-0.30, 0.30)
DIRECT_COMMAND_RESAMPLING_TIME_RANGE = (4.0, 8.0)

# Playback command used by 04_unitree_g1_walk_play.py.
DIRECT_PLAY_LIN_VEL_X = 0.5
DIRECT_PLAY_LIN_VEL_Y = 0.0
DIRECT_PLAY_ANG_VEL_Z = 0.0

DIRECT_EPISODE_LENGTH = 20.0
DIRECT_NUM_ENVS = 2048
DIRECT_MAX_ITERATIONS = 15000
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]

# True: continue from the newest checkpoint if one exists. False: start fresh.
DIRECT_RESUME_LATEST = True

# Used only for fresh runs. Resume never deletes old networks.
DIRECT_DELETE_OLD_NETWORKS = True

GAIT_PERIOD = 0.8
GAIT_OFFSET = [0.0, 0.5]
GAIT_STANCE_THRESHOLD = 0.55


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


@dataclass(frozen=True)
class UnitreeG1WalkTrainConfig:
  task: str = DIRECT_TASK
  experiment_name: str = DIRECT_EXPERIMENT_NAME
  lin_vel_x_range: tuple[float, float] = DIRECT_LIN_VEL_X_RANGE
  lin_vel_y_range: tuple[float, float] = DIRECT_LIN_VEL_Y_RANGE
  ang_vel_z_range: tuple[float, float] = DIRECT_ANG_VEL_Z_RANGE
  command_resampling_time_range: tuple[float, float] = (
    DIRECT_COMMAND_RESAMPLING_TIME_RANGE
  )
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


def _configure_velocity_command(
  cfg: TrainConfig,
  args: UnitreeG1WalkTrainConfig,
) -> None:
  cfg.env.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=args.lin_vel_x_range,
      lin_vel_y=args.lin_vel_y_range,
      ang_vel_z=args.ang_vel_z_range,
      heading=None,
    ),
    heading_command=False,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    resampling_time_range=args.command_resampling_time_range,
  )


def build_train_config(args: UnitreeG1WalkTrainConfig) -> TrainConfig:
  cfg = TrainConfig.from_task(args.task)
  _configure_velocity_command(cfg, args)

  cfg.env.curriculum = {}
  cfg.env.events.pop("push_robot", None)
  for observation_group in ("actor", "critic"):
    group = cfg.env.observations.get(observation_group)
    if group is None:
      continue
    phase_term = group.terms.get("phase")
    if phase_term is not None:
      phase_term.params["period"] = GAIT_PERIOD
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
    cfg.env.rewards["track_linear_velocity"].weight = 1.0
    cfg.env.rewards["track_linear_velocity"].params["std"] = 0.50
  if "track_body_forward_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_body_forward_velocity"].weight = 2.0
    cfg.env.rewards["track_body_forward_velocity"].params["std"] = 0.35
  if "track_angular_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_angular_velocity"].weight = 0.5
    cfg.env.rewards["track_angular_velocity"].params["std"] = 0.50
  if "body_orientation_l2" in cfg.env.rewards:
    cfg.env.rewards["body_orientation_l2"].weight = -1.0
  if "pose" in cfg.env.rewards:
    cfg.env.rewards["pose"].weight = 0.35
  if "foot_gait" in cfg.env.rewards:
    cfg.env.rewards["foot_gait"].weight = 1.5
    cfg.env.rewards["foot_gait"].params["period"] = GAIT_PERIOD
    cfg.env.rewards["foot_gait"].params["offset"] = GAIT_OFFSET
    cfg.env.rewards["foot_gait"].params["threshold"] = GAIT_STANCE_THRESHOLD
    cfg.env.rewards["foot_gait"].params["command_threshold"] = 0.05
  if "foot_clearance" in cfg.env.rewards:
    cfg.env.rewards["foot_clearance"].weight = -0.50
    cfg.env.rewards["foot_clearance"].params["target_height"] = 0.08
    cfg.env.rewards["foot_clearance"].params["command_threshold"] = 0.05
  if "action_rate_l2" in cfg.env.rewards:
    cfg.env.rewards["action_rate_l2"].weight = -0.01
  if "foot_slip" in cfg.env.rewards:
    cfg.env.rewards["foot_slip"].weight = -0.20
    cfg.env.rewards["foot_slip"].params["command_threshold"] = 0.05
  if "body_ang_vel" in cfg.env.rewards:
    cfg.env.rewards["body_ang_vel"].weight = -0.05
  if "angular_momentum" in cfg.env.rewards:
    cfg.env.rewards["angular_momentum"].weight = -0.025
  for termination_name in (
    "lateral_deviation",
    "heading_deviation",
    "insufficient_forward_progress",
    "sine_path_complete",
    "sine_path_deviation",
  ):
    cfg.env.terminations.pop(termination_name, None)

  cfg.env.episode_length_s = args.episode_length

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
    "[INFO] G1 velocity-gait training: "
    f"lin_vel_x_range={args.lin_vel_x_range}, "
    f"lin_vel_y_range={args.lin_vel_y_range}, "
    f"ang_vel_z_range={args.ang_vel_z_range}, "
    f"gait_period={GAIT_PERIOD}, curriculum=off, push_robot=off"
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
