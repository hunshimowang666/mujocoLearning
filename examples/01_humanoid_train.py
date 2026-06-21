"""01_humanoid_train.py
======================
Train the Gymnasium Humanoid-v5 model to walk forward with MJLab/Warp
GPU-parallel physics and a clock-based gait prior.

The MJCF body/joint/motor model is copied from Gymnasium's Humanoid-v5 asset.
Only the XML world floor/light are removed so MJLab can provide the multi-env
terrain.  Run this file directly from VS Code.
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

DIRECT_TASK = "Gym-HumanoidV5-Flat"
DIRECT_EXPERIMENT_NAME = "humanoid_v5_mjlab_cpg_walk_v1"

DIRECT_LIN_VEL_X = 0.6
DIRECT_LIN_VEL_Y = 0.0
DIRECT_ANG_VEL_Z = 0.0
DIRECT_EPISODE_LENGTH = 20.0

DIRECT_NUM_ENVS = 4096
DIRECT_MAX_ITERATIONS = 10000
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]

# False: start a fresh run. True: continue from the newest checkpoint.
DIRECT_RESUME_LATEST = True

# Used only for fresh runs. Resume never deletes old networks.
DIRECT_DELETE_OLD_NETWORKS = True


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


def find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = MJLAB_ROOT / "logs" / "rsl_rl" / experiment_name
  checkpoints = sorted(
    log_root.glob("*/model_*.pt"),
    key=lambda path: (path.parent.name, int(path.stem.split("_")[-1])),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return checkpoints[-1]


@dataclass(frozen=True)
class HumanoidV5MjlabTrainConfig:
  task: str = DIRECT_TASK
  lin_vel_x: float = DIRECT_LIN_VEL_X
  lin_vel_y: float = DIRECT_LIN_VEL_Y
  ang_vel_z: float = DIRECT_ANG_VEL_Z
  episode_length: float = DIRECT_EPISODE_LENGTH
  num_envs: int | None = DIRECT_NUM_ENVS
  max_iterations: int | None = DIRECT_MAX_ITERATIONS
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  delete_old_networks: bool = DIRECT_DELETE_OLD_NETWORKS
  resume_latest: bool = DIRECT_RESUME_LATEST
  resume_run: str | None = None
  resume_checkpoint: str = "model_.*.pt"


def direct_train_config() -> HumanoidV5MjlabTrainConfig:
  return HumanoidV5MjlabTrainConfig(
    task=DIRECT_TASK,
    lin_vel_x=DIRECT_LIN_VEL_X,
    lin_vel_y=DIRECT_LIN_VEL_Y,
    ang_vel_z=DIRECT_ANG_VEL_Z,
    episode_length=DIRECT_EPISODE_LENGTH,
    num_envs=DIRECT_NUM_ENVS,
    max_iterations=DIRECT_MAX_ITERATIONS,
    gpu_ids=DIRECT_GPU_IDS,
    delete_old_networks=DIRECT_DELETE_OLD_NETWORKS,
    resume_latest=DIRECT_RESUME_LATEST,
  )


def build_train_config(args: HumanoidV5MjlabTrainConfig) -> TrainConfig:
  cfg = TrainConfig.from_task(args.task)

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

  cfg.env.curriculum = {}
  cfg.env.events.pop("push_robot", None)
  for reward_name in (
    "track_world_forward_velocity",
    "track_world_lateral_velocity_zero",
    "lateral_position",
    "heading",
    "sine_path_position",
    "sine_path_position_l2",
    "sine_path_tangent_velocity",
    "sine_path_heading",
    "sine_path_heading_l2",
    "sine_path_lateral_velocity_l2",
  ):
    cfg.env.rewards.pop(reward_name, None)
  if "track_forward_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_forward_velocity"].weight = 1.5
    cfg.env.rewards["track_forward_velocity"].params["std"] = 0.45
  if "track_yaw_velocity" in cfg.env.rewards:
    cfg.env.rewards["track_yaw_velocity"].weight = 0.5
    cfg.env.rewards["track_yaw_velocity"].params["std"] = 0.50
  if "track_lateral_velocity_zero" in cfg.env.rewards:
    cfg.env.rewards["track_lateral_velocity_zero"].weight = 0.4
    cfg.env.rewards["track_lateral_velocity_zero"].params["std"] = 0.35
  if "alternating_foot_placement" in cfg.env.rewards:
    cfg.env.rewards["alternating_foot_placement"].weight = 2.5
  if "foot_fore_aft_separation" in cfg.env.rewards:
    cfg.env.rewards["foot_fore_aft_separation"].weight = 1.2
  if "alternating_foot_velocity" in cfg.env.rewards:
    cfg.env.rewards["alternating_foot_velocity"].weight = 1.0
  if "swing_foot_clearance" in cfg.env.rewards:
    cfg.env.rewards["swing_foot_clearance"].weight = 0.9
  if "foot_lateral_width" in cfg.env.rewards:
    cfg.env.rewards["foot_lateral_width"].weight = 0.8
  if "leg_motion_imbalance" in cfg.env.rewards:
    cfg.env.rewards["leg_motion_imbalance"].weight = -0.8
  if "reference_gait" in cfg.env.rewards:
    cfg.env.rewards["reference_gait"].weight = 6.0
    cfg.env.rewards["reference_gait"].params["period"] = 0.85
    cfg.env.rewards["reference_gait"].params["hip_swing"] = 0.28
    cfg.env.rewards["reference_gait"].params["hip_offset"] = -0.12
    cfg.env.rewards["reference_gait"].params["knee_stance"] = -0.16
    cfg.env.rewards["reference_gait"].params["knee_swing"] = 0.38
    cfg.env.rewards["reference_gait"].params["std"] = 0.28
  if "upper_body_pose" in cfg.env.rewards:
    cfg.env.rewards["upper_body_pose"].weight = -0.8
  if "alive" in cfg.env.rewards:
    cfg.env.rewards["alive"].weight = 2.0
  if "upright" in cfg.env.rewards:
    cfg.env.rewards["upright"].weight = -10.0
  if "height" in cfg.env.rewards:
    cfg.env.rewards["height"].weight = -16.0
  if "deep_knee_bend" in cfg.env.rewards:
    cfg.env.rewards["deep_knee_bend"].weight = -4.0
  if "control" in cfg.env.rewards:
    cfg.env.rewards["control"].weight = -0.025
  if "action_rate" in cfg.env.rewards:
    cfg.env.rewards["action_rate"].weight = -0.01
  for termination_name in (
    "heading_deviation",
    "lateral_deviation",
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

  cfg.env.episode_length_s = args.episode_length
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
      try:
        checkpoint_path = find_latest_checkpoint(cfg.agent.experiment_name)
      except FileNotFoundError:
        print("[INFO] No Humanoid-v5 MJLab checkpoint found; starting a fresh run.")
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
        "[INFO] Resuming Humanoid-v5 MJLab training from "
        f"run={cfg.agent.load_run}, checkpoint={cfg.agent.load_checkpoint}"
      )

  print(
    "[INFO] Humanoid-v5 MJLab gait-walk command: "
    f"lin_vel_x={args.lin_vel_x}, lin_vel_y={args.lin_vel_y}, "
    f"ang_vel_z={args.ang_vel_z}, "
    f"num_envs={args.num_envs}"
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
    print("[INFO] Running 01_humanoid_train.py with Humanoid-v5 MJLab settings")
  else:
    args = tyro.cli(HumanoidV5MjlabTrainConfig)

  os.chdir(MJLAB_ROOT)
  train_cfg = build_train_config(args)
  launch_training(task_id=args.task, args=train_cfg)


if __name__ == "__main__":
  main()
