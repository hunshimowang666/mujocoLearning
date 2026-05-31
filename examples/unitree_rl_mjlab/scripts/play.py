"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  hide_motion_reference: bool = False
  constant_lin_vel_x: float | None = None
  constant_lin_vel_y: float = 0.0
  constant_ang_vel_z: float = 0.0
  sine_lin_vel_x: float | None = None
  sine_lin_vel_y: float = 0.0
  sine_ang_vel_z_amplitude: float = 0.6
  sine_period: float = 5.0
  show_sine_path: bool = True
  sine_path_duration: float = 20.0
  sine_path_points: int = 160
  sine_path_z: float = 0.035
  sine_fixed_start: bool = True
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def apply_constant_twist_command(env: ManagerBasedRlEnv, cfg: PlayConfig, device: str):
  if cfg.constant_lin_vel_x is None:
    return
  if cfg.sine_lin_vel_x is not None:
    raise ValueError("Use either --constant-lin-vel-x or --sine-lin-vel-x, not both.")

  term = env.command_manager.get_term("twist")
  if not hasattr(term, "vel_command_b"):
    raise ValueError("--constant-lin-vel-x requires a velocity task with a twist command.")

  command = torch.tensor(
    [cfg.constant_lin_vel_x, cfg.constant_lin_vel_y, cfg.constant_ang_vel_z],
    device=device,
  )

  def set_command(env_ids=None):
    if env_ids is None:
      term.vel_command_b[:, :] = command
    else:
      term.vel_command_b[env_ids, :] = command
    if hasattr(term, "is_standing_env"):
      term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
      term.is_heading_env[:] = False

  def resample_command(env_ids):
    set_command(env_ids)

  def update_command():
    set_command()

  term._resample_command = resample_command
  term._update_command = update_command
  set_command()
  print(
    "[INFO] Using constant twist command: "
    f"lin_vel_x={cfg.constant_lin_vel_x}, "
    f"lin_vel_y={cfg.constant_lin_vel_y}, "
    f"ang_vel_z={cfg.constant_ang_vel_z}"
  )


def apply_sine_twist_command(env: ManagerBasedRlEnv, cfg: PlayConfig):
  if cfg.sine_lin_vel_x is None:
    return
  if cfg.sine_period <= 0.0:
    raise ValueError("--sine-period must be positive.")

  term = env.command_manager.get_term("twist")
  if not hasattr(term, "vel_command_b"):
    raise ValueError("--sine-lin-vel-x requires a velocity task with a twist command.")

  phase_offset = torch.zeros(env.num_envs, device=env.device)

  def set_command(env_ids=None):
    if env_ids is None:
      phase = (
        env.episode_length_buf.to(dtype=torch.float32)
        * env.step_dt
        / cfg.sine_period
        + phase_offset
      )
      term.vel_command_b[:, 0] = cfg.sine_lin_vel_x
      term.vel_command_b[:, 1] = cfg.sine_lin_vel_y
      term.vel_command_b[:, 2] = cfg.sine_ang_vel_z_amplitude * torch.sin(
        2.0 * torch.pi * phase
      )
    else:
      phase_offset[env_ids] = 0.0
      term.vel_command_b[env_ids, 0] = cfg.sine_lin_vel_x
      term.vel_command_b[env_ids, 1] = cfg.sine_lin_vel_y
      term.vel_command_b[env_ids, 2] = 0.0
    if hasattr(term, "is_standing_env"):
      term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
      term.is_heading_env[:] = False

  def resample_command(env_ids):
    set_command(env_ids)

  def update_command():
    set_command()

  term._resample_command = resample_command
  term._update_command = update_command
  set_command()
  print(
    "[INFO] Using sine twist command: "
    f"lin_vel_x={cfg.sine_lin_vel_x}, "
    f"lin_vel_y={cfg.sine_lin_vel_y}, "
    f"ang_vel_z_amplitude={cfg.sine_ang_vel_z_amplitude}, "
    f"period={cfg.sine_period}"
  )


def make_sine_path_points(cfg: PlayConfig) -> np.ndarray:
  if cfg.sine_lin_vel_x is None:
    return np.zeros((0, 3), dtype=np.float32)
  if cfg.sine_path_duration <= 0.0:
    raise ValueError("--sine-path-duration must be positive.")
  if cfg.sine_path_points < 2:
    raise ValueError("--sine-path-points must be at least 2.")

  path = np.zeros((cfg.sine_path_points, 3), dtype=np.float32)
  path[:, 2] = cfg.sine_path_z
  heading = 0.0
  dt = cfg.sine_path_duration / (cfg.sine_path_points - 1)
  for i in range(1, cfg.sine_path_points):
    t = (i - 1) * dt
    yaw_rate = cfg.sine_ang_vel_z_amplitude * np.sin(2.0 * np.pi * t / cfg.sine_period)
    heading += yaw_rate * dt
    c = np.cos(heading)
    s = np.sin(heading)
    vx_w = c * cfg.sine_lin_vel_x - s * cfg.sine_lin_vel_y
    vy_w = s * cfg.sine_lin_vel_x + c * cfg.sine_lin_vel_y
    path[i, 0] = path[i - 1, 0] + vx_w * dt
    path[i, 1] = path[i - 1, 1] + vy_w * dt
  return path


def install_sine_path_visualizer(env: ManagerBasedRlEnv, cfg: PlayConfig) -> None:
  if cfg.sine_lin_vel_x is None or not cfg.show_sine_path:
    return

  path = make_sine_path_points(cfg)
  original_update_visualizers = env.update_visualizers

  def update_visualizers_with_sine_path(visualizer):
    original_update_visualizers(visualizer)
    env_origins = getattr(env.scene, "env_origins", None)
    for env_idx in visualizer.get_env_indices(env.num_envs):
      if env_origins is None:
        origin = np.zeros(3, dtype=np.float32)
      else:
        origin = env_origins[env_idx].detach().cpu().numpy()
      points = path + origin
      for start, end in zip(points[:-1], points[1:]):
        visualizer.add_cylinder(
          start,
          end,
          radius=0.018,
          color=(0.1, 0.75, 1.0, 0.9),
        )
      visualizer.add_sphere(
        points[0],
        radius=0.06,
        color=(0.0, 0.9, 0.2, 0.95),
      )
      visualizer.add_sphere(
        points[-1],
        radius=0.06,
        color=(1.0, 0.75, 0.1, 0.95),
      )

  env.update_visualizers = update_visualizers_with_sine_path
  print(
    "[INFO] Drawing sine reference path: "
    f"duration={cfg.sine_path_duration}s, points={cfg.sine_path_points}"
  )


def configure_sine_fixed_start(env_cfg, cfg: PlayConfig) -> None:
  if cfg.sine_lin_vel_x is None or not cfg.sine_fixed_start:
    return

  reset_base = env_cfg.events.get("reset_base")
  if reset_base is None:
    return

  pose_range = reset_base.params.get("pose_range")
  if pose_range is None:
    return

  pose_range["x"] = (0.0, 0.0)
  pose_range["y"] = (0.0, 0.0)
  pose_range["yaw"] = (0.0, 0.0)
  print("[INFO] Sine play fixed start enabled: x=0, y=0, yaw=0")


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if cfg.hide_motion_reference:
      motion_cmd.debug_vis = False

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width
  configure_sine_fixed_start(env_cfg, cfg)

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  apply_constant_twist_command(env, cfg, device)
  apply_sine_twist_command(env, cfg)
  install_sine_path_visualizer(env, cfg)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
