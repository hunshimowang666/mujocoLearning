"""Script to play RL agent with RSL-RL."""

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from train_go2_sine import (  # noqa: E402
  DIRECT_ANG_VEL_Z_AMPLITUDE as TRAIN_SINE_ANG_VEL_Z_AMPLITUDE,
)
from train_go2_sine import DIRECT_EXPERIMENT_NAME as TRAIN_EXPERIMENT_NAME  # noqa: E402
from train_go2_sine import DIRECT_LIN_VEL_X as TRAIN_SINE_LIN_VEL_X  # noqa: E402
from train_go2_sine import DIRECT_LIN_VEL_Y as TRAIN_SINE_LIN_VEL_Y  # noqa: E402
from train_go2_sine import DIRECT_PATH_DURATION as TRAIN_SINE_PATH_DURATION  # noqa: E402
from train_go2_sine import DIRECT_PERIOD as TRAIN_SINE_PERIOD  # noqa: E402
from train_go2_sine import DIRECT_TASK as TRAIN_TASK  # noqa: E402
from train_go2_straight import (  # noqa: E402
  DIRECT_ANG_VEL_Z as TRAIN_STRAIGHT_ANG_VEL_Z,
)
from train_go2_straight import (  # noqa: E402
  DIRECT_EXPERIMENT_NAME as TRAIN_STRAIGHT_EXPERIMENT_NAME,
)
from train_go2_straight import (  # noqa: E402
  DIRECT_EPISODE_LENGTH as TRAIN_STRAIGHT_PATH_DURATION,
)
from train_go2_straight import DIRECT_LIN_VEL_X as TRAIN_STRAIGHT_LIN_VEL_X  # noqa: E402
from train_go2_straight import DIRECT_LIN_VEL_Y as TRAIN_STRAIGHT_LIN_VEL_Y  # noqa: E402


##
# Direct-run settings.
# Edit these values, then run this file directly from the IDE.
##

DIRECT_PLAY_MODE: Literal["sine", "straight"] = "sine"
# DIRECT_PLAY_MODE: Literal["sine", "straight"] = "straight"
DIRECT_PLAY_TASK = TRAIN_TASK
DIRECT_PLAY_EXPERIMENT_NAME = TRAIN_EXPERIMENT_NAME
DIRECT_PLAY_AGENT: Literal["zero", "random", "trained"] = "trained"
DIRECT_PLAY_LATEST_CHECKPOINT = True
DIRECT_PLAY_CHECKPOINT_FILE: str | None = None
DIRECT_PLAY_NUM_ENVS = 1
DIRECT_PLAY_DEVICE = "cuda:0"
DIRECT_PLAY_VIEWER: Literal["auto", "native", "viser"] = "viser"

# True keeps simulation steps from being dropped, but it can starve rendering on
# slower machines. Keep it off for smooth interactive viewing.
DIRECT_STRICT_REALTIME_STEPS = False
DIRECT_RENDER_FRAME_RATE = 10.0
DIRECT_TARGET_STEPS_PER_SECOND: float | None = 10.0

DIRECT_SHOW_SINE_PATH = True


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


class StrictRealtimeStepMixin:
  """Advance every due control step instead of dropping sim-time budget."""

  def _step_physics(self, dt: float) -> None:
    step_dt = self.env.unwrapped.step_dt
    self._sim_budget += dt * self._time_multiplier
    self._was_capped = False

    if self._sim_budget < step_dt:
      return

    self.sync_viewer_to_env()
    while self._sim_budget >= step_dt:
      if not self._execute_step():
        self._sim_budget = 0.0
        return
      self._sim_budget -= step_dt


class StrictRealtimeViserPlayViewer(StrictRealtimeStepMixin, ViserPlayViewer):
  pass


class StrictRealtimeNativeMujocoViewer(StrictRealtimeStepMixin, NativeMujocoViewer):
  pass


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
  sine_ang_vel_z_amplitude: float = 0.35
  sine_period: float = 10.0
  show_sine_path: bool = True
  sine_path_duration: float = 20.0
  sine_path_points: int = 160
  sine_path_z: float = 0.035
  show_straight_path: bool = True
  straight_path_duration: float = 20.0
  straight_path_points: int = 160
  straight_path_z: float = 0.035
  sine_fixed_start: bool = True
  sine_hide_twist_debug: bool = True
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  strict_realtime_steps: bool = False
  render_frame_rate: float = 50.0
  target_steps_per_second: float | None = None
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name
  checkpoints = sorted(
    log_root.glob("*/model_*.pt"),
    key=lambda path: (path.parent.name, int(path.stem.split("_")[-1])),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return checkpoints[-1]


def direct_play_config(task_id: str) -> PlayConfig:
  checkpoint_file = DIRECT_PLAY_CHECKPOINT_FILE

  if DIRECT_PLAY_MODE == "sine":
    if DIRECT_PLAY_AGENT == "trained" and DIRECT_PLAY_LATEST_CHECKPOINT:
      checkpoint_file = str(find_latest_checkpoint(DIRECT_PLAY_EXPERIMENT_NAME))
    return PlayConfig(
      agent=DIRECT_PLAY_AGENT,
      checkpoint_file=checkpoint_file,
      num_envs=DIRECT_PLAY_NUM_ENVS,
      device=DIRECT_PLAY_DEVICE,
      viewer=DIRECT_PLAY_VIEWER,
      strict_realtime_steps=DIRECT_STRICT_REALTIME_STEPS,
      render_frame_rate=DIRECT_RENDER_FRAME_RATE,
      target_steps_per_second=DIRECT_TARGET_STEPS_PER_SECOND,
      sine_lin_vel_x=TRAIN_SINE_LIN_VEL_X,
      sine_lin_vel_y=TRAIN_SINE_LIN_VEL_Y,
      sine_ang_vel_z_amplitude=TRAIN_SINE_ANG_VEL_Z_AMPLITUDE,
      sine_period=TRAIN_SINE_PERIOD,
      show_sine_path=DIRECT_SHOW_SINE_PATH,
      sine_path_duration=TRAIN_SINE_PATH_DURATION,
    )
  if DIRECT_PLAY_MODE == "straight":
    if DIRECT_PLAY_AGENT == "trained" and DIRECT_PLAY_LATEST_CHECKPOINT:
      checkpoint_file = str(find_latest_checkpoint(TRAIN_STRAIGHT_EXPERIMENT_NAME))
    return PlayConfig(
      agent=DIRECT_PLAY_AGENT,
      checkpoint_file=checkpoint_file,
      num_envs=DIRECT_PLAY_NUM_ENVS,
      device=DIRECT_PLAY_DEVICE,
      viewer=DIRECT_PLAY_VIEWER,
      strict_realtime_steps=DIRECT_STRICT_REALTIME_STEPS,
      render_frame_rate=DIRECT_RENDER_FRAME_RATE,
      target_steps_per_second=DIRECT_TARGET_STEPS_PER_SECOND,
      constant_lin_vel_x=TRAIN_STRAIGHT_LIN_VEL_X,
      constant_lin_vel_y=TRAIN_STRAIGHT_LIN_VEL_Y,
      constant_ang_vel_z=TRAIN_STRAIGHT_ANG_VEL_Z,
      show_straight_path=True,
      straight_path_duration=TRAIN_STRAIGHT_PATH_DURATION,
    )
  raise ValueError(f"Unsupported DIRECT_PLAY_MODE: {DIRECT_PLAY_MODE}")


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
  term.reference_pos_w = torch.zeros(env.num_envs, 2, device=env.device)
  term.reference_heading_w = torch.zeros(env.num_envs, device=env.device)

  def set_command(env_ids=None):
    if env_ids is None:
      phase = (
        env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
        / cfg.sine_period
        + phase_offset
      )
      term.vel_command_b[:, 0] = cfg.sine_lin_vel_x
      term.vel_command_b[:, 1] = cfg.sine_lin_vel_y
      term.vel_command_b[:, 2] = cfg.sine_ang_vel_z_amplitude * torch.sin(
        2.0 * torch.pi * phase
      )
      dt = env.step_dt
      term.reference_heading_w += term.vel_command_b[:, 2] * dt
      cos_h = torch.cos(term.reference_heading_w)
      sin_h = torch.sin(term.reference_heading_w)
      vel_x_w = cos_h * cfg.sine_lin_vel_x - sin_h * cfg.sine_lin_vel_y
      vel_y_w = sin_h * cfg.sine_lin_vel_x + cos_h * cfg.sine_lin_vel_y
      term.reference_pos_w[:, 0] += vel_x_w * dt
      term.reference_pos_w[:, 1] += vel_y_w * dt
    else:
      phase_offset[env_ids] = 0.0
      term.vel_command_b[env_ids, 0] = cfg.sine_lin_vel_x
      term.vel_command_b[env_ids, 1] = cfg.sine_lin_vel_y
      term.vel_command_b[env_ids, 2] = 0.0
      term.reference_pos_w[env_ids] = term.robot.data.root_link_pos_w[env_ids, :2]
      term.reference_heading_w[env_ids] = term.robot.data.heading_w[env_ids]
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
  term.reference_pos_w[:] = term.robot.data.root_link_pos_w[:, :2]
  term.reference_heading_w[:] = term.robot.data.heading_w
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


def make_straight_path_points(cfg: PlayConfig) -> np.ndarray:
  if cfg.constant_lin_vel_x is None:
    return np.zeros((0, 3), dtype=np.float32)
  if cfg.straight_path_duration <= 0.0:
    raise ValueError("--straight-path-duration must be positive.")
  if cfg.straight_path_points < 2:
    raise ValueError("--straight-path-points must be at least 2.")

  path = np.zeros((cfg.straight_path_points, 3), dtype=np.float32)
  path[:, 2] = cfg.straight_path_z
  duration = np.linspace(0.0, cfg.straight_path_duration, cfg.straight_path_points)
  path[:, 0] = cfg.constant_lin_vel_x * duration
  path[:, 1] = cfg.constant_lin_vel_y * duration
  return path


def install_straight_path_visualizer(env: ManagerBasedRlEnv, cfg: PlayConfig) -> None:
  if cfg.constant_lin_vel_x is None or not cfg.show_straight_path:
    return

  path = make_straight_path_points(cfg)
  original_update_visualizers = env.update_visualizers

  def update_visualizers_with_straight_path(visualizer):
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
          color=(1.0, 0.62, 0.1, 0.9),
        )
      visualizer.add_sphere(
        points[0],
        radius=0.06,
        color=(0.0, 0.9, 0.2, 0.95),
      )
      visualizer.add_sphere(
        points[-1],
        radius=0.06,
        color=(1.0, 0.2, 0.1, 0.95),
      )

  env.update_visualizers = update_visualizers_with_straight_path
  print(
    "[INFO] Drawing straight reference path: "
    f"duration={cfg.straight_path_duration}s, points={cfg.straight_path_points}"
  )


def configure_sine_fixed_start(env_cfg, cfg: PlayConfig) -> None:
  if (
    cfg.sine_lin_vel_x is None
    and cfg.constant_lin_vel_x is None
  ) or not cfg.sine_fixed_start:
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
  print("[INFO] Play fixed start enabled: x=0, y=0, yaw=0")


def configure_sine_debug_visualization(env_cfg, cfg: PlayConfig) -> None:
  if cfg.sine_lin_vel_x is None or not cfg.sine_hide_twist_debug:
    return

  twist_cmd = env_cfg.commands.get("twist")
  if twist_cmd is not None and hasattr(twist_cmd, "debug_vis"):
    twist_cmd.debug_vis = False
    print("[INFO] Sine play twist debug arrows hidden")


def configure_sine_terminations(env_cfg, cfg: PlayConfig) -> None:
  if cfg.sine_lin_vel_x is None:
    return

  import src.tasks.velocity.mdp as velocity_mdp

  env_cfg.episode_length_s = cfg.sine_path_duration
  env_cfg.terminations["sine_path_complete"] = TerminationTermCfg(
    func=velocity_mdp.sine_path_complete,
    time_out=True,
    params={"duration": cfg.sine_path_duration},
  )
  print(
    "[INFO] Sine play terminations enabled: "
    f"duration={cfg.sine_path_duration}s"
  )


def configure_straight_terminations(env_cfg, cfg: PlayConfig) -> None:
  if cfg.constant_lin_vel_x is None:
    return

  import src.tasks.velocity.mdp as velocity_mdp

  env_cfg.episode_length_s = cfg.straight_path_duration
  env_cfg.terminations["straight_path_complete"] = TerminationTermCfg(
    func=velocity_mdp.sine_path_complete,
    time_out=True,
    params={"duration": cfg.straight_path_duration},
  )
  print(
    "[INFO] Straight play terminations enabled: "
    f"duration={cfg.straight_path_duration}s"
  )


def apply_viewer_step_rate(viewer, env, cfg: PlayConfig) -> None:
  if cfg.target_steps_per_second is None:
    return
  if cfg.target_steps_per_second <= 0.0:
    raise ValueError("--target-steps-per-second must be positive.")

  step_dt = env.unwrapped.step_dt
  viewer._time_multiplier = cfg.target_steps_per_second * step_dt
  print(
    "[INFO] Viewer target step rate: "
    f"{cfg.target_steps_per_second:.1f} steps/s "
    f"(step_dt={step_dt:.3f}s, speed={viewer._time_multiplier:.3f}x)"
  )


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
  configure_sine_debug_visualization(env_cfg, cfg)
  if not cfg.no_terminations:
    configure_sine_terminations(env_cfg, cfg)
    configure_straight_terminations(env_cfg, cfg)

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  apply_constant_twist_command(env, cfg, device)
  apply_sine_twist_command(env, cfg)
  install_sine_path_visualizer(env, cfg)
  install_straight_path_visualizer(env, cfg)

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
    viewer_cls = (
      StrictRealtimeNativeMujocoViewer
      if cfg.strict_realtime_steps
      else NativeMujocoViewer
    )
    viewer = viewer_cls(env, policy, frame_rate=cfg.render_frame_rate)
    apply_viewer_step_rate(viewer, env, cfg)
    viewer.run()
  elif resolved_viewer == "viser":
    viewer_cls = StrictRealtimeViserPlayViewer if cfg.strict_realtime_steps else ViserPlayViewer
    viewer = viewer_cls(env, policy, frame_rate=cfg.render_frame_rate)
    apply_viewer_step_rate(viewer, env, cfg)
    viewer.run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  configure_runtime_environment()

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  os.chdir(PROJECT_ROOT)
  if len(sys.argv) == 1:
    chosen_task = DIRECT_PLAY_TASK
    args = direct_play_config(chosen_task)
    print("[INFO] Running play.py with direct-run settings")
  else:
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
