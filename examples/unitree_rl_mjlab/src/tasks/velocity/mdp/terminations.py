from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def sine_path_complete(
  env: ManagerBasedRlEnv,
  duration: float,
  warmup_duration: float = 0.0,
) -> torch.Tensor:
  """Finish the episode after the planned sine path duration."""
  elapsed = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  return elapsed >= duration + warmup_duration


def sine_path_deviation_over_limit(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_path_error: float,
  grace_duration: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the root is too far away from the live sine-path reference."""
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  reference_pos_w = getattr(command_term, "reference_pos_w", None)
  if reference_pos_w is None:
    raise AttributeError(
      f"Command '{command_name}' does not expose reference_pos_w; "
      "sine_path_deviation_over_limit requires SineVelocityCommand."
    )

  elapsed = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  path_error = torch.linalg.norm(
    asset.data.root_link_pos_w[:, :2] - reference_pos_w,
    dim=1,
  )
  return (elapsed > grace_duration) & (path_error > max_path_error)


def insufficient_forward_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_progress_fraction: float,
  grace_duration: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when commanded-forward tasks make too little world-X progress."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  elapsed = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  expected_progress = (
    torch.clamp(command[:, 0], min=0.0) * elapsed * min_progress_fraction
  )
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None:
    forward_progress = asset.data.root_link_pos_w[:, 0]
  else:
    forward_progress = asset.data.root_link_pos_w[:, 0] - env_origins[:, 0]

  return (elapsed > grace_duration) & (forward_progress < expected_progress)


def lateral_position_over_limit(
  env: ManagerBasedRlEnv,
  max_lateral_distance: float,
  grace_duration: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when root position drifts too far away from the world-X line."""
  asset: Entity = env.scene[asset_cfg.name]
  elapsed = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None:
    lateral_position = asset.data.root_link_pos_w[:, 1]
  else:
    lateral_position = asset.data.root_link_pos_w[:, 1] - env_origins[:, 1]
  return (elapsed > grace_duration) & (
    torch.abs(lateral_position) > max_lateral_distance
  )


def heading_over_limit(
  env: ManagerBasedRlEnv,
  max_heading_error: float,
  grace_duration: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot yaw deviates too far from world +X."""
  asset: Entity = env.scene[asset_cfg.name]
  elapsed = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  heading = torch.atan2(torch.sin(asset.data.heading_w), torch.cos(asset.data.heading_w))
  return (elapsed > grace_duration) & (torch.abs(heading) > max_heading_error)
