"""MDP terms for the Gymnasium Humanoid-v5 MJLab task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def gait_phase(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Sin/cos gait clock used by the policy for left-right alternation."""
  global_phase = (env.episode_length_buf.to(torch.float32) * env.step_dt) / period
  phase = torch.stack(
    (
      torch.sin(2.0 * torch.pi * global_phase),
      torch.cos(2.0 * torch.pi * global_phase),
    ),
    dim=1,
  )
  command = env.command_manager.get_command(command_name)
  active = torch.linalg.norm(command, dim=1) > command_threshold
  return torch.where(active.unsqueeze(1), phase, torch.zeros_like(phase))


def _body_positions_in_root_frame(
  asset: Entity,
  body_ids: list[int] | slice,
) -> torch.Tensor:
  body_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
  rel_pos_w = body_pos_w - asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_pos_w.shape[1], -1
  )
  return quat_apply_inverse(
    root_quat_w.reshape(-1, 4),
    rel_pos_w.reshape(-1, 3),
  ).reshape_as(rel_pos_w)


def _body_linear_velocities_in_root_frame(
  asset: Entity,
  body_ids: list[int] | slice,
) -> torch.Tensor:
  body_vel_w = asset.data.body_link_lin_vel_w[:, body_ids, :]
  rel_vel_w = body_vel_w - asset.data.root_link_lin_vel_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_vel_w.shape[1], -1
  )
  return quat_apply_inverse(
    root_quat_w.reshape(-1, 4),
    rel_vel_w.reshape(-1, 3),
  ).reshape_as(rel_vel_w)


def _command_active(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  return (torch.linalg.norm(command, dim=1) > command_threshold).float()


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def _root_xy_relative_to_env_origin(
  env: ManagerBasedRlEnv,
  asset: Entity,
) -> torch.Tensor:
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None:
    return asset.data.root_link_pos_w[:, :2]
  return asset.data.root_link_pos_w[:, :2] - env_origins[:, :2]


def track_forward_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  target = command[:, 0]
  actual = asset.data.root_link_lin_vel_b[:, 0]
  return torch.exp(-torch.square(target - actual) / std**2)


def track_world_forward_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward moving along world +X, not merely along whichever way the torso faces."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  target = command[:, 0]
  actual = asset.data.root_link_lin_vel_w[:, 0]
  return torch.exp(-torch.square(target - actual) / std**2)


def track_lateral_velocity_zero(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  lateral = asset.data.root_link_lin_vel_b[:, 1]
  return torch.exp(-torch.square(lateral) / std**2)


def track_world_lateral_velocity_zero(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping world-Y drift small so the humanoid follows the forward line."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral = asset.data.root_link_lin_vel_w[:, 1]
  return torch.exp(-torch.square(lateral) / std**2)


def lateral_position_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize drifting away from the world +X reference line."""
  asset: Entity = env.scene[asset_cfg.name]
  rel_xy = _root_xy_relative_to_env_origin(env, asset)
  return torch.square(rel_xy[:, 1])


def track_yaw_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  target = command[:, 2]
  actual = asset.data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(target - actual) / std**2)


def alternating_foot_placement(
  env: ManagerBasedRlEnv,
  period: float,
  stride_length: float,
  std: float,
  command_name: str,
  command_threshold: float,
  foot_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward right/left feet taking opposite fore-aft positions over a gait clock."""
  asset: Entity = env.scene[foot_body_cfg.name]
  foot_pos_b = _body_positions_in_root_frame(asset, foot_body_cfg.body_ids)
  foot_x = foot_pos_b[:, :, 0]
  phase = 2.0 * torch.pi * (
    env.episode_length_buf.to(torch.float32) * env.step_dt / period
  )
  target = torch.stack(
    (
      stride_length * torch.sin(phase),
      -stride_length * torch.sin(phase),
    ),
    dim=1,
  )
  error = torch.mean(torch.square(foot_x - target), dim=1)
  reward = torch.exp(-error / std**2)
  return reward * _command_active(env, command_name, command_threshold)


def foot_fore_aft_separation(
  env: ManagerBasedRlEnv,
  target_separation: float,
  std: float,
  command_name: str,
  command_threshold: float,
  foot_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward the two feet not staying side-by-side while walking."""
  asset: Entity = env.scene[foot_body_cfg.name]
  foot_pos_b = _body_positions_in_root_frame(asset, foot_body_cfg.body_ids)
  foot_x = foot_pos_b[:, :, 0]
  separation = torch.abs(foot_x[:, 0] - foot_x[:, 1])
  shortfall = torch.clamp(target_separation - separation, min=0.0)
  reward = torch.exp(-torch.square(shortfall) / std**2)
  return reward * _command_active(env, command_name, command_threshold)


def foot_lateral_width_target(
  env: ManagerBasedRlEnv,
  target_width: float,
  std: float,
  command_name: str,
  command_threshold: float,
  foot_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward keeping a reasonable left-right stance instead of crossing/kneeling."""
  asset: Entity = env.scene[foot_body_cfg.name]
  foot_pos_b = _body_positions_in_root_frame(asset, foot_body_cfg.body_ids)
  foot_y = foot_pos_b[:, :, 1]
  width = torch.abs(foot_y[:, 0] - foot_y[:, 1])
  reward = torch.exp(-torch.square(width - target_width) / std**2)
  return reward * _command_active(env, command_name, command_threshold)


def alternating_foot_velocity(
  env: ManagerBasedRlEnv,
  period: float,
  stride_length: float,
  std: float,
  command_name: str,
  command_threshold: float,
  foot_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward right/left feet moving in opposite fore-aft directions."""
  asset: Entity = env.scene[foot_body_cfg.name]
  foot_vel_b = _body_linear_velocities_in_root_frame(asset, foot_body_cfg.body_ids)
  foot_vx = foot_vel_b[:, :, 0]
  phase = 2.0 * torch.pi * (
    env.episode_length_buf.to(torch.float32) * env.step_dt / period
  )
  target_speed = stride_length * (2.0 * torch.pi / period) * torch.cos(phase)
  target = torch.stack((target_speed, -target_speed), dim=1)
  error = torch.mean(torch.square(foot_vx - target), dim=1)
  reward = torch.exp(-error / std**2)
  return reward * _command_active(env, command_name, command_threshold)


def swing_foot_clearance(
  env: ManagerBasedRlEnv,
  period: float,
  target_clearance: float,
  stance_fraction: float,
  command_name: str,
  command_threshold: float,
  foot_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward only the scheduled swing leg for lifting above the stance foot height."""
  asset: Entity = env.scene[foot_body_cfg.name]
  foot_pos_w = asset.data.body_link_pos_w[:, foot_body_cfg.body_ids, :]
  foot_z = foot_pos_w[:, :, 2]
  stance_height = torch.min(foot_z, dim=1, keepdim=True).values
  relative_height = torch.clamp(
    (foot_z - stance_height) / target_clearance,
    min=0.0,
    max=1.0,
  )
  global_phase = (env.episode_length_buf.to(torch.float32) * env.step_dt) / period
  leg_phase = torch.stack((global_phase, global_phase + 0.5), dim=1) % 1.0
  is_swing = leg_phase >= stance_fraction
  reward = torch.mean(relative_height * is_swing.float(), dim=1)
  return reward * _command_active(env, command_name, command_threshold)


def bilateral_leg_motion_imbalance(
  env: ManagerBasedRlEnv,
  leg_joint_cfg: SceneEntityCfg,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Penalize one leg doing nearly all the work while the other stays passive."""
  asset: Entity = env.scene[leg_joint_cfg.name]
  joint_vel = torch.abs(asset.data.joint_vel[:, leg_joint_cfg.joint_ids])
  half = joint_vel.shape[1] // 2
  right_motion = torch.mean(joint_vel[:, :half], dim=1)
  left_motion = torch.mean(joint_vel[:, half:], dim=1)
  imbalance = torch.square(right_motion - left_motion) / (
    right_motion + left_motion + 1.0e-4
  )
  return imbalance * _command_active(env, command_name, command_threshold)


def root_height_target_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def heading_zero_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize yaw drift away from world +X."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(_wrap_to_pi(asset.data.heading_w))


def heading_over_limit(
  env: ManagerBasedRlEnv,
  max_heading_error: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if the torso yaw turns too far away from world +X."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(_wrap_to_pi(asset.data.heading_w)) > max_heading_error


def lateral_position_over_limit(
  env: ManagerBasedRlEnv,
  max_lateral_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if the root drifts too far sideways from the straight path."""
  asset: Entity = env.scene[asset_cfg.name]
  rel_xy = _root_xy_relative_to_env_origin(env, asset)
  return torch.abs(rel_xy[:, 1]) > max_lateral_distance


def deep_knee_bend_l2(
  env: ManagerBasedRlEnv,
  max_bend: float,
  knee_joint_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize crouching/kneeling while still allowing normal knee swing."""
  asset: Entity = env.scene[knee_joint_cfg.name]
  knee_angle = torch.abs(asset.data.joint_pos[:, knee_joint_cfg.joint_ids])
  excess = torch.clamp(knee_angle - max_bend, min=0.0)
  return torch.mean(torch.square(excess), dim=1)


def upright_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1)
