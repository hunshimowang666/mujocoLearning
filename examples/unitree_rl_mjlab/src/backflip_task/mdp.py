"""MDP terms for a GPU-parallel Go2 backflip task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _robot(env: ManagerBasedRlEnv, asset_cfg=None):
  return env.scene[asset_cfg.name if asset_cfg is not None else "robot"]


def elapsed_time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf.to(dtype=torch.float32) * env.step_dt


def phase_scalar(env: ManagerBasedRlEnv, duration: float) -> torch.Tensor:
  return torch.clamp(elapsed_time(env) / max(duration, 1e-6), 0.0, 1.0).unsqueeze(-1)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  shape = vec.shape
  quat = quat.reshape(-1, 4)
  vec = vec.reshape(-1, 3)
  xyz = quat[:, 1:]
  t = xyz.cross(vec, dim=-1) * 2.0
  return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def _base_x_axis_w(asset) -> torch.Tensor:
  forward_b = torch.zeros(
    (asset.data.root_link_quat_w.shape[0], 3),
    device=asset.data.root_link_quat_w.device,
  )
  forward_b[:, 0] = 1.0
  return _quat_apply(asset.data.root_link_quat_w, forward_b)


def flip_angle(env: ManagerBasedRlEnv, asset_cfg=None) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  base_x = _base_x_axis_w(asset)
  return torch.atan2(-base_x[:, 2], base_x[:, 0])


def flip_angle_obs(env: ManagerBasedRlEnv, asset_cfg=None) -> torch.Tensor:
  angle = flip_angle(env, asset_cfg)
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def target_flip_angle(env: ManagerBasedRlEnv) -> torch.Tensor:
  t = elapsed_time(env)
  target = torch.zeros_like(t)
  target = torch.where(t >= 0.65, torch.full_like(target, torch.pi / 6.0), target)
  target = torch.where(t >= 0.90, torch.full_like(target, torch.pi / 2.0), target)
  target = torch.where(t >= 1.20, torch.full_like(target, 0.9444 * torch.pi), target)
  target = torch.where(t >= 1.55, torch.full_like(target, -0.5 * torch.pi), target)
  target = torch.where(t >= 1.90, torch.zeros_like(target), target)
  return target


def target_flip_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  angle = target_flip_angle(env)
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def body_height(env: ManagerBasedRlEnv, asset_cfg=None) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def base_linear_velocity_w(env: ManagerBasedRlEnv, asset_cfg=None) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return asset.data.root_link_lin_vel_w


def base_angular_velocity_w(env: ManagerBasedRlEnv, asset_cfg=None) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return asset.data.root_link_ang_vel_w


def feet_contact_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().flatten(start_dim=1)


def nonfoot_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > 10.0).any(dim=-1).float().sum(dim=-1)
  assert data.found is not None
  return (data.found > 0).float().flatten(start_dim=1).sum(dim=-1)


def reference_joint_pose(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  t = elapsed_time(env).unsqueeze(-1)

  pose = default_joint_pos.clone()
  crouch = default_joint_pos.clone()
  crouch[:, [1, 4, 7, 10]] = torch.deg2rad(torch.tensor(72.0, device=env.device))
  crouch[:, [2, 5, 8, 11]] = torch.deg2rad(torch.tensor(-138.0, device=env.device))

  thrust = default_joint_pos.clone()
  thrust[:, [1, 4]] = torch.deg2rad(torch.tensor(76.0, device=env.device))
  thrust[:, [2, 5]] = torch.deg2rad(torch.tensor(-142.0, device=env.device))
  thrust[:, [7, 10]] = torch.deg2rad(torch.tensor(15.0, device=env.device))
  thrust[:, [8, 11]] = torch.deg2rad(torch.tensor(-86.0, device=env.device))

  tuck = default_joint_pos.clone()
  tuck[:, [1, 4, 7, 10]] = torch.deg2rad(torch.tensor(96.0, device=env.device))
  tuck[:, [2, 5, 8, 11]] = torch.deg2rad(torch.tensor(-157.0, device=env.device))

  open_pose = default_joint_pos.clone()
  open_pose[:, [1, 4, 7, 10]] = torch.deg2rad(torch.tensor(35.0, device=env.device))
  open_pose[:, [2, 5, 8, 11]] = torch.deg2rad(torch.tensor(-94.0, device=env.device))

  pose = torch.where((t >= 0.25) & (t < 0.62), crouch, pose)
  pose = torch.where((t >= 0.62) & (t < 0.92), thrust, pose)
  pose = torch.where((t >= 0.92) & (t < 1.45), tuck, pose)
  pose = torch.where((t >= 1.45) & (t < 1.90), open_pose, pose)
  return pose


def reference_joint_error(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return reference_joint_pose(env, asset_cfg) - asset.data.joint_pos[:, asset_cfg.joint_ids]


def flip_tracking_reward(env: ManagerBasedRlEnv, std: float, asset_cfg=None) -> torch.Tensor:
  error = torch.atan2(
    torch.sin(target_flip_angle(env) - flip_angle(env, asset_cfg)),
    torch.cos(target_flip_angle(env) - flip_angle(env, asset_cfg)),
  )
  return torch.exp(-torch.square(error) / (std * std))


def jump_height_reward(
  env: ManagerBasedRlEnv,
  min_time: float,
  max_time: float,
  target_height: float,
  asset_cfg=None,
) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  t = elapsed_time(env)
  active = ((t >= min_time) & (t <= max_time)).float()
  height = torch.clamp((asset.data.root_link_pos_w[:, 2] - 0.28) / target_height, 0.0, 1.0)
  vz = torch.clamp(asset.data.root_link_lin_vel_w[:, 2] / 2.0, 0.0, 1.0)
  pitch_rate = torch.clamp(asset.data.root_link_ang_vel_w[:, 1] / 12.0, 0.0, 1.0)
  return active * (2.0 * height + vz + pitch_rate)


def feet_air_reward(env: ManagerBasedRlEnv, sensor_name: str, min_time: float, max_time: float) -> torch.Tensor:
  t = elapsed_time(env)
  active = ((t >= min_time) & (t <= max_time)).float()
  sensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = (sensor.data.found > 0).float().flatten(start_dim=1).sum(dim=-1)
  return active * (contacts == 0).float()


def landing_reward(env: ManagerBasedRlEnv, sensor_name: str, min_time: float, asset_cfg=None) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  t = elapsed_time(env)
  active = (t >= min_time).float()
  angle = flip_angle(env, asset_cfg)
  final_angle_reward = torch.exp(-torch.square(angle) / (0.55 * 0.55))
  upright_reward = torch.clamp((-asset.data.projected_gravity_b[:, 2] - 0.45) / 0.55, 0.0, 1.0)
  height_reward = torch.exp(-torch.square(asset.data.root_link_pos_w[:, 2] - 0.27) / (0.12 * 0.12))
  sensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = (sensor.data.found > 0).float().flatten(start_dim=1).sum(dim=-1)
  contact_reward = torch.clamp(contacts / 4.0, 0.0, 1.0)
  return active * (8.0 * final_angle_reward * upright_reward + 2.0 * height_reward + contact_reward)


def reference_pose_reward(env: ManagerBasedRlEnv, std: float, asset_cfg) -> torch.Tensor:
  error = reference_joint_error(env, asset_cfg)
  return torch.exp(-torch.mean(torch.square(error), dim=-1) / (std * std))


def nonfoot_contact_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return nonfoot_contact_count(env, sensor_name)


def base_height_below(env: ManagerBasedRlEnv, minimum_height: float, asset_cfg=None) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def illegal_nonfoot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  grace_time: float = 0.20,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  t = elapsed_time(env)
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  else:
    assert data.force is not None
    hit = (torch.norm(data.force, dim=-1) > force_threshold).any(dim=-1)
  return hit & (t > grace_time)
