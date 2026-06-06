"""MDP terms and custom actions for the water snake path task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PATH_TABLE = REPO_ROOT / "examples" / "newSnakeExplore2D.txt"

THRUSTER_SITE_NAMES = ("FR_site", "FL_site", "FU_site", "FD_site", "BR_site", "BL_site", "BU_site", "BD_site")
JOINT_NAMES = ("J1", "J2")
BODY_NAMES = ("backDrivenCabin", "frontDrivenCabin", "headCabin")
# The exported head cabin has a different local mesh frame: its local +Y axis is
# the cabin's upright axis, while the two cylindrical cabins use local +Z.
BODY_UP_AXES = (
  (0.0, 0.0, 1.0),
  (0.0, 0.0, 1.0),
  (0.0, 1.0, 0.0),
)

BACK_POSE_COLS = (1, 2, 3)
JOINT_COLS = (4, 5)
FRONT_POSE_COLS = (8, 9, 10)
HEAD_POSE_COLS = (11, 12, 13)

ZERO_DEPTH_WORLD_Z = 0.45
ROW_UPDATE_INTERVAL = 0.1

_PATH_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def _path_tensor(device: str, path_table: str | Path = DEFAULT_PATH_TABLE) -> torch.Tensor:
  path = Path(path_table).resolve()
  key = (str(path), str(device))
  if key not in _PATH_CACHE:
    table = np.loadtxt(path)
    if table.ndim == 1:
      table = table.reshape(1, -1)
    if table.shape[1] < 14:
      raise RuntimeError(f"{path} needs at least 14 columns.")
    _PATH_CACHE[key] = torch.tensor(table, dtype=torch.float32, device=device)
  return _PATH_CACHE[key]


def path_total_time(path_table: str | Path = DEFAULT_PATH_TABLE) -> float:
  table = np.loadtxt(path_table)
  if table.ndim == 1:
    table = table.reshape(1, -1)
  return float(table[-1, 0])


def first_row_initial_state(path_table: str | Path = DEFAULT_PATH_TABLE) -> tuple[tuple[float, float, float], tuple[float, float, float, float], dict[str, float]]:
  table = np.loadtxt(path_table)
  if table.ndim == 1:
    table = table.reshape(1, -1)
  row = table[0]
  x = float(row[BACK_POSE_COLS[0]])
  y = -float(row[BACK_POSE_COLS[1]])
  yaw = 0.5 * np.pi - np.deg2rad(float(row[BACK_POSE_COLS[2]]))
  pos = (x, y, ZERO_DEPTH_WORLD_Z)
  rot = (float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw)))
  joints = {
    "J1": -float(np.deg2rad(row[JOINT_COLS[0]])),
    "J2": -float(np.deg2rad(row[JOINT_COLS[1]])),
  }
  return pos, rot, joints


def elapsed_time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf.to(dtype=torch.float32) * env.step_dt


def path_row_indices(
  env: ManagerBasedRlEnv,
  path_table: str | Path = DEFAULT_PATH_TABLE,
  row_update_interval: float = ROW_UPDATE_INTERVAL,
) -> torch.Tensor:
  table = _path_tensor(env.device, path_table)
  row = torch.floor((elapsed_time(env) + 1.0e-6) / row_update_interval).long()
  return torch.clamp(row, 0, table.shape[0] - 1)


def current_path_rows(
  env: ManagerBasedRlEnv,
  path_table: str | Path = DEFAULT_PATH_TABLE,
  row_update_interval: float = ROW_UPDATE_INTERVAL,
) -> torch.Tensor:
  table = _path_tensor(env.device, path_table)
  return table[path_row_indices(env, path_table, row_update_interval)]


def target_features(
  env: ManagerBasedRlEnv,
  path_table: str | Path = DEFAULT_PATH_TABLE,
  row_update_interval: float = ROW_UPDATE_INTERVAL,
) -> torch.Tensor:
  rows = current_path_rows(env, path_table, row_update_interval)
  total_time = torch.clamp(_path_tensor(env.device, path_table)[-1, 0], min=1.0)
  return torch.stack(
    (
      rows[:, 0] / total_time,
      rows[:, BACK_POSE_COLS[0]] / 5.0,
      rows[:, BACK_POSE_COLS[1]] / 5.0,
      torch.deg2rad(rows[:, BACK_POSE_COLS[2]]) / torch.pi,
      torch.deg2rad(rows[:, JOINT_COLS[0]]) / torch.pi,
      torch.deg2rad(rows[:, JOINT_COLS[1]]) / torch.pi,
      rows[:, FRONT_POSE_COLS[0]] / 5.0,
      rows[:, FRONT_POSE_COLS[1]] / 5.0,
      torch.deg2rad(rows[:, FRONT_POSE_COLS[2]]) / torch.pi,
      rows[:, HEAD_POSE_COLS[0]] / 5.0,
      rows[:, HEAD_POSE_COLS[1]] / 5.0,
      torch.deg2rad(rows[:, HEAD_POSE_COLS[2]]) / torch.pi,
    ),
    dim=-1,
  )


def target_joint_pos_mj(
  env: ManagerBasedRlEnv,
  path_table: str | Path = DEFAULT_PATH_TABLE,
  row_update_interval: float = ROW_UPDATE_INTERVAL,
) -> torch.Tensor:
  rows = current_path_rows(env, path_table, row_update_interval)
  return -torch.deg2rad(rows[:, list(JOINT_COLS)])


def target_body_pose_planner(
  env: ManagerBasedRlEnv,
  path_table: str | Path = DEFAULT_PATH_TABLE,
  row_update_interval: float = ROW_UPDATE_INTERVAL,
) -> torch.Tensor:
  rows = current_path_rows(env, path_table, row_update_interval)
  back = torch.stack(
    (
      rows[:, BACK_POSE_COLS[0]],
      rows[:, BACK_POSE_COLS[1]],
      torch.deg2rad(rows[:, BACK_POSE_COLS[2]]),
    ),
    dim=-1,
  )
  front = torch.stack(
    (
      rows[:, FRONT_POSE_COLS[0]],
      rows[:, FRONT_POSE_COLS[1]],
      torch.deg2rad(rows[:, FRONT_POSE_COLS[2]]),
    ),
    dim=-1,
  )
  head = torch.stack(
    (
      rows[:, HEAD_POSE_COLS[0]],
      rows[:, HEAD_POSE_COLS[1]],
      torch.deg2rad(rows[:, HEAD_POSE_COLS[2]]),
    ),
    dim=-1,
  )
  return torch.stack((back, front, head), dim=1)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  shape = vec.shape
  quat = quat.reshape(-1, 4)
  vec = vec.reshape(-1, 3)
  xyz = quat[:, 1:]
  t = xyz.cross(vec, dim=-1) * 2.0
  return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def body_pose_planner(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  pos = asset.data.body_link_pos_w[:, body_ids, :]
  quat = asset.data.body_link_quat_w[:, body_ids, :]
  forward_b = torch.zeros_like(pos)
  forward_b[..., 1] = -1.0
  forward_w = _quat_apply(quat, forward_b)
  yaw = torch.atan2(-forward_w[..., 1], forward_w[..., 0])
  return torch.stack((pos[..., 0], -pos[..., 1], yaw), dim=-1)


def body_velocity_planner(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  lin = asset.data.body_link_lin_vel_w[:, body_ids, :]
  ang = asset.data.body_link_ang_vel_w[:, body_ids, :]
  return torch.stack((lin[..., 0], -lin[..., 1], -ang[..., 2]), dim=-1)


def pose_error(
  env: ManagerBasedRlEnv,
  asset_cfg,
  path_table: str | Path = DEFAULT_PATH_TABLE,
  row_update_interval: float = ROW_UPDATE_INTERVAL,
) -> torch.Tensor:
  actual = body_pose_planner(env, asset_cfg)
  target = target_body_pose_planner(env, path_table, row_update_interval)
  error = actual - target
  error[..., 2] = wrap_to_pi(error[..., 2])
  return error


def pose_error_obs(env: ManagerBasedRlEnv, asset_cfg, **kwargs) -> torch.Tensor:
  error = pose_error(env, asset_cfg, **kwargs)
  return torch.cat((error[..., 0:1] / 5.0, error[..., 1:2] / 5.0, error[..., 2:3] / torch.pi), dim=-1).flatten(start_dim=1)


def joint_state_obs(env: ManagerBasedRlEnv, asset_cfg, **kwargs) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  target = target_joint_pos_mj(env, **kwargs)
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  dq = asset.data.joint_vel[:, asset_cfg.joint_ids]
  err = wrap_to_pi(q - target)
  return torch.cat((err / torch.pi, dq / 5.0), dim=-1)


def body_velocity_obs(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  return (body_velocity_planner(env, asset_cfg) / 2.0).flatten(start_dim=1)


def body_roll_pitch_error(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  quat = asset.data.body_link_quat_w[:, body_ids, :]
  up_b = torch.tensor(BODY_UP_AXES, dtype=quat.dtype, device=quat.device)
  up_b = up_b.unsqueeze(0).expand(quat.shape[0], -1, -1)
  up_w = _quat_apply(quat, up_b)
  return up_w[..., :2]


def body_roll_pitch_obs(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  return body_roll_pitch_error(env, asset_cfg).flatten(start_dim=1)


def path_progress_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  del kwargs
  return elapsed_time(env).unsqueeze(-1)


def body_position_tracking_reward(env: ManagerBasedRlEnv, asset_cfg, std: float, **kwargs) -> torch.Tensor:
  error = pose_error(env, asset_cfg, **kwargs)
  pos_err_sum = torch.linalg.norm(error[..., :2], dim=-1).sum(dim=-1)
  return torch.exp(-pos_err_sum / std)


def body_yaw_tracking_reward(env: ManagerBasedRlEnv, asset_cfg, std: float, **kwargs) -> torch.Tensor:
  error = pose_error(env, asset_cfg, **kwargs)
  yaw_err_sum = torch.abs(error[..., 2]).sum(dim=-1)
  return torch.exp(-yaw_err_sum / std)


def joint_tracking_reward(env: ManagerBasedRlEnv, asset_cfg, std: float, **kwargs) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  target = target_joint_pos_mj(env, **kwargs)
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  err = wrap_to_pi(q - target)
  return torch.exp(-torch.linalg.norm(err, dim=-1) / std)


def body_velocity_l2(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  vel = body_velocity_planner(env, asset_cfg)
  return torch.mean(torch.square(vel), dim=(1, 2))


def body_roll_pitch_l2(env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
  error = body_roll_pitch_error(env, asset_cfg)
  return torch.mean(torch.square(error), dim=(1, 2))


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.mean(torch.square(env.action_manager.action), dim=-1)


def back_tracking_error_too_large(env: ManagerBasedRlEnv, asset_cfg, limit: float, **kwargs) -> torch.Tensor:
  error = pose_error(env, asset_cfg, **kwargs)
  return torch.linalg.norm(error[:, 0, :2], dim=-1) > limit


@dataclass(kw_only=True)
class WaterSnakeThrusterPidActionCfg(ActionTermCfg):
  site_names: tuple[str, ...] = THRUSTER_SITE_NAMES
  joint_names: tuple[str, ...] = JOINT_NAMES
  max_thrust: float = 30.0
  pid_kp: float = 8.0
  pid_ki: float = 0.05
  pid_kd: float = 1.2
  pid_integral_limit: float = 1.5
  pid_torque_limit: float = 20.0
  path_table: str = str(DEFAULT_PATH_TABLE)
  row_update_interval: float = ROW_UPDATE_INTERVAL

  def build(self, env: ManagerBasedRlEnv) -> "WaterSnakeThrusterPidAction":
    return WaterSnakeThrusterPidAction(self, env)


class WaterSnakeThrusterPidAction(ActionTerm):
  cfg: WaterSnakeThrusterPidActionCfg

  def __init__(self, cfg: WaterSnakeThrusterPidActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    site_ids, _ = self._entity.find_sites(cfg.site_names, preserve_order=True)
    joint_ids, _ = self._entity.find_joints(cfg.joint_names, preserve_order=True)
    self._site_ids = torch.tensor(site_ids, dtype=torch.long, device=self.device)
    self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
    self._raw_actions = torch.zeros(self.num_envs, len(site_ids), device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._pid_integral = torch.zeros(self.num_envs, len(joint_ids), device=self.device)

  @property
  def action_dim(self) -> int:
    return self._raw_actions.shape[1]

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions.to(self.device)
    self._processed_actions[:] = torch.clamp(self._raw_actions, -1.0, 1.0) * self.cfg.max_thrust

  def apply_actions(self) -> None:
    self._entity.set_site_effort_target(self._processed_actions, site_ids=self._site_ids)
    target = target_joint_pos_mj(
      self._env,
      path_table=self.cfg.path_table,
      row_update_interval=self.cfg.row_update_interval,
    )
    q = self._entity.data.joint_pos[:, self._joint_ids]
    dq = self._entity.data.joint_vel[:, self._joint_ids]
    err = wrap_to_pi(target - q)
    self._pid_integral += err * self._env.physics_dt
    self._pid_integral.clamp_(-self.cfg.pid_integral_limit, self.cfg.pid_integral_limit)
    torque = self.cfg.pid_kp * err + self.cfg.pid_ki * self._pid_integral - self.cfg.pid_kd * dq
    torque = torch.clamp(torque, -self.cfg.pid_torque_limit, self.cfg.pid_torque_limit)
    self._entity.set_joint_effort_target(torque, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._pid_integral[env_ids] = 0.0
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
