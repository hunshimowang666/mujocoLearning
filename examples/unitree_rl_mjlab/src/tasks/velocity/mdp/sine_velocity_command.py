from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import wrap_to_pi

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


class SineVelocityCommand(UniformVelocityCommand):
  cfg: SineVelocityCommandCfg

  def __init__(self, cfg: "SineVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.phase_offset = torch.zeros(self.num_envs, device=self.device)
    self.reference_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.reference_heading_w = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.vel_command_b[env_ids, 0] = self.cfg.lin_vel_x
    self.vel_command_b[env_ids, 1] = self.cfg.lin_vel_y
    if self.cfg.randomize_phase:
      self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device)
    else:
      self.phase_offset[env_ids] = 0.0
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    self.reference_pos_w[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]
    self.reference_heading_w[env_ids] = self.robot.data.heading_w[env_ids]

  def _update_command(self) -> None:
    phase = (
      self._env.episode_length_buf.to(dtype=torch.float32)
      * self._env.step_dt
      / self.cfg.period
      + self.phase_offset
    )
    self.vel_command_b[:, 0] = self.cfg.lin_vel_x
    self.vel_command_b[:, 1] = self.cfg.lin_vel_y
    self.vel_command_b[:, 2] = self.cfg.ang_vel_z_amplitude * torch.sin(
      2.0 * torch.pi * phase
    )
    self.is_standing_env[:] = False
    self.is_heading_env[:] = False

    dt = self._env.step_dt
    self.reference_heading_w = wrap_to_pi(
      self.reference_heading_w + self.vel_command_b[:, 2] * dt
    )
    cos_h = torch.cos(self.reference_heading_w)
    sin_h = torch.sin(self.reference_heading_w)
    vel_x_w = cos_h * self.cfg.lin_vel_x - sin_h * self.cfg.lin_vel_y
    vel_y_w = sin_h * self.cfg.lin_vel_x + cos_h * self.cfg.lin_vel_y
    self.reference_pos_w[:, 0] += vel_x_w * dt
    self.reference_pos_w[:, 1] += vel_y_w * dt


@dataclass(kw_only=True)
class SineVelocityCommandCfg(UniformVelocityCommandCfg):
  lin_vel_x: float = 0.5
  lin_vel_y: float = 0.0
  ang_vel_z_amplitude: float = 0.6
  period: float = 5.0
  randomize_phase: bool = True

  def build(self, env: ManagerBasedRlEnv) -> SineVelocityCommand:
    return SineVelocityCommand(self, env)
