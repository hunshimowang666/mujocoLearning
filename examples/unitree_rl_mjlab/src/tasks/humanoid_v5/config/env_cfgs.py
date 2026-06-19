"""MJLab environment config for Gymnasium Humanoid-v5."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from src.assets.robots import HUMANOID_V5_ACTION_SCALE, get_humanoid_v5_robot_cfg
from src.assets.robots.gym_humanoid_v5.humanoid_v5_constants import (
  HUMANOID_V5_MOTOR_JOINTS,
)
from src.tasks.humanoid_v5 import mdp


def _robot_joints() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=list(HUMANOID_V5_MOTOR_JOINTS),
    preserve_order=True,
  )


def _foot_bodies() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=("right_foot", "left_foot"),
    preserve_order=True,
  )


def _leg_joints() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=(
      "right_hip_x",
      "right_hip_z",
      "right_hip_y",
      "right_knee",
      "left_hip_x",
      "left_hip_z",
      "left_hip_y",
      "left_knee",
    ),
    preserve_order=True,
  )


def _knee_joints() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=("right_knee", "left_knee"),
    preserve_order=True,
  )


def humanoid_v5_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      noise=None if play else Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=None if play else Unoise(n_min=-0.05, n_max=0.05),
    ),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "gait_phase": ObservationTermCfg(
      func=mdp.gait_phase,
      params={
        "period": 0.8,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": _robot_joints()},
      noise=None if play else Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": _robot_joints()},
      noise=None if play else Unoise(n_min=-0.1, n_max=0.1),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  actions = {
    "motor_ctrl": JointEffortActionCfg(
      entity_name="robot",
      actuator_names=list(HUMANOID_V5_MOTOR_JOINTS),
      scale=HUMANOID_V5_ACTION_SCALE,
      preserve_order=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9) if play else (3.0, 8.0),
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.25),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=None,
      ),
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (0.0, 0.0),
          "yaw": (-0.05, 0.05),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.005, 0.005),
        "velocity_range": (-0.005, 0.005),
        "asset_cfg": _robot_joints(),
      },
    ),
  }

  rewards = {
    "track_forward_velocity": RewardTermCfg(
      func=mdp.track_forward_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": 0.35},
    ),
    "track_world_forward_velocity": RewardTermCfg(
      func=mdp.track_world_forward_velocity,
      weight=4.0,
      params={"command_name": "twist", "std": 0.35},
    ),
    "track_lateral_velocity_zero": RewardTermCfg(
      func=mdp.track_lateral_velocity_zero,
      weight=0.8,
      params={"std": 0.22},
    ),
    "track_world_lateral_velocity_zero": RewardTermCfg(
      func=mdp.track_world_lateral_velocity_zero,
      weight=2.0,
      params={"std": 0.18},
    ),
    "lateral_position": RewardTermCfg(func=mdp.lateral_position_l2, weight=-5.0),
    "track_yaw_velocity": RewardTermCfg(
      func=mdp.track_yaw_velocity,
      weight=0.8,
      params={"command_name": "twist", "std": 0.25},
    ),
    "alternating_foot_placement": RewardTermCfg(
      func=mdp.alternating_foot_placement,
      weight=2.2,
      params={
        "period": 0.8,
        "stride_length": 0.24,
        "std": 0.18,
        "command_name": "twist",
        "command_threshold": 0.1,
        "foot_body_cfg": _foot_bodies(),
      },
    ),
    "foot_fore_aft_separation": RewardTermCfg(
      func=mdp.foot_fore_aft_separation,
      weight=1.0,
      params={
        "target_separation": 0.22,
        "std": 0.12,
        "command_name": "twist",
        "command_threshold": 0.1,
        "foot_body_cfg": _foot_bodies(),
      },
    ),
    "foot_lateral_width": RewardTermCfg(
      func=mdp.foot_lateral_width_target,
      weight=0.5,
      params={
        "target_width": 0.22,
        "std": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "foot_body_cfg": _foot_bodies(),
      },
    ),
    "alternating_foot_velocity": RewardTermCfg(
      func=mdp.alternating_foot_velocity,
      weight=0.9,
      params={
        "period": 0.8,
        "stride_length": 0.24,
        "std": 0.65,
        "command_name": "twist",
        "command_threshold": 0.1,
        "foot_body_cfg": _foot_bodies(),
      },
    ),
    "swing_foot_clearance": RewardTermCfg(
      func=mdp.swing_foot_clearance,
      weight=0.8,
      params={
        "period": 0.8,
        "target_clearance": 0.12,
        "stance_fraction": 0.58,
        "command_name": "twist",
        "command_threshold": 0.1,
        "foot_body_cfg": _foot_bodies(),
      },
    ),
    "leg_motion_imbalance": RewardTermCfg(
      func=mdp.bilateral_leg_motion_imbalance,
      weight=-0.45,
      params={
        "leg_joint_cfg": _leg_joints(),
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    "upright": RewardTermCfg(func=mdp.upright_l2, weight=-8.0),
    "heading": RewardTermCfg(func=mdp.heading_zero_l2, weight=-8.0),
    "height": RewardTermCfg(
      func=mdp.root_height_target_l2,
      weight=-14.0,
      params={"target_height": 1.35},
    ),
    "deep_knee_bend": RewardTermCfg(
      func=mdp.deep_knee_bend_l2,
      weight=-3.0,
      params={"max_bend": 0.85, "knee_joint_cfg": _knee_joints()},
    ),
    "control": RewardTermCfg(func=mdp.action_l2, weight=-0.025),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
    "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "low_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 1.15},
    ),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
    "heading_deviation": TerminationTermCfg(
      func=mdp.heading_over_limit,
      params={"max_heading_error": math.radians(35.0)},
    ),
    "lateral_deviation": TerminationTermCfg(
      func=mdp.lateral_position_over_limit,
      params={"max_lateral_distance": 0.75},
    ),
  }

  metrics = {"mean_action_acc": MetricsTermCfg(func=envs_mdp.mean_action_acc)}

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=4.0),
      entities={"robot": get_humanoid_v5_robot_cfg()},
      num_envs=1,
      env_spacing=4.0,
      extent=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso",
      distance=4.0,
      elevation=-10.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=64,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=0.003,
        iterations=50,
        ls_iterations=20,
      ),
    ),
    decimation=5,
    episode_length_s=20.0,
  )

  if play:
    cfg.observations["actor"].enable_corruption = False
    twist_ranges = cfg.commands["twist"].ranges
    twist_ranges.lin_vel_x = (0.6, 0.6)
    # Viser creates joystick sliders from command upper bounds and requires
    # each "Max ..." slider to start at least at 0.1.  The play script still
    # overrides the actual command to the fixed values from 01_humanoid_train.py.
    twist_ranges.lin_vel_y = (-0.1, 0.1)
    twist_ranges.ang_vel_z = (-0.1, 0.1)

  return cfg
