"""Unitree Go2 backflip environment configuration for mjlab."""

from __future__ import annotations

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig
from src.assets.robots.unitree_go2.go2_constants import INIT_STATE, get_spec
from src.backflip_task import mdp


TASK_ID = "Unitree-Go2-Backflip"

GO2_JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
GO2_FOOT_NAMES = ("FR", "FL", "RR", "RL")
GO2_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in GO2_FOOT_NAMES)
ROBOT_JOINTS = SceneEntityCfg(
  "robot",
  joint_names=GO2_JOINT_NAMES,
  preserve_order=True,
)

_FOOT_REGEX = "^[FR][LR]_foot_collision$"


def _get_torque_go2_spec() -> mujoco.MjSpec:
  spec = get_spec()
  for joint_name in GO2_JOINT_NAMES:
    joint = spec.joint(joint_name)
    joint.damping = 0.10
  return spec


def get_go2_torque_robot_cfg() -> EntityCfg:
  """Build a Go2 entity that consumes direct joint-effort targets."""
  collision = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FOOT_REGEX: 6, ".*_collision": 1},
    priority={_FOOT_REGEX: 1},
    friction={_FOOT_REGEX: (1.6, 0.25, 0.05)},
    solimp={_FOOT_REGEX: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
  )
  articulation = EntityArticulationInfoCfg(
    actuators=(
      BuiltinMotorActuatorCfg(
        target_names_expr=(".*hip_joint",),
        effort_limit=23.5,
        armature=0.01,
        frictionloss=0.10,
      ),
      BuiltinMotorActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        effort_limit=23.5,
        armature=0.01,
        frictionloss=0.10,
      ),
      BuiltinMotorActuatorCfg(
        target_names_expr=(".*calf_joint",),
        effort_limit=45.0,
        armature=0.02,
        frictionloss=0.12,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(collision,),
    spec_fn=_get_torque_go2_spec,
    articulation=articulation,
  )


def unitree_go2_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=GO2_FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=GO2_FOOT_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=None if play else Unoise(n_min=-0.10, n_max=0.10),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=None if play else Unoise(n_min=-0.02, n_max=0.02),
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase_scalar,
      params={"duration": 3.2},
    ),
    "flip_angle": ObservationTermCfg(func=mdp.flip_angle_obs),
    "target_flip": ObservationTermCfg(func=mdp.target_flip_obs),
    "base_height": ObservationTermCfg(func=mdp.body_height),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": ROBOT_JOINTS},
      noise=None if play else Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": ROBOT_JOINTS},
      noise=None if play else Unoise(n_min=-0.50, n_max=0.50),
      scale=0.1,
    ),
    "reference_joint_error": ObservationTermCfg(
      func=mdp.reference_joint_error,
      params={"asset_cfg": ROBOT_JOINTS},
    ),
    "feet_contact": ObservationTermCfg(
      func=mdp.feet_contact_obs,
      params={"sensor_name": feet_ground_cfg.name},
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel_w": ObservationTermCfg(func=mdp.base_linear_velocity_w),
    "base_ang_vel_w": ObservationTermCfg(func=mdp.base_angular_velocity_w),
    "nonfoot_contact_count": ObservationTermCfg(
      func=mdp.nonfoot_contact_count,
      params={"sensor_name": nonfoot_ground_cfg.name},
    ),
  }

  actions = {
    "joint_effort": JointEffortActionCfg(
      entity_name="robot",
      actuator_names=GO2_JOINT_NAMES,
      scale={
        r".*hip_joint.*": 23.5,
        r".*thigh_joint.*": 23.5,
        r".*calf_joint.*": 45.0,
      },
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.03, 0.03),
          "y": (-0.03, 0.03),
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
        "position_range": (-0.02, 0.02),
        "velocity_range": (-0.02, 0.02),
        "asset_cfg": ROBOT_JOINTS,
      },
    ),
  }

  rewards = {
    "flip_tracking": RewardTermCfg(
      func=mdp.flip_tracking_reward,
      weight=4.0,
      params={"std": 0.9},
    ),
    "jump_height": RewardTermCfg(
      func=mdp.jump_height_reward,
      weight=1.0,
      params={"min_time": 0.55, "max_time": 1.25, "target_height": 0.30},
    ),
    "feet_air": RewardTermCfg(
      func=mdp.feet_air_reward,
      weight=1.0,
      params={"sensor_name": feet_ground_cfg.name, "min_time": 0.82, "max_time": 1.65},
    ),
    "landing": RewardTermCfg(
      func=mdp.landing_reward,
      weight=1.2,
      params={"sensor_name": feet_ground_cfg.name, "min_time": 1.75},
    ),
    "reference_pose": RewardTermCfg(
      func=mdp.reference_pose_reward,
      weight=0.8,
      params={"std": 0.55, "asset_cfg": ROBOT_JOINTS},
    ),
    "nonfoot_contact": RewardTermCfg(
      func=mdp.nonfoot_contact_penalty,
      weight=-8.0,
      params={"sensor_name": nonfoot_ground_cfg.name},
    ),
    "joint_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.004,
      params={"asset_cfg": ROBOT_JOINTS},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": ROBOT_JOINTS},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": ROBOT_JOINTS},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.04),
    "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-80.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "too_low": TerminationTermCfg(
      func=mdp.base_height_below,
      params={"minimum_height": 0.10},
    ),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_nonfoot_contact,
      params={
        "sensor_name": nonfoot_ground_cfg.name,
        "force_threshold": 15.0,
        "grace_time": 0.20,
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_go2_torque_robot_cfg()},
      sensors=(feet_ground_cfg, nonfoot_ground_cfg),
      num_envs=4096,
      env_spacing=2.0,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=1,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
      ),
    },
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=120,
      njmax=600,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=80,
        multiccd=True,
      ),
    ),
    decimation=4,
    episode_length_s=3.2,
  )

  if play:
    cfg.scene.num_envs = 1
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

  return cfg
