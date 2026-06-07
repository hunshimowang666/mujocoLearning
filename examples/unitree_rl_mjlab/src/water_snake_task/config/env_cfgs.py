"""MJLab environment configuration for the 3-cabin water snake path task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from src.water_snake_task import mdp


TASK_ID = "WaterSnake-Path-MJLab"
REPO_ROOT = Path(__file__).resolve().parents[5]
XML_PATH = REPO_ROOT / "examples" / "3dModels" / "sw2urdfWS2d" / "view.xml"
PATH_TABLE = REPO_ROOT / "examples" / "newSnakeExplore2D.txt"

THRUSTER_SITE_NAMES = mdp.THRUSTER_SITE_NAMES
JOINT_NAMES = mdp.JOINT_NAMES
BODY_NAMES = mdp.BODY_NAMES
def robot_joints_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)


def robot_bodies_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=list(BODY_NAMES), preserve_order=True)

MAX_THRUST = 30.0
MAX_JOINT_ANGLE_DEG = 60.0
JOINT_POSITION_STIFFNESS = 8.0
JOINT_POSITION_DAMPING = 1.2
JOINT_POSITION_EFFORT_LIMIT = 20.0
PATH_ROW_UPDATE_INTERVAL = 0.1
MAX_TRACKING_ERROR = 3.0
ROLL_PITCH_PENALTY_WEIGHT = -2.0
DRAW_PATH_IN_PLAY = True


def _add_motor(
  spec: mujoco.MjSpec,
  *,
  name: str,
  target: str,
  trntype: mujoco.mjtTrn,
  gear: tuple[float, float, float, float, float, float],
  effort_limit: float,
) -> None:
  actuator = spec.add_actuator(name=name, target=target)
  actuator.trntype = trntype
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
  actuator.gear[:] = np.asarray(gear, dtype=np.float64)
  actuator.forcelimited = True
  actuator.forcerange[:] = np.asarray([-effort_limit, effort_limit], dtype=np.float64)
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.asarray([-effort_limit, effort_limit], dtype=np.float64)


def _planner_xy_to_mujoco_path_point(x: float, y: float) -> tuple[float, float, float]:
  return (float(x), -float(y), mdp.ZERO_DEPTH_WORLD_Z)


def _add_path_visuals(spec: mujoco.MjSpec) -> None:
  table = np.loadtxt(PATH_TABLE)
  if table.ndim == 1:
    table = table.reshape(1, -1)
  points = [
    _planner_xy_to_mujoco_path_point(row[mdp.BACK_POSE_COLS[0]], row[mdp.BACK_POSE_COLS[1]])
    for row in table
  ]
  for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
    if np.linalg.norm(np.asarray(end) - np.asarray(start)) < 1.0e-6:
      continue
    geom = spec.worldbody.add_geom(
      name=f"path_line_{index:03d}",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    )
    geom.fromto[:] = np.asarray([*start, *end], dtype=np.float64)
    geom.size[0] = 0.012
    geom.rgba[:] = np.asarray([0.0, 0.9, 1.0, 0.82], dtype=np.float64)
    geom.contype = 0
    geom.conaffinity = 0
  if points:
    for name, point, rgba in (
      ("path_start", points[0], (0.0, 1.0, 0.2, 1.0)),
      ("path_goal", points[-1], (1.0, 0.15, 0.0, 1.0)),
    ):
      geom = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE)
      geom.pos[:] = np.asarray(point, dtype=np.float64)
      geom.size[0] = 0.04
      geom.rgba[:] = np.asarray(rgba, dtype=np.float64)
      geom.contype = 0
      geom.conaffinity = 0


def _get_water_snake_spec(draw_path: bool = False) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(XML_PATH))
  spec.option.gravity[:] = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
  spec.option.density = 1000.0
  spec.option.viscosity = 0.0009
  for joint_name in JOINT_NAMES:
    joint = spec.joint(joint_name)
    joint.damping = 0.02
    joint.armature = 0.0001

  for site_name in THRUSTER_SITE_NAMES:
    _add_motor(
      spec,
      name=f"{site_name}_motor",
      target=site_name,
      trntype=mujoco.mjtTrn.mjTRN_SITE,
      gear=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
      effort_limit=MAX_THRUST,
    )
  if draw_path:
    _add_path_visuals(spec)
  return spec


def get_water_snake_robot_cfg(draw_path: bool = False) -> EntityCfg:
  init_pos, init_rot, init_joints = mdp.first_row_initial_state(PATH_TABLE)
  articulation = EntityArticulationInfoCfg(
    actuators=(
      XmlMotorActuatorCfg(
        target_names_expr=THRUSTER_SITE_NAMES,
        transmission_type=TransmissionType.SITE,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=JOINT_NAMES,
        transmission_type=TransmissionType.JOINT,
        stiffness=JOINT_POSITION_STIFFNESS,
        damping=JOINT_POSITION_DAMPING,
        effort_limit=JOINT_POSITION_EFFORT_LIMIT,
      ),
    ),
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=init_pos,
      rot=init_rot,
      joint_pos=init_joints,
      joint_vel={".*": 0.0},
    ),
    spec_fn=lambda: _get_water_snake_spec(draw_path=draw_path),
    articulation=articulation,
    sort_actuators=False,
  )


def _path_params() -> dict[str, str | float]:
  return {
    "path_table": str(PATH_TABLE),
    "row_update_interval": PATH_ROW_UPDATE_INTERVAL,
  }


def water_snake_path_env_cfg(
  play: bool = False,
  draw_path: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  if draw_path is None:
    draw_path = play and DRAW_PATH_IN_PLAY
  path_params = _path_params()

  def observation_terms() -> dict[str, ObservationTermCfg]:
    return {
      "target": ObservationTermCfg(func=mdp.target_features, params=path_params),
      "pose_error": ObservationTermCfg(
        func=mdp.pose_error_obs,
        params={"asset_cfg": robot_bodies_cfg(), **path_params},
      ),
      "joint_state": ObservationTermCfg(
        func=mdp.joint_state_obs,
        params={"asset_cfg": robot_joints_cfg(), **path_params},
      ),
      "body_velocity": ObservationTermCfg(
        func=mdp.body_velocity_obs,
        params={"asset_cfg": robot_bodies_cfg()},
      ),
      "roll_pitch_error": ObservationTermCfg(
        func=mdp.body_roll_pitch_obs,
        params={"asset_cfg": robot_bodies_cfg()},
      ),
      "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }

  actor_terms = observation_terms()
  critic_terms = observation_terms()

  actions = {
    "thrusters_and_joints": mdp.WaterSnakeThrusterJointAngleActionCfg(
      entity_name="robot",
      max_thrust=MAX_THRUST,
      max_joint_angle=np.deg2rad(MAX_JOINT_ANGLE_DEG),
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (0.0, 0.0) if play else (-0.02, 0.02),
          "y": (0.0, 0.0) if play else (-0.02, 0.02),
          "z": (0.0, 0.0),
          "yaw": (0.0, 0.0) if play else (-0.03, 0.03),
        },
        "velocity_range": {},
      },
    ),
    "reset_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0) if play else (-0.02, 0.02),
        "velocity_range": (0.0, 0.0) if play else (-0.02, 0.02),
        "asset_cfg": robot_joints_cfg(),
      },
    ),
  }

  rewards = {
    "body_position": RewardTermCfg(
      func=mdp.body_position_tracking_reward,
      weight=8.0,
      params={"asset_cfg": robot_bodies_cfg(), "std": 0.8, **path_params},
    ),
    "body_yaw": RewardTermCfg(
      func=mdp.body_yaw_tracking_reward,
      weight=3.0,
      params={"asset_cfg": robot_bodies_cfg(), "std": 1.2, **path_params},
    ),
    "joint_tracking": RewardTermCfg(
      func=mdp.joint_tracking_reward,
      weight=2.0,
      params={"asset_cfg": robot_joints_cfg(), "std": 0.4, **path_params},
    ),
    "body_velocity_l2": RewardTermCfg(
      func=mdp.body_velocity_l2,
      weight=-0.02,
      params={"asset_cfg": robot_bodies_cfg()},
    ),
    "roll_pitch_l2": RewardTermCfg(
      func=mdp.body_roll_pitch_l2,
      weight=ROLL_PITCH_PENALTY_WEIGHT,
      params={"asset_cfg": robot_bodies_cfg()},
    ),
    "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-0.04),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-20.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "tracking_error": TerminationTermCfg(
      func=mdp.back_tracking_error_too_large,
      params={"asset_cfg": robot_bodies_cfg(), "limit": MAX_TRACKING_ERROR, **path_params},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=None,
      entities={"robot": get_water_snake_robot_cfg(draw_path=draw_path)},
      num_envs=2048,
      env_spacing=3.0,
      extent=5.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=False,
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
      body_name="backDrivenCabin",
      distance=3.0,
      elevation=-25.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=8,
      njmax=64,
      mujoco=MujocoCfg(
        timestep=0.002,
        gravity=(0.0, 0.0, 0.0),
        iterations=20,
        ls_iterations=20,
        ccd_iterations=30,
        integrator="implicitfast",
      ),
    ),
    decimation=10,
    episode_length_s=mdp.path_total_time(PATH_TABLE),
  )

  if play:
    cfg.scene.num_envs = 1

  return cfg
