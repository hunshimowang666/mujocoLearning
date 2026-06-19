"""Gymnasium Humanoid-v5 asset configuration for MJLab."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from src import SRC_PATH


HUMANOID_V5_XML: Path = (
  SRC_PATH / "assets" / "robots" / "gym_humanoid_v5" / "xmls" / "humanoid_v5.xml"
)
assert HUMANOID_V5_XML.exists()

HUMANOID_V5_MOTOR_JOINTS: tuple[str, ...] = (
  "abdomen_y",
  "abdomen_z",
  "abdomen_x",
  "right_hip_x",
  "right_hip_z",
  "right_hip_y",
  "right_knee",
  "left_hip_x",
  "left_hip_z",
  "left_hip_y",
  "left_knee",
  "right_shoulder1",
  "right_shoulder2",
  "right_elbow",
  "left_shoulder1",
  "left_shoulder2",
  "left_elbow",
)

HUMANOID_V5_JOINTS: tuple[str, ...] = HUMANOID_V5_MOTOR_JOINTS


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HUMANOID_V5_XML))


HUMANOID_V5_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.4),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

HUMANOID_V5_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlMotorActuatorCfg(target_names_expr=HUMANOID_V5_MOTOR_JOINTS),
  ),
  soft_joint_pos_limit_factor=1.0,
)

HUMANOID_V5_ACTION_SCALE = 0.4


def get_humanoid_v5_robot_cfg() -> EntityCfg:
  """Return the Gymnasium Humanoid-v5 MJCF as an MJLab entity.

  The XML is copied from Gymnasium and only its world floor/light are removed;
  bodies, joints, geoms, tendons, and motors are kept intact.
  """
  return EntityCfg(
    init_state=HUMANOID_V5_INIT_STATE,
    spec_fn=get_spec,
    articulation=HUMANOID_V5_ARTICULATION,
    sort_actuators=False,
  )
