import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(SCRIPT_DIR, "3dModels", "sw2urdfWS2d", "view.xml")
DEFAULT_PATH_TABLE = os.path.join(SCRIPT_DIR, "newSnakeExplore2D.txt")

THRUSTERS = ("FR", "FL", "FU", "FD", "BR", "BL", "BU", "BD")
JOINT_NAMES = ("J1", "J2")
BODY_NAMES = ("backDrivenCabin", "frontDrivenCabin", "headCabin")

BACK_POSE_COLS = (1, 2, 3)
JOINT_COLS = (4, 5)
FRONT_POSE_COLS = (8, 9, 10)
HEAD_POSE_COLS = (11, 12, 13)

ZERO_DEPTH_WORLD_Z = 0.45
PATH_CURVE_DEPTH = 0.0


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def planner_position_to_mujoco(x, y, z_down=0.0):
    return np.array([x, -y, ZERO_DEPTH_WORLD_Z - z_down], dtype=np.float64)


def mujoco_position_to_planner(pos):
    return np.array([pos[0], -pos[1], ZERO_DEPTH_WORLD_Z - pos[2]], dtype=np.float64)


def planner_yaw_to_mujoco_root_yaw(yaw):
    return 0.5 * np.pi - yaw


def planner_joint_deg_to_mujoco(joint_deg):
    return -joint_deg


def read_path_table(path):
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 14:
        raise RuntimeError(
            f"{path} needs at least 14 columns for time, poses, and J1/J2 targets"
        )
    return table.astype(np.float64)


def path_row_to_target(row):
    return {
        "time": float(row[0]),
        "back_pose": np.array(
            [row[BACK_POSE_COLS[0]], row[BACK_POSE_COLS[1]], np.deg2rad(row[BACK_POSE_COLS[2]])],
            dtype=np.float64,
        ),
        "joint_deg": np.array([row[JOINT_COLS[0]], row[JOINT_COLS[1]]], dtype=np.float64),
        "front_pose": np.array(
            [
                row[FRONT_POSE_COLS[0]],
                row[FRONT_POSE_COLS[1]],
                np.deg2rad(row[FRONT_POSE_COLS[2]]),
            ],
            dtype=np.float64,
        ),
        "head_pose": np.array(
            [row[HEAD_POSE_COLS[0]], row[HEAD_POSE_COLS[1]], np.deg2rad(row[HEAD_POSE_COLS[2]])],
            dtype=np.float64,
        ),
    }


def path_row_index_from_time(sim_time, row_interval, num_rows):
    return min(int(np.floor((sim_time + 1.0e-9) / row_interval)), num_rows - 1)


def add_arrow(scene, start, end, rgba, radius=0.018):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([radius, radius, radius * 4.0], dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def add_capsule(scene, start, end, rgba, radius=0.01):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_debug_scene(scene, path_table, axes_length=0.45):
    scene.ngeom = 0
    origin = np.zeros(3, dtype=np.float64)
    add_arrow(scene, origin, np.array([axes_length, 0.0, 0.0]), (1.0, 0.0, 0.0, 1.0))
    add_arrow(scene, origin, np.array([0.0, -axes_length, 0.0]), (0.0, 0.8, 0.0, 1.0))
    add_arrow(scene, origin, np.array([0.0, 0.0, -axes_length]), (0.1, 0.3, 1.0, 1.0))

    points = np.array(
        [
            planner_position_to_mujoco(row[BACK_POSE_COLS[0]], row[BACK_POSE_COLS[1]], PATH_CURVE_DEPTH)
            for row in path_table
        ],
        dtype=np.float64,
    )
    for start, end in zip(points[:-1], points[1:]):
        add_capsule(scene, start, end, (0.0, 0.9, 1.0, 0.85), radius=0.008)
    if len(points) > 0:
        add_sphere(scene, points[0], 0.025, (0.0, 1.0, 0.0, 1.0))
        add_sphere(scene, points[-1], 0.025, (1.0, 0.2, 0.0, 1.0))


class JointPID:
    def __init__(self, kp=8.0, ki=0.05, kd=1.2, integral_limit=1.5, torque_limit=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.torque_limit = torque_limit
        self.integral = np.zeros(2, dtype=np.float64)

    def reset(self):
        self.integral[:] = 0.0

    def compute(self, error, qvel, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        torque = self.kp * error + self.ki * self.integral - self.kd * qvel
        return np.clip(torque, -self.torque_limit, self.torque_limit)


class WaterSnakePathEnv(gym.Env):
    """8-thruster path-tracking task for the 3-cabin water snake model."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=DEFAULT_XML,
        path_table_path=DEFAULT_PATH_TABLE,
        row_update_interval=0.1,
        frame_skip=10,
        max_thrust=30.0,
        max_joint_angle=np.deg2rad(60.0),
        max_tracking_error=3.0,
        randomize_initial_pose=False,
    ):
        super().__init__()
        self.xml_path = xml_path
        self.path_table_path = path_table_path
        self.row_update_interval = row_update_interval
        self.frame_skip = frame_skip
        self.max_thrust = max_thrust
        self.max_joint_angle = max_joint_angle
        self.max_tracking_error = max_tracking_error
        self.randomize_initial_pose = randomize_initial_pose

        self.path_table = read_path_table(path_table_path)
        self.path_targets = [path_row_to_target(row) for row in self.path_table]
        self.path_total_time = float(self.path_table[-1, 0])

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        free_joint_ids = np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if len(free_joint_ids) == 0:
            raise RuntimeError("No freejoint found for backDrivenCabin root pose")
        self.root_qpos_adr = int(self.model.jnt_qposadr[free_joint_ids[0]])
        self.root_dof_adr = int(self.model.jnt_dofadr[free_joint_ids[0]])

        self.body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in BODY_NAMES
        }
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.joint_qpos_adrs = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_dof_adrs = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32
        )

        self.thruster_body_ids = []
        self.thruster_site_ids = []
        self.thruster_axis_site_ids = []
        for name in THRUSTERS:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site")
            axis_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_thrust_axis"
            )
            if min(body_id, site_id, axis_site_id) < 0:
                raise RuntimeError(f"Thruster '{name}' body/site/axis site not found")
            self.thruster_body_ids.append(body_id)
            self.thruster_site_ids.append(site_id)
            self.thruster_axis_site_ids.append(axis_site_id)

        self._rng = np.random.default_rng()
        self._steps = 0
        self._row_index = 0
        self._target = self.path_targets[0]
        self._last_action = np.zeros(len(THRUSTERS) + len(JOINT_NAMES), dtype=np.float32)
        self._last_reward_terms = {}

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(THRUSTERS) + len(JOINT_NAMES),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(44,),
            dtype=np.float32,
        )

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _target_for_time(self, sim_time):
        self._row_index = path_row_index_from_time(
            sim_time,
            self.row_update_interval,
            len(self.path_targets),
        )
        self._target = self.path_targets[self._row_index]
        return self._target

    def _target_joint_rad(self, target):
        return np.deg2rad([planner_joint_deg_to_mujoco(v) for v in target["joint_deg"]])

    def _apply_initial_pose(self, target):
        back_pose = target["back_pose"]
        root_pos = planner_position_to_mujoco(back_pose[0], back_pose[1])
        yaw = planner_yaw_to_mujoco_root_yaw(back_pose[2])
        if self.randomize_initial_pose:
            root_pos[:2] += self._rng.uniform(-0.03, 0.03, size=2)
            yaw += self._rng.uniform(-np.deg2rad(3.0), np.deg2rad(3.0))
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = root_pos
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = [
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        ]
        self.data.qvel[self.root_dof_adr : self.root_dof_adr + 6] = 0.0
        self.data.qpos[self.joint_qpos_adrs] = self._target_joint_rad(target)
        self.data.qvel[self.joint_dof_adrs] = 0.0

    def _body_pose_planner(self, body_name):
        body_id = self.body_ids[body_name]
        pos = mujoco_position_to_planner(self.data.xpos[body_id])
        rotation = self.data.xmat[body_id].reshape(3, 3)
        forward_mj = rotation @ np.array([0.0, -1.0, 0.0])
        yaw = np.arctan2(-forward_mj[1], forward_mj[0])
        return np.array([pos[0], pos[1], yaw], dtype=np.float64)

    def _body_velocity_planner(self, body_name):
        body_id = self.body_ids[body_name]
        cvel = self.data.cvel[body_id]
        ang = cvel[:3]
        lin = cvel[3:]
        return np.array([lin[0], -lin[1], -ang[2]], dtype=np.float64)

    def _pose_error(self, pose, target_pose):
        return np.array(
            [
                pose[0] - target_pose[0],
                pose[1] - target_pose[1],
                wrap_to_pi(pose[2] - target_pose[2]),
            ],
            dtype=np.float64,
        )

    def _target_feature_vector(self, target):
        return np.array(
            [
                target["time"] / max(self.path_total_time, 1.0),
                target["back_pose"][0] / 5.0,
                target["back_pose"][1] / 5.0,
                target["back_pose"][2] / np.pi,
                np.deg2rad(target["joint_deg"][0]) / np.pi,
                np.deg2rad(target["joint_deg"][1]) / np.pi,
                target["front_pose"][0] / 5.0,
                target["front_pose"][1] / 5.0,
                target["front_pose"][2] / np.pi,
                target["head_pose"][0] / 5.0,
                target["head_pose"][1] / 5.0,
                target["head_pose"][2] / np.pi,
            ],
            dtype=np.float64,
        )

    def _get_obs(self):
        target = self._target_for_time(min(self.data.time, max(self.path_total_time - 1e-9, 0.0)))
        poses = {
            "backDrivenCabin": self._body_pose_planner("backDrivenCabin"),
            "frontDrivenCabin": self._body_pose_planner("frontDrivenCabin"),
            "headCabin": self._body_pose_planner("headCabin"),
        }
        target_poses = {
            "backDrivenCabin": target["back_pose"],
            "frontDrivenCabin": target["front_pose"],
            "headCabin": target["head_pose"],
        }
        pose_errors = []
        for name in BODY_NAMES:
            err = self._pose_error(poses[name], target_poses[name])
            pose_errors.extend([err[0] / 5.0, err[1] / 5.0, err[2] / np.pi])

        joint_target = self._target_joint_rad(target)
        q = self.data.qpos[self.joint_qpos_adrs]
        dq = self.data.qvel[self.joint_dof_adrs]
        joint_features = np.array(
            [
                wrap_to_pi(q[0] - joint_target[0]) / np.pi,
                wrap_to_pi(q[1] - joint_target[1]) / np.pi,
                dq[0] / 5.0,
                dq[1] / 5.0,
            ],
            dtype=np.float64,
        )

        velocities = []
        for name in BODY_NAMES:
            vel = self._body_velocity_planner(name)
            velocities.extend([vel[0] / 2.0, vel[1] / 2.0, vel[2] / 2.0])

        obs = np.concatenate(
            [
                self._target_feature_vector(target),
                np.asarray(pose_errors, dtype=np.float64),
                joint_features,
                np.asarray(velocities, dtype=np.float64),
                self._last_action.astype(np.float64),
            ]
        )
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    def _apply_thrusters(self, action, qfrc):
        forces = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * self.max_thrust
        for force_value, body_id, site_id, axis_site_id in zip(
            forces,
            self.thruster_body_ids,
            self.thruster_site_ids,
            self.thruster_axis_site_ids,
        ):
            point = self.data.site_xpos[site_id].copy()
            direction = self.data.site_xpos[axis_site_id] - self.data.site_xpos[site_id]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction / norm
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_value * direction,
                np.zeros(3, dtype=np.float64),
                point,
                body_id,
                qfrc,
            )
        return forces

    def _apply_joint_action(self, joint_action):
        joint_pos = np.clip(np.asarray(joint_action, dtype=np.float64), -1.0, 1.0) * self.max_joint_angle
        self.data.qpos[self.joint_qpos_adrs] = joint_pos
        self.data.qvel[self.joint_dof_adrs] = 0.0
        return joint_pos

    def _reward(self, forces):
        target = self._target
        pose_targets = {
            "backDrivenCabin": target["back_pose"],
            "frontDrivenCabin": target["front_pose"],
            "headCabin": target["head_pose"],
        }
        pos_err_sum = 0.0
        yaw_err_sum = 0.0
        body_errors = {}
        for name in BODY_NAMES:
            err = self._pose_error(self._body_pose_planner(name), pose_targets[name])
            body_errors[name] = err
            pos_err_sum += float(np.linalg.norm(err[:2]))
            yaw_err_sum += abs(float(err[2]))

        joint_target = self._target_joint_rad(target)
        joint_err = np.array(
            [
                wrap_to_pi(self.data.qpos[self.joint_qpos_adrs[0]] - joint_target[0]),
                wrap_to_pi(self.data.qpos[self.joint_qpos_adrs[1]] - joint_target[1]),
            ],
            dtype=np.float64,
        )
        vel_cost = 0.0
        for name in BODY_NAMES:
            vel = self._body_velocity_planner(name)
            vel_cost += float(np.dot(vel, vel))

        thrust_cost = float(np.mean((forces / max(self.max_thrust, 1e-6)) ** 2))
        reward = (
            8.0 * np.exp(-1.3 * pos_err_sum)
            + 3.0 * np.exp(-1.8 * yaw_err_sum)
            + 2.0 * np.exp(-5.0 * float(np.linalg.norm(joint_err)))
            - 0.02 * vel_cost
            - 0.04 * thrust_cost
        )
        self._last_reward_terms = {
            "pos_err_sum": pos_err_sum,
            "yaw_err_sum_deg": float(np.rad2deg(yaw_err_sum)),
            "joint_err_deg": float(np.rad2deg(np.linalg.norm(joint_err))),
            "thrust_cost": thrust_cost,
            "back_pos_err": float(np.linalg.norm(body_errors["backDrivenCabin"][:2])),
        }
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._steps = 0
        self._row_index = 0
        self._target = self.path_targets[0]
        self._last_action[:] = 0.0
        self._apply_initial_pose(self._target)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = action.copy()
        thruster_action = action[: len(THRUSTERS)]
        joint_action = action[len(THRUSTERS) :]

        forces = np.zeros(len(THRUSTERS), dtype=np.float64)
        joint_pos = np.zeros(2, dtype=np.float64)
        for _ in range(self.frame_skip):
            self._target_for_time(min(self.data.time, max(self.path_total_time - 1e-9, 0.0)))
            joint_pos = self._apply_joint_action(joint_action)
            mujoco.mj_forward(self.model, self.data)
            qfrc = np.zeros(self.model.nv, dtype=np.float64)
            forces = self._apply_thrusters(thruster_action, qfrc)
            self.data.qfrc_applied[:] = qfrc
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        reward = self._reward(forces)

        back_err = self._last_reward_terms.get("back_pos_err", 0.0)
        terminated = bool(back_err > self.max_tracking_error or not np.isfinite(obs).all())
        truncated = bool(self.data.time >= self.path_total_time)
        info = {
            "row_index": self._row_index,
            "file_time": self._target["time"],
            "sim_time": float(self.data.time),
            "forces": forces.copy(),
            "joint_action_deg": np.rad2deg(joint_pos),
            **self._last_reward_terms,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
