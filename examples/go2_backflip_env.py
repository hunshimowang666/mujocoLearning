import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class Go2BackflipEnv(gym.Env):
    """Torque-control task for training a MuJoCo Go2 backflip attempt."""

    metadata = {"render_modes": []}

    JOINT_NAMES = (
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

    HIP = np.array([0, 3, 6, 9], dtype=np.int32)
    THIGH = np.array([1, 4, 7, 10], dtype=np.int32)
    CALF = np.array([2, 5, 8, 11], dtype=np.int32)
    FRONT_THIGH = np.array([1, 4], dtype=np.int32)
    FRONT_CALF = np.array([2, 5], dtype=np.int32)
    REAR_THIGH = np.array([7, 10], dtype=np.int32)
    REAR_CALF = np.array([8, 11], dtype=np.int32)

    HOME_DEG = np.array([0.0, 51.6, -103.1] * 4, dtype=np.float64)

    def __init__(
        self,
        xml_path=None,
        frame_skip=10,
        max_steps=160,
        hip_torque_limit=23.7,
        thigh_torque_limit=23.7,
        calf_torque_limit=45.43,
        initial_joint_noise_deg=1.0,
        initial_joint_vel_noise=0.05,
        torque_smoothing=1.0,
        scripted_torque_scale=0.0,
        reference_kp=(32.0, 72.0, 86.0),
        reference_kd=(1.1, 3.0, 3.4),
        min_base_height=0.11,
        body_contact_grace_time=0.20,
        success_time=2.25,
        debug_assist_torque_y=0.0,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(
            script_dir,
            "unitree_rl_mjlab",
            "src",
            "assets",
            "robots",
            "unitree_go2",
            "xmls",
            "scene_go2_torque.xml",
        )
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.initial_joint_noise = np.deg2rad(initial_joint_noise_deg)
        self.initial_joint_vel_noise = float(initial_joint_vel_noise)
        self.torque_smoothing = float(np.clip(torque_smoothing, 0.0, 1.0))
        self.scripted_torque_scale = float(scripted_torque_scale)
        self.reference_kp = np.array(reference_kp * 4, dtype=np.float64)
        self.reference_kd = np.array(reference_kd * 4, dtype=np.float64)
        self.min_base_height = float(min_base_height)
        self.body_contact_grace_time = float(body_contact_grace_time)
        self.success_time = float(success_time)
        self.debug_assist_torque_y = float(debug_assist_torque_y)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            raise RuntimeError("Body 'base_link' not found")

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [name for name, jid in zip(self.JOINT_NAMES, self.joint_ids) if jid < 0]
            raise RuntimeError(f"Could not find Go2 joints: {missing}")

        self.qpos_adrs = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.dof_adrs = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()

        self.foot_geom_ids = tuple(
            gid
            for gid in (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("FL", "FR", "RL", "RR")
            )
            if gid >= 0
        )

        self.torque_limits = np.array(
            [
                hip_torque_limit,
                thigh_torque_limit,
                calf_torque_limit,
                hip_torque_limit,
                thigh_torque_limit,
                calf_torque_limit,
                hip_torque_limit,
                thigh_torque_limit,
                calf_torque_limit,
                hip_torque_limit,
                thigh_torque_limit,
                calf_torque_limit,
            ],
            dtype=np.float64,
        )

        if self.model.nkey > 0:
            self.home_qpos = self.model.key_qpos[0].copy()
        else:
            self.home_qpos = self.data.qpos.copy()
            self.home_qpos[2] = 0.27
            self.home_qpos[3] = 1.0
        self.home_qpos[2] = max(self.home_qpos[2], 0.285)
        self.home_qvel = np.zeros(self.model.nv, dtype=np.float64)

        self.phase_table = self._make_phase_table()
        self.reference_joint_pos = np.deg2rad(self.HOME_DEG.copy())

        obs_low = np.concatenate(
            [
                np.full(12, -6.0, dtype=np.float32),     # joint reference error
                np.full(12, -120.0, dtype=np.float32),   # joint velocities
                np.array([-1.0, -1.0, 0.0], dtype=np.float32),  # sin, cos, phase
                np.array([-20.0, -2.0, -4.0], dtype=np.float32),  # flip error, flip angle, z
                np.full(3, -20.0, dtype=np.float32),     # base linear velocity
                np.full(3, -80.0, dtype=np.float32),     # base angular velocity
                np.full(6, -1.0, dtype=np.float32),      # body x and z axes
                np.zeros(4, dtype=np.float32),           # foot contact flags
            ]
        )
        obs_high = np.concatenate(
            [
                np.full(12, 6.0, dtype=np.float32),
                np.full(12, 120.0, dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.array([20.0, 20.0, 2.0], dtype=np.float32),
                np.full(3, 20.0, dtype=np.float32),
                np.full(3, 80.0, dtype=np.float32),
                np.full(6, 1.0, dtype=np.float32),
                np.ones(4, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._steps = 0
        self._last_action = np.zeros(12, dtype=np.float64)
        self._filtered_torque = np.zeros(12, dtype=np.float64)
        self.last_torque = np.zeros(12, dtype=np.float64)
        self.last_reference_torque = np.zeros(12, dtype=np.float64)
        self._last_raw_flip_angle = 0.0
        self._unwrapped_flip_angle = 0.0
        self.max_abs_flip_angle = 0.0
        self.max_base_z = 0.0
        self._initial_base_xy = np.zeros(2, dtype=np.float64)

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def episode_duration(self):
        return self.max_steps * self.control_dt

    def _deg_pose(self, hip=0.0, thigh=51.6, calf=-103.1):
        return np.array([hip, thigh, calf] * 4, dtype=np.float64)

    def _make_pose(self, **updates):
        pose = self._deg_pose()
        for key, value in updates.items():
            if key == "front_thigh":
                pose[self.FRONT_THIGH] = value
            elif key == "front_calf":
                pose[self.FRONT_CALF] = value
            elif key == "rear_thigh":
                pose[self.REAR_THIGH] = value
            elif key == "rear_calf":
                pose[self.REAR_CALF] = value
            elif key == "all_thigh":
                pose[self.THIGH] = value
            elif key == "all_calf":
                pose[self.CALF] = value
            elif key == "all_hip":
                pose[self.HIP] = value
            else:
                raise ValueError(f"Unknown pose field: {key}")
        return pose

    def _make_phase_table(self):
        return (
            ("settle", 0.25, self._deg_pose(0.0, 51.6, -103.1), 0.0),
            ("crouch", 0.62, self._deg_pose(0.0, 72.0, -138.0), 0.0),
            (
                "rear_thrust",
                0.78,
                self._make_pose(
                    front_thigh=76.0,
                    front_calf=-142.0,
                    rear_thigh=15.0,
                    rear_calf=-86.0,
                ),
                np.deg2rad(30.0),
            ),
            (
                "full_thrust",
                0.92,
                self._make_pose(
                    front_thigh=10.0,
                    front_calf=-86.0,
                    rear_thigh=3.0,
                    rear_calf=-86.0,
                ),
                np.deg2rad(90.0),
            ),
            ("tuck", 1.35, self._deg_pose(0.0, 96.0, -157.0), np.deg2rad(285.0)),
            ("open", 1.72, self._deg_pose(0.0, 35.0, -94.0), np.deg2rad(350.0)),
            ("recover", 2.30, self._deg_pose(0.0, 51.6, -103.1), 2.0 * np.pi),
            ("hold", 10.0, self._deg_pose(0.0, 51.6, -103.1), 2.0 * np.pi),
        )

    def _episode_time(self):
        return self._steps * self.control_dt

    def _episode_phase(self):
        return min(self._episode_time() / max(self.episode_duration, 1e-9), 1.0)

    def _phase_targets(self):
        t = self._episode_time()
        previous_time = 0.0
        previous_pose = self.HOME_DEG
        previous_angle = 0.0
        for name, end_time, pose_deg, desired_angle in self.phase_table:
            if t <= end_time:
                alpha = (t - previous_time) / max(end_time - previous_time, 1e-9)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                pose = (1.0 - alpha) * previous_pose + alpha * pose_deg
                angle = (1.0 - alpha) * previous_angle + alpha * desired_angle
                return name, np.deg2rad(pose), float(angle)
            previous_time = end_time
            previous_pose = pose_deg
            previous_angle = desired_angle
        name, _, pose_deg, desired_angle = self.phase_table[-1]
        return name, np.deg2rad(pose_deg), float(desired_angle)

    def _joint_pos(self):
        return self.data.qpos[self.qpos_adrs].astype(np.float64)

    def _joint_vel(self):
        return self.data.qvel[self.dof_adrs].astype(np.float64)

    def _body_axes(self):
        xmat = self.data.xmat[self.base_id].reshape(3, 3)
        return xmat[:, 0].copy(), xmat[:, 2].copy()

    def _base_up_z(self):
        _, body_z = self._body_axes()
        return float(body_z[2])

    def _raw_flip_angle(self):
        body_x, _ = self._body_axes()
        return float(np.arctan2(-body_x[2], body_x[0]))

    def _update_unwrapped_flip_angle(self):
        raw = self._raw_flip_angle()
        delta = raw - self._last_raw_flip_angle
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        self._unwrapped_flip_angle += delta
        self._last_raw_flip_angle = raw
        self.max_abs_flip_angle = max(self.max_abs_flip_angle, abs(self._unwrapped_flip_angle))

    def _contact_info(self):
        foot_contacts = np.zeros(4, dtype=np.float64)
        nonfoot_contacts = 0
        hard_nonfoot_contacts = 0
        foot_index = {gid: i for i, gid in enumerate(self.foot_geom_ids)}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            is_foot = False
            if g1 in foot_index:
                foot_contacts[foot_index[g1]] = 1.0
                is_foot = True
            if g2 in foot_index:
                foot_contacts[foot_index[g2]] = 1.0
                is_foot = True
            if not is_foot:
                nonfoot_contacts += 1
                if contact.dist < -0.001:
                    hard_nonfoot_contacts += 1
        return foot_contacts, nonfoot_contacts, hard_nonfoot_contacts

    def _get_obs(self):
        phase_name, target, desired_angle = self._phase_targets()
        self.reference_joint_pos = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        error = self.reference_joint_pos - self._joint_pos()
        dq = self._joint_vel()
        base_z = float(self.data.qpos[2])
        base_lin_vel = self.data.qvel[:3].astype(np.float64)
        base_ang_vel = self.data.qvel[3:6].astype(np.float64)
        body_x, body_z = self._body_axes()
        foot_contacts, _, _ = self._contact_info()
        flip_error = desired_angle - self._unwrapped_flip_angle
        return np.concatenate(
            [
                error,
                dq,
                [np.sin(desired_angle), np.cos(desired_angle), self._episode_phase()],
                [flip_error, self._unwrapped_flip_angle, base_z],
                base_lin_vel,
                base_ang_vel,
                body_x,
                body_z,
                foot_contacts,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = self.home_qvel

        if self.initial_joint_noise > 0.0:
            noise = self._rng.uniform(-self.initial_joint_noise, self.initial_joint_noise, size=12)
            self.data.qpos[self.qpos_adrs] = np.clip(
                self.data.qpos[self.qpos_adrs] + noise,
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
            )
        if self.initial_joint_vel_noise > 0.0:
            self.data.qvel[self.dof_adrs] = self._rng.normal(
                0.0, self.initial_joint_vel_noise, size=12
            )

        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action[:] = 0.0
        self._filtered_torque[:] = 0.0
        self.last_torque[:] = 0.0
        self.last_reference_torque[:] = 0.0
        self._last_raw_flip_angle = self._raw_flip_angle()
        self._unwrapped_flip_angle = 0.0
        self.max_abs_flip_angle = 0.0
        self.max_base_z = float(self.data.qpos[2])
        self._initial_base_xy = self.data.qpos[:2].copy()
        return self._get_obs(), {}

    def step(self, action):
        action_value = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_value.size != 12:
            raise ValueError(f"Expected 12 actions, got {action_value.size}")

        phase_name, target, desired_angle = self._phase_targets()
        self.reference_joint_pos = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        q = self._joint_pos()
        dq = self._joint_vel()
        joint_error = self.reference_joint_pos - q
        reference_torque = np.clip(
            self.reference_kp * joint_error - self.reference_kd * dq,
            -self.torque_limits,
            self.torque_limits,
        )

        previous_action = self._last_action.copy()
        self._last_action = np.clip(action_value, -1.0, 1.0).astype(np.float64)
        rl_torque = self._last_action * self.torque_limits
        desired_torque = rl_torque + self.scripted_torque_scale * reference_torque
        desired_torque = np.clip(desired_torque, -self.torque_limits, self.torque_limits)
        torque = (
            (1.0 - self.torque_smoothing) * self._filtered_torque
            + self.torque_smoothing * desired_torque
        )
        self._filtered_torque = torque.copy()
        self.last_torque = torque.copy()
        self.last_reference_torque = reference_torque.copy()

        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_adrs] = torque
            if self.debug_assist_torque_y != 0.0 and 0.75 <= self._episode_time() <= 1.05:
                self.data.qfrc_applied[4] += self.debug_assist_torque_y
            mujoco.mj_step(self.model, self.data)
            self._update_unwrapped_flip_angle()

        self._steps += 1

        obs = self._get_obs()
        q = self._joint_pos()
        dq = self._joint_vel()
        joint_error = self.reference_joint_pos - q
        base_z = float(self.data.qpos[2])
        self.max_base_z = max(self.max_base_z, base_z)
        base_up_z = self._base_up_z()
        base_lin_vel = self.data.qvel[:3].astype(np.float64)
        base_ang_vel = self.data.qvel[3:6].astype(np.float64)
        foot_contacts, nonfoot_contacts, hard_nonfoot_contacts = self._contact_info()
        foot_contact_count = float(np.sum(foot_contacts))
        flip_angle = float(self._unwrapped_flip_angle)
        flip_error = float(desired_angle - flip_angle)
        final_error = float(2.0 * np.pi - flip_angle)
        elapsed = self._episode_time()

        mean_abs_joint_error = float(np.mean(np.abs(joint_error)))
        normalized_torque = torque / np.maximum(self.torque_limits, 1e-6)
        action_change = self._last_action - previous_action

        reference_reward = 1.2 * float(
            np.exp(-np.mean(np.square(joint_error)) / (0.45**2))
        )
        angle_std = 1.5 if elapsed < 1.55 else 0.75
        angle_reward = 5.0 * float(np.exp(-(flip_error * flip_error) / (angle_std * angle_std)))
        progress_reward = 0.6 * float(np.clip(flip_angle / (2.0 * np.pi), -0.5, 1.2))

        jump_reward = 0.0
        if 0.55 <= elapsed <= 1.25:
            jump_reward += 2.0 * float(np.clip((base_z - 0.28) / 0.28, 0.0, 1.0))
            jump_reward += 1.2 * float(np.clip(base_lin_vel[2] / 2.0, 0.0, 1.0))
            jump_reward += 1.4 * float(np.clip(base_ang_vel[1] / 12.0, 0.0, 1.0))
            if foot_contact_count == 0 and base_z > 0.30:
                jump_reward += 1.5

        air_reward = 0.0
        if 0.85 <= elapsed <= 1.65:
            air_reward += 0.8 * float(foot_contact_count == 0)
            air_reward += 0.8 * float(np.clip((base_z - 0.20) / 0.25, 0.0, 1.0))

        landing_reward = 0.0
        success = False
        if elapsed >= 1.75:
            final_angle_reward = float(np.exp(-(final_error * final_error) / (0.55**2)))
            upright_reward = float(np.clip((base_up_z - 0.45) / 0.55, 0.0, 1.0))
            height_reward = float(np.exp(-((base_z - 0.27) ** 2) / (0.12**2)))
            contact_reward = float(np.clip(foot_contact_count / 4.0, 0.0, 1.0))
            landing_reward = 10.0 * final_angle_reward * upright_reward
            landing_reward += 2.0 * height_reward + 1.5 * contact_reward
            success = (
                elapsed >= self.success_time
                and abs(final_error) < np.deg2rad(35.0)
                and base_up_z > 0.70
                and base_z > 0.16
                and foot_contact_count >= 2
                and nonfoot_contacts == 0
                and np.linalg.norm(base_ang_vel) < 7.0
            )
            if success:
                landing_reward += 60.0

        alive_reward = 0.4 * float(np.clip((base_z - self.min_base_height) / 0.20, 0.0, 1.0))
        drift_penalty = 0.6 * float(np.linalg.norm(self.data.qpos[:2] - self._initial_base_xy))
        body_contact_penalty = 0.6 * float(nonfoot_contacts) + 8.0 * float(hard_nonfoot_contacts)
        reward = (
            alive_reward
            + reference_reward
            + angle_reward
            + progress_reward
            + jump_reward
            + air_reward
            + landing_reward
            - 0.004 * float(np.mean(np.square(dq)))
            - 0.025 * float(np.mean(np.square(normalized_torque)))
            - 0.035 * float(np.mean(np.square(action_change)))
            - body_contact_penalty
            - drift_penalty
        )

        too_low = base_z < self.min_base_height and elapsed > 0.15
        body_hit_ground = hard_nonfoot_contacts > 0 and elapsed > self.body_contact_grace_time
        nonfinite = not np.isfinite(obs).all()
        terminated = bool(too_low or body_hit_ground or nonfinite or success)
        if too_low:
            reward -= 40.0
        if body_hit_ground:
            reward -= 60.0
        if nonfinite:
            reward -= 80.0
        truncated = self._steps >= self.max_steps

        info = {
            "phase_name": phase_name,
            "phase": float(self._episode_phase()),
            "elapsed": float(elapsed),
            "base_z": base_z,
            "max_base_z": float(self.max_base_z),
            "base_up_z": float(base_up_z),
            "base_vz": float(base_lin_vel[2]),
            "base_ang_vel_y": float(base_ang_vel[1]),
            "flip_angle_deg": float(np.rad2deg(flip_angle)),
            "desired_flip_angle_deg": float(np.rad2deg(desired_angle)),
            "flip_error_deg": float(np.rad2deg(flip_error)),
            "final_flip_error_deg": float(np.rad2deg(final_error)),
            "max_abs_flip_angle_deg": float(np.rad2deg(self.max_abs_flip_angle)),
            "foot_contacts": foot_contact_count,
            "nonfoot_contacts": int(nonfoot_contacts),
            "hard_nonfoot_contacts": int(hard_nonfoot_contacts),
            "mean_joint_ref_error_deg": float(np.rad2deg(mean_abs_joint_error)),
            "torque_rms": float(np.sqrt(np.mean(np.square(torque)))),
            "reference_torque_rms": float(np.sqrt(np.mean(np.square(reference_torque)))),
            "success": bool(success),
            "too_low": bool(too_low),
            "body_hit_ground": bool(body_hit_ground),
            "nonfinite": bool(nonfinite),
            "reward_angle": float(angle_reward),
            "reward_jump": float(jump_reward),
            "reward_air": float(air_reward),
            "reward_landing": float(landing_reward),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
