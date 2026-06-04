import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class Go2JointTableEnv(gym.Env):
    """Torque-control task for tracking a table of Go2 joint targets."""

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

    def __init__(
        self,
        xml_path=None,
        target_table_path=None,
        target_row_duration=0.8,
        interpolate_targets=True,
        frame_skip=10,
        max_steps=300,
        hip_torque_limit=23.7,
        thigh_torque_limit=23.7,
        calf_torque_limit=45.43,
        initial_joint_noise_deg=1.0,
        initial_joint_vel_noise=0.05,
        min_base_height=0.08,
        min_base_up_z=0.65,
        reference_kp=(18.0, 45.0, 55.0),
        reference_kd=(0.8, 2.0, 2.5),
        torque_smoothing=1.0,
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
        self.target_table_path = target_table_path or os.path.join(
            script_dir, "go2_joint_target_angles.csv"
        )
        self.target_row_duration = float(target_row_duration)
        self.interpolate_targets = bool(interpolate_targets)
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.initial_joint_noise = np.deg2rad(initial_joint_noise_deg)
        self.initial_joint_vel_noise = float(initial_joint_vel_noise)
        self.min_base_height = float(min_base_height)
        self.min_base_up_z = float(min_base_up_z)
        self.reference_kp = np.array(reference_kp * 4, dtype=np.float64)
        self.reference_kd = np.array(reference_kd * 4, dtype=np.float64)
        self.torque_smoothing = float(np.clip(torque_smoothing, 0.0, 1.0))

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

        self.target_table = self._load_target_table(self.target_table_path)
        self.target_joint_pos = self.target_table[0].copy()
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
        self.home_qvel = np.zeros(self.model.nv, dtype=np.float64)

        obs_low = np.concatenate(
            [
                np.full(len(self.JOINT_NAMES), -6.0, dtype=np.float32),
                np.full(len(self.JOINT_NAMES), -80.0, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            ]
        )
        obs_high = np.concatenate(
            [
                np.full(len(self.JOINT_NAMES), 6.0, dtype=np.float32),
                np.full(len(self.JOINT_NAMES), 80.0, dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.JOINT_NAMES),), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._steps = 0
        self._last_action = np.zeros(len(self.JOINT_NAMES), dtype=np.float64)
        self.last_torque = np.zeros(len(self.JOINT_NAMES), dtype=np.float64)
        self._filtered_torque = np.zeros(len(self.JOINT_NAMES), dtype=np.float64)
        self._initial_base_xy = np.zeros(2, dtype=np.float64)

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def episode_duration(self):
        return self.max_steps * self.control_dt

    def _load_target_table(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target angle table not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        has_header = False
        try:
            [float(v) for v in first_line.split(",")]
        except ValueError:
            has_header = True

        if has_header:
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
            table_deg = np.column_stack([data[name] for name in self.JOINT_NAMES])
        else:
            table_deg = np.loadtxt(path, delimiter=",", dtype=np.float64)
            table_deg = np.atleast_2d(table_deg)

        if table_deg.shape[1] != len(self.JOINT_NAMES):
            raise ValueError(
                f"Expected {len(self.JOINT_NAMES)} target columns, got {table_deg.shape[1]}"
            )
        table = np.deg2rad(table_deg.astype(np.float64))
        return np.clip(table, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _base_up_z(self):
        xmat = self.data.xmat[self.base_id].reshape(3, 3)
        return float(xmat[:, 2][2])

    def _joint_pos(self):
        return self.data.qpos[self.qpos_adrs].astype(np.float64)

    def _joint_vel(self):
        return self.data.qvel[self.dof_adrs].astype(np.float64)

    def _episode_phase(self):
        return min((self._steps * self.control_dt) / max(self.episode_duration, 1e-9), 1.0)

    def _target_index_and_alpha(self):
        t = self._steps * self.control_dt
        raw = t / max(self.target_row_duration, 1e-9)
        i0 = min(int(np.floor(raw)), len(self.target_table) - 1)
        i1 = min(i0 + 1, len(self.target_table) - 1)
        alpha = float(raw - np.floor(raw)) if self.interpolate_targets and i0 != i1 else 0.0
        return i0, i1, alpha

    def _update_target(self):
        i0, i1, alpha = self._target_index_and_alpha()
        self.target_joint_pos = (1.0 - alpha) * self.target_table[i0] + alpha * self.target_table[i1]

    def _get_obs(self):
        self._update_target()
        error = self.target_joint_pos - self._joint_pos()
        dq = self._joint_vel()
        return np.concatenate([error, dq, [self._episode_phase()]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = self.home_qvel

        if self.initial_joint_noise > 0.0:
            noise = self._rng.uniform(
                -self.initial_joint_noise, self.initial_joint_noise, size=len(self.JOINT_NAMES)
            )
            q = np.clip(
                self.data.qpos[self.qpos_adrs] + noise,
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
            )
            self.data.qpos[self.qpos_adrs] = q
        if self.initial_joint_vel_noise > 0.0:
            self.data.qvel[self.dof_adrs] = self._rng.normal(
                0.0, self.initial_joint_vel_noise, size=len(self.JOINT_NAMES)
            )

        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action[:] = 0.0
        self.last_torque[:] = 0.0
        self._filtered_torque[:] = 0.0
        self._initial_base_xy = self.data.qpos[:2].copy()
        self._update_target()
        return self._get_obs(), {}

    def step(self, action):
        action_value = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_value.size != len(self.JOINT_NAMES):
            raise ValueError(f"Expected {len(self.JOINT_NAMES)} actions, got {action_value.size}")

        self._update_target()
        previous_action = self._last_action.copy()
        self._last_action = np.clip(action_value, -1.0, 1.0).astype(np.float64)
        desired_torque = self._last_action * self.torque_limits
        torque = (
            (1.0 - self.torque_smoothing) * self._filtered_torque
            + self.torque_smoothing * desired_torque
        )
        self._filtered_torque = torque.copy()
        self.last_torque = torque.copy()

        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_adrs] = torque
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        error = obs[: len(self.JOINT_NAMES)].astype(np.float64)
        dq = obs[len(self.JOINT_NAMES) : 2 * len(self.JOINT_NAMES)].astype(np.float64)
        base_z = float(self.data.qpos[2])
        base_up_z = self._base_up_z()
        base_xy_drift = float(np.linalg.norm(self.data.qpos[:2] - self._initial_base_xy))
        base_xy_speed = float(np.linalg.norm(self.data.qvel[:2]))

        mean_abs_error = float(np.mean(np.abs(error)))
        max_abs_error = float(np.max(np.abs(error)))
        mean_abs_dq = float(np.mean(np.abs(dq)))
        normalized_torque = torque / np.maximum(self.torque_limits, 1e-6)
        reference_torque = np.clip(
            self.reference_kp * error - self.reference_kd * dq,
            -self.torque_limits,
            self.torque_limits,
        )
        normalized_reference_torque = reference_torque / np.maximum(self.torque_limits, 1e-6)
        action_change = self._last_action - previous_action

        joint_error_l2 = float(np.mean(np.square(error)))
        tracking_reward = 10.0 * float(np.exp(-joint_error_l2 / (0.35**2)))
        fine_reward = 6.0 * float(np.exp(-joint_error_l2 / (0.10**2)))
        upright_reward = 2.0 * float(np.clip((base_up_z - self.min_base_up_z) / 0.3, 0.0, 1.0))
        alive_height_reward = 0.8 * float(np.clip((base_z - self.min_base_height) / 0.18, 0.0, 1.0))
        reference_reward = 2.0 * float(
            np.exp(-np.mean(np.square(normalized_torque - normalized_reference_torque)) / 0.20)
        )
        reward = (
            tracking_reward
            + fine_reward
            + upright_reward
            + alive_height_reward
            + reference_reward
            - 4.0 * mean_abs_error
            - 0.5 * max_abs_error
            - 8.0 * base_xy_drift
            - 1.2 * base_xy_speed
            - 0.01 * float(np.mean(np.square(dq)))
            - 0.04 * float(np.mean(np.square(normalized_torque)))
            - 0.05 * float(np.mean(np.square(action_change)))
        )

        settled = mean_abs_error < np.deg2rad(3.0) and mean_abs_dq < 0.45
        if settled:
            reward += 4.0

        too_low = base_z < self.min_base_height
        too_tilted = base_up_z < self.min_base_up_z
        nonfinite = not np.isfinite(obs).all()
        terminated = bool(too_low or too_tilted or nonfinite)
        if terminated:
            reward -= 50.0
        truncated = self._steps >= self.max_steps
        i0, _, _ = self._target_index_and_alpha()

        q = self._joint_pos()
        target = self.target_joint_pos
        info = {
            "target_row": int(i0),
            "phase": float(self._episode_phase()),
            "mean_error_deg": float(np.rad2deg(mean_abs_error)),
            "max_error_deg": float(np.rad2deg(max_abs_error)),
            "mean_dq_deg_s": float(np.rad2deg(mean_abs_dq)),
            "base_z": base_z,
            "base_up_z": base_up_z,
            "base_xy_drift": base_xy_drift,
            "base_xy_speed": base_xy_speed,
            "torque_rms": float(np.sqrt(np.mean(np.square(torque)))),
            "desired_torque_rms": float(np.sqrt(np.mean(np.square(desired_torque)))),
            "reference_torque_rms": float(np.sqrt(np.mean(np.square(reference_torque)))),
            "settled": bool(settled),
            "too_low": bool(too_low),
            "too_tilted": bool(too_tilted),
            "nonfinite": bool(nonfinite),
            "hip_q_deg": float(np.rad2deg(np.mean(q[[0, 3, 6, 9]]))),
            "thigh_q_deg": float(np.rad2deg(np.mean(q[[1, 4, 7, 10]]))),
            "calf_q_deg": float(np.rad2deg(np.mean(q[[2, 5, 8, 11]]))),
            "target_hip_deg": float(np.rad2deg(np.mean(target[[0, 3, 6, 9]]))),
            "target_thigh_deg": float(np.rad2deg(np.mean(target[[1, 4, 7, 10]]))),
            "target_calf_deg": float(np.rad2deg(np.mean(target[[2, 5, 8, 11]]))),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
