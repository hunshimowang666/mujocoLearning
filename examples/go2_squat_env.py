import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class Go2SquatEnv(gym.Env):
    """Torque-control task for making Unitree Go2 squat to a fixed joint pose."""

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
        target_hip_deg=0.0,
        target_thigh_deg=68.8,
        target_calf_deg=-131.8,
        frame_skip=10,
        max_steps=500,
        hip_torque_limit=23.7,
        thigh_torque_limit=23.7,
        calf_torque_limit=45.43,
        initial_joint_noise_deg=1.0,
        initial_joint_vel_noise=0.05,
        target_base_height=0.17,
        min_base_height=0.08,
        min_base_up_z=0.65,
        reference_kp=(18.0, 45.0, 55.0),
        reference_kd=(0.8, 2.0, 2.5),
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
        self.target_base_height = float(target_base_height)
        self.min_base_height = float(min_base_height)
        self.min_base_up_z = float(min_base_up_z)
        self.reference_kp = np.array(reference_kp * 4, dtype=np.float64)
        self.reference_kd = np.array(reference_kd * 4, dtype=np.float64)

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

        self.target_joint_pos = self._make_target_joint_pos(
            target_hip_deg, target_thigh_deg, target_calf_deg
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
        self.home_qvel = np.zeros(self.model.nv, dtype=np.float64)

        obs_low = np.concatenate(
            [
                np.full(len(self.JOINT_NAMES), -6.0, dtype=np.float32),
                np.full(len(self.JOINT_NAMES), -80.0, dtype=np.float32),
            ]
        )
        obs_high = -obs_low
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.JOINT_NAMES),), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._steps = 0
        self._last_action = np.zeros(len(self.JOINT_NAMES), dtype=np.float64)
        self.last_torque = np.zeros(len(self.JOINT_NAMES), dtype=np.float64)

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def episode_duration(self):
        return self.max_steps * self.control_dt

    def _make_target_joint_pos(self, hip_deg, thigh_deg, calf_deg):
        target = np.deg2rad(np.array([hip_deg, thigh_deg, calf_deg] * 4, dtype=np.float64))
        return np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _base_up_z(self):
        xmat = self.data.xmat[self.base_id].reshape(3, 3)
        return float(xmat[:, 2][2])

    def _joint_pos(self):
        return self.data.qpos[self.qpos_adrs].astype(np.float64)

    def _joint_vel(self):
        return self.data.qvel[self.dof_adrs].astype(np.float64)

    def _get_obs(self):
        error = self.target_joint_pos - self._joint_pos()
        dq = self._joint_vel()
        return np.concatenate([error, dq]).astype(np.float32)

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
        return self._get_obs(), {}

    def step(self, action):
        action_value = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_value.size != len(self.JOINT_NAMES):
            raise ValueError(f"Expected {len(self.JOINT_NAMES)} actions, got {action_value.size}")

        previous_action = self._last_action.copy()
        self._last_action = np.clip(action_value, -1.0, 1.0).astype(np.float64)
        torque = self._last_action * self.torque_limits
        self.last_torque = torque.copy()

        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_adrs] = torque
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        error = obs[: len(self.JOINT_NAMES)].astype(np.float64)
        dq = obs[len(self.JOINT_NAMES) :].astype(np.float64)
        base_z = float(self.data.qpos[2])
        base_up_z = self._base_up_z()

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
        fine_reward = 8.0 * float(np.exp(-joint_error_l2 / (0.10**2)))
        height_reward = 0.5 * float(np.exp(-((base_z - self.target_base_height) ** 2) / (0.06**2)))
        upright_reward = 2.0 * float(np.clip((base_up_z - self.min_base_up_z) / 0.3, 0.0, 1.0))
        reference_reward = 2.0 * float(
            np.exp(-np.mean(np.square(normalized_torque - normalized_reference_torque)) / 0.20)
        )
        reward = (
            tracking_reward
            + fine_reward
            + height_reward
            + upright_reward
            + reference_reward
            - 5.0 * mean_abs_error
            - 0.6 * max_abs_error
            - 0.01 * float(np.mean(np.square(dq)))
            - 0.03 * float(np.mean(np.square(normalized_torque)))
            - 0.02 * float(np.mean(np.square(action_change)))
        )

        settled = mean_abs_error < np.deg2rad(3.0) and mean_abs_dq < 0.35
        if settled:
            reward += 6.0
        if mean_abs_error < np.deg2rad(1.5) and mean_abs_dq < 0.2:
            reward += 6.0

        too_low = base_z < self.min_base_height
        too_tilted = base_up_z < self.min_base_up_z
        nonfinite = not np.isfinite(obs).all()
        terminated = bool(too_low or too_tilted or nonfinite)
        if terminated:
            reward -= 50.0
        truncated = self._steps >= self.max_steps

        info = {
            "mean_error_deg": float(np.rad2deg(mean_abs_error)),
            "max_error_deg": float(np.rad2deg(max_abs_error)),
            "mean_dq_deg_s": float(np.rad2deg(mean_abs_dq)),
            "base_z": base_z,
            "base_up_z": base_up_z,
            "torque_rms": float(np.sqrt(np.mean(np.square(torque)))),
            "reference_torque_rms": float(np.sqrt(np.mean(np.square(reference_torque)))),
            "settled": bool(settled),
            "too_low": bool(too_low),
            "too_tilted": bool(too_tilted),
            "nonfinite": bool(nonfinite),
            "hip_q_deg": float(np.rad2deg(np.mean(self._joint_pos()[[0, 3, 6, 9]]))),
            "thigh_q_deg": float(np.rad2deg(np.mean(self._joint_pos()[[1, 4, 7, 10]]))),
            "calf_q_deg": float(np.rad2deg(np.mean(self._joint_pos()[[2, 5, 8, 11]]))),
            "target_hip_deg": float(np.rad2deg(np.mean(self.target_joint_pos[[0, 3, 6, 9]]))),
            "target_thigh_deg": float(np.rad2deg(np.mean(self.target_joint_pos[[1, 4, 7, 10]]))),
            "target_calf_deg": float(np.rad2deg(np.mean(self.target_joint_pos[[2, 5, 8, 11]]))),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
