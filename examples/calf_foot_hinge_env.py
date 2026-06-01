import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class CalfFootHingeEnv(gym.Env):
    """MuJoCo task for torque control of calf_foot_hinge.xml."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=None,
        initial_q_deg=0.0,
        target_q_deg=0.0,
        target_table_path=None,
        frame_skip=10,
        # 设置每回合最高步数
        max_steps=500,
        max_torque=2.0,
        max_tracking_error_deg=10.0,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(script_dir, "3dModels", "calf_foot_hinge.xml")
        self.initial_q = np.deg2rad(initial_q_deg)
        self.target_q = np.deg2rad(target_q_deg)
        self.target_table_path = target_table_path
        self.target_table_deg = self._load_target_table(target_table_path)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.max_torque = max_torque
        self.max_tracking_error = np.deg2rad(max_tracking_error_deg)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_hinge")
        if self.joint_id < 0:
            raise RuntimeError("Joint 'ankle_hinge' not found")

        self.calf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "calf")
        self.foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot")
        self.qpos_adr = self.model.jnt_qposadr[self.joint_id]
        self.dof_adr = self.model.jnt_dofadr[self.joint_id]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -25.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([np.pi, 25.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._steps = 0
        self._last_action = 0.0
        self.external_torque = 0.0

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def episode_duration(self):
        return self.max_steps * self.control_dt

    def _load_target_table(self, target_table_path):
        if target_table_path is None:
            return None
        if not os.path.exists(target_table_path):
            raise FileNotFoundError(f"Target angle table not found: {target_table_path}")

        table = np.loadtxt(target_table_path, delimiter=",", ndmin=2)
        if table.size == 0:
            raise ValueError(f"Target angle table is empty: {target_table_path}")
        return np.asarray(table[:, 0], dtype=np.float64)

    def _target_from_table(self):
        if self.target_table_deg is None:
            return self.target_q

        t = min(self._steps * self.control_dt, self.episode_duration)
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, len(self.target_table_deg) - 1)
        i0 = min(i0, len(self.target_table_deg) - 1)
        alpha = float(t - np.floor(t)) if i0 != i1 else 0.0
        target_deg = (1.0 - alpha) * self.target_table_deg[i0] + alpha * self.target_table_deg[i1]
        return float(np.deg2rad(target_deg))

    def _update_target(self):
        self.target_q = self._target_from_table()

    def _get_obs(self):
        self._update_target()
        q = float(self.data.qpos[self.qpos_adr])
        dq = float(self.data.qvel[self.dof_adr])
        error = self.target_q - q
        calf_z_axis = self.data.xmat[self.calf_id].reshape(3, 3)[:, 2]
        episode_phase = min((self._steps * self.control_dt) / self.episode_duration, 1.0)
        return np.array(
            [
                error,
                dq,
                np.sin(q),
                np.cos(q),
                np.sin(self.target_q),
                np.cos(self.target_q),
                episode_phase,
                calf_z_axis[2],
                self._last_action,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_adr] = self.initial_q
        self.data.qvel[self.dof_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action = 0.0
        self.external_torque = 0.0
        self._update_target()
        return self._get_obs(), {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._last_action = float(np.clip(action_value, -1.0, 1.0))
        self._update_target()
        torque = self._last_action * self.max_torque + self.external_torque

        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_adr] = torque
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        error = float(obs[0])
        dq = float(obs[1])
        q = float(self.data.qpos[self.qpos_adr])
        calf_height = float(self.data.xpos[self.calf_id][2])

        reward = (
            2.0
            - 8.0 * abs(error)
            - 0.08 * abs(dq)
            - 0.04 * abs(torque)
            - 0.03 * abs(self._last_action)
        )
        error_too_large = abs(self.target_q - q) > self.max_tracking_error
        calf_too_low = calf_height < 0.04
        terminated = bool(error_too_large or calf_too_low)
        truncated = self._steps >= self.max_steps
        info = {
            "q_deg": float(np.rad2deg(q)),
            "dq_deg_s": float(np.rad2deg(dq)),
            "torque": torque,
            "target_q_deg": float(np.rad2deg(self.target_q)),
            "error_deg": float(np.rad2deg(self.target_q - q)),
            "error_too_large": bool(error_too_large),
            "calf_too_low": bool(calf_too_low),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
