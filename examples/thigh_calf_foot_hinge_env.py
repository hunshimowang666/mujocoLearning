import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class ThighCalfFootHingeEnv(gym.Env):
    """MuJoCo task for torque control of thigh_calf_foot_hinge.xml."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=None,
        initial_q_deg=None,
        target_q_deg=None,
        target_table_path=None,
        frame_skip=10,
        max_steps=500,
        max_torque=2.0,
        max_tracking_error_deg=15.0,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(script_dir, "3dModels", "thigh_calf_foot_hinge.xml")
        self.initial_q = np.deg2rad(initial_q_deg if initial_q_deg is not None else [0.0, 0.0])
        self.target_q = np.deg2rad(target_q_deg if target_q_deg is not None else [0.0, 0.0])
        self.target_table_path = target_table_path
        self.target_table_deg = self._load_target_table(target_table_path)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.max_torque = max_torque
        self.max_tracking_error = np.deg2rad(max_tracking_error_deg)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = ("knee_hinge", "ankle_hinge")
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ])
        if np.any(self.joint_ids < 0):
            raise RuntimeError(f"Could not find joints: {self.joint_names}")

        self.qpos_adrs = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids])
        self.dof_adrs = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids])
        self.thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "thigh")
        self.calf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "calf")
        self.foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.pi, -np.pi, -25.0, -25.0]
                + [-1.0] * 8
                + [0.0, -1.0, -1.0, -1.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [np.pi, np.pi, 25.0, 25.0]
                + [1.0] * 8
                + [1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float64)
        self.external_torque = np.zeros(2, dtype=np.float64)

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
        if table.shape[1] == 1:
            ankle_deg = table[:, 0]
            knee_deg = table[:, 0]
        else:
            # Shared target table convention:
            # column 1 is ankle target for the calf-foot task,
            # column 2 is knee target for the thigh-calf-foot task.
            ankle_deg = table[:, 0]
            knee_deg = table[:, 1]
        return np.column_stack([knee_deg, ankle_deg]).astype(np.float64)

    def _target_from_table(self):
        if self.target_table_deg is None:
            return self.target_q

        t = min(self._steps * self.control_dt, self.episode_duration)
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, len(self.target_table_deg) - 1)
        i0 = min(i0, len(self.target_table_deg) - 1)
        alpha = float(t - np.floor(t)) if i0 != i1 else 0.0
        target_deg = (1.0 - alpha) * self.target_table_deg[i0] + alpha * self.target_table_deg[i1]
        return np.deg2rad(target_deg)

    def _update_target(self):
        self.target_q = self._target_from_table()

    def _get_obs(self):
        self._update_target()
        q = self.data.qpos[self.qpos_adrs].astype(np.float64)
        dq = self.data.qvel[self.dof_adrs].astype(np.float64)
        error = self.target_q - q
        phase = min((self._steps * self.control_dt) / self.episode_duration, 1.0)
        thigh_z = self.data.xmat[self.thigh_id].reshape(3, 3)[:, 2][2]
        calf_z = self.data.xmat[self.calf_id].reshape(3, 3)[:, 2][2]
        return np.array(
            [
                error[0],
                error[1],
                dq[0],
                dq[1],
                np.sin(q[0]),
                np.sin(q[1]),
                np.cos(q[0]),
                np.cos(q[1]),
                np.sin(self.target_q[0]),
                np.sin(self.target_q[1]),
                np.cos(self.target_q[0]),
                np.cos(self.target_q[1]),
                phase,
                thigh_z,
                calf_z,
                self._last_action[0],
                self._last_action[1],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_adrs] = self.initial_q
        self.data.qvel[self.dof_adrs] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action[:] = 0.0
        self.external_torque[:] = 0.0
        self._update_target()
        return self._get_obs(), {}

    def step(self, action):
        action_value = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        self._last_action = np.clip(action_value, -1.0, 1.0).astype(np.float64)
        self._update_target()
        torque = self._last_action * self.max_torque + self.external_torque

        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_adrs] = torque
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        error = obs[:2].astype(np.float64)
        dq = obs[2:4].astype(np.float64)
        q = self.data.qpos[self.qpos_adrs].astype(np.float64)
        foot_height = float(self.data.xpos[self.foot_id][2])

        reward = (
            4.0
            - 8.0 * float(np.sum(np.abs(error)))
            - 0.06 * float(np.sum(np.abs(dq)))
            - 0.04 * float(np.sum(np.abs(torque)))
            - 0.03 * float(np.sum(np.abs(self._last_action)))
        )
        error_too_large = np.any(np.abs(self.target_q - q) > self.max_tracking_error)
        foot_too_low = foot_height < 0.0
        terminated = bool(error_too_large or foot_too_low)
        truncated = self._steps >= self.max_steps
        info = {
            "knee_q_deg": float(np.rad2deg(q[0])),
            "ankle_q_deg": float(np.rad2deg(q[1])),
            "knee_target_deg": float(np.rad2deg(self.target_q[0])),
            "ankle_target_deg": float(np.rad2deg(self.target_q[1])),
            "knee_error_deg": float(np.rad2deg(self.target_q[0] - q[0])),
            "ankle_error_deg": float(np.rad2deg(self.target_q[1] - q[1])),
            "knee_dq_deg_s": float(np.rad2deg(dq[0])),
            "ankle_dq_deg_s": float(np.rad2deg(dq[1])),
            "knee_torque": float(torque[0]),
            "ankle_torque": float(torque[1]),
            "error_too_large": bool(error_too_large),
            "foot_too_low": bool(foot_too_low),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
