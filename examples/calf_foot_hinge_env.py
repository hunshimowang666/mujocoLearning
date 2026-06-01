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
        initial_q_deg=7.0,
        target_q_deg=7.0,
        frame_skip=10,
        # 设置每回合最高步数
        max_steps=500,
        max_torque=2.0,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(script_dir, "3dModels", "calf_foot_hinge.xml")
        self.initial_q = np.deg2rad(initial_q_deg)
        self.target_q = np.deg2rad(target_q_deg)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.max_torque = max_torque

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
            low=np.array([-np.pi, -25.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([np.pi, 25.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._steps = 0
        self._last_action = 0.0
        self.external_torque = 0.0

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _get_obs(self):
        q = float(self.data.qpos[self.qpos_adr])
        dq = float(self.data.qvel[self.dof_adr])
        error = self.target_q - q
        calf_z_axis = self.data.xmat[self.calf_id].reshape(3, 3)[:, 2]
        return np.array(
            [
                error,
                dq,
                np.sin(q),
                np.cos(q),
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
        self.data.qpos[self.qpos_adr] = self.initial_q + self._rng.uniform(-0.08, 0.08)
        self.data.qvel[self.dof_adr] = self._rng.uniform(-0.15, 0.15)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action = 0.0
        self.external_torque = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._last_action = float(np.clip(action_value, -1.0, 1.0))
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
        terminated = bool(abs(q - self.target_q) > np.deg2rad(70.0) or calf_height < 0.04)
        truncated = self._steps >= self.max_steps
        info = {
            "q_deg": float(np.rad2deg(q)),
            "dq_deg_s": float(np.rad2deg(dq)),
            "torque": torque,
            "target_q_deg": float(np.rad2deg(self.target_q)),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
