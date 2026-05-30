import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class SimpleBoxHoverEnv(gym.Env):
    """A tiny MuJoCo task: learn vertical force control for simpleBox.xml."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=None,
        target_z=0.5,
        frame_skip=10,
        max_steps=500,
        max_force=30.0,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(script_dir, "3dModels", "simpleBox.xml")
        self.target_z = target_z
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.max_force = max_force

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -20.0, -1.0], dtype=np.float32),
            high=np.array([5.0, 20.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._steps = 0
        self._last_action = 0.0

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _get_obs(self):
        z = float(self.data.body("box").xpos[2])
        vz = float(self.data.qvel[2])
        return np.array([z - self.target_z, vz, self._last_action], dtype=np.float32)

    def _apply_force(self, action_value):
        force = float(np.clip(action_value, -1.0, 1.0)) * self.max_force
        qfrc = np.zeros(self.model.nv, dtype=np.float64)
        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.array([0.0, 0.0, force], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.data.xipos[self.box_id],
            self.box_id,
            qfrc,
        )
        self.data.qfrc_applied[:] = qfrc
        return force

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # freejoint qpos: xyz + quaternion. Start near the target height.
        self.data.qpos[0] = self._rng.uniform(-0.05, 0.05)
        self.data.qpos[1] = self._rng.uniform(-0.05, 0.05)
        self.data.qpos[2] = self.target_z + self._rng.uniform(-0.25, 0.25)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._last_action = float(np.clip(action_value, -1.0, 1.0))

        force = 0.0
        for _ in range(self.frame_skip):
            force = self._apply_force(self._last_action)
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._get_obs()
        z_error, vz, _ = obs

        reward = (
            2.0
            - 8.0 * abs(float(z_error))
            - 0.08 * abs(float(vz))
            - 0.002 * abs(force)
        )
        terminated = bool(abs(float(z_error)) > 1.5)
        truncated = self._steps >= self.max_steps
        info = {"force": force, "z": float(self.data.body("box").xpos[2])}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
