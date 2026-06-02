import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class CarAllHingeSinePathEnv(gym.Env):
    """MuJoCo sine-path tracking task for carAll_hinge.xml."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=None,
        path_speed=0.15,
        path_amplitude=0.25,
        path_wavelength=1.2,
        frame_skip=10,
        max_steps=350,
        max_torque=0.25,
        max_wheel_speed=40.0,
        wheel_velocity_kp=0.02,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(script_dir, "3dModels", "carAll_hinge.xml")
        self.path_speed = float(path_speed)
        self.path_amplitude = float(path_amplitude)
        self.path_wavelength = float(path_wavelength)
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.max_torque = float(max_torque)
        self.max_wheel_speed = float(max_wheel_speed)
        self.wheel_velocity_kp = float(wheel_velocity_kp)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if self.chassis_id < 0:
            raise RuntimeError("Body 'chassis' not found")

        self.wheel_names = (
            "wheel_fl_hinge",
            "wheel_fr_hinge",
            "wheel_rl_hinge",
            "wheel_rr_hinge",
        )
        self.wheel_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.wheel_names
            ],
            dtype=np.int32,
        )
        if np.any(self.wheel_joint_ids < 0):
            raise RuntimeError(f"Could not find wheel joints: {self.wheel_names}")

        self.wheel_qpos_adrs = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.wheel_joint_ids],
            dtype=np.int32,
        )
        self.wheel_dof_adrs = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.wheel_joint_ids],
            dtype=np.int32,
        )

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self._last_xpos = np.zeros(3, dtype=np.float64)
        self._last_vel_world = np.zeros(3, dtype=np.float64)
        self._last_wheel_vel = np.zeros(4, dtype=np.float64)
        self._last_action = np.zeros(4, dtype=np.float64)
        self._rng = np.random.default_rng()
        self._steps = 0

        obs_low = np.full(26, -np.inf, dtype=np.float32)
        obs_high = np.full(26, np.inf, dtype=np.float32)
        obs_low[18] = 0.0
        obs_high[18] = 1.0
        obs_low[19:23] = -1.0
        obs_high[19:23] = 1.0
        obs_low[23] = 0.0
        obs_high[23] = 1.0
        obs_low[24] = 0.0
        obs_high[24] = 2.0
        obs_low[25] = 0.1
        obs_high[25] = 5.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _chassis_axes(self):
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        forward_axis = xmat[:, 0].copy()
        up_axis = xmat[:, 1].copy()
        return forward_axis, up_axis

    def _base_velocity_world(self):
        pos = self.data.xpos[self.chassis_id].copy()
        return (pos - self._last_xpos) / max(self.control_dt, 1e-9)

    def _wrap_angle(self, angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _path_target(self):
        x = self.path_speed * self._steps * self.control_dt
        k = 2.0 * np.pi / self.path_wavelength
        y = self.path_amplitude * np.sin(k * x)
        dydx = self.path_amplitude * k * np.cos(k * x)
        heading = np.arctan2(dydx, 1.0)
        return float(x), float(y), float(heading)

    def _pose_error(self):
        x_ref, y_ref, heading_ref = self._path_target()
        pos = self.data.xpos[self.chassis_id]
        forward_axis, _ = self._chassis_axes()
        heading = np.arctan2(forward_axis[1], forward_axis[0])
        dx = x_ref - float(pos[0])
        dy = y_ref - float(pos[1])
        c = np.cos(heading)
        s = np.sin(heading)
        error_body_x = c * dx + s * dy
        error_body_y = -s * dx + c * dy
        heading_error = self._wrap_angle(heading_ref - heading)
        return error_body_x, error_body_y, heading_error, x_ref, y_ref, heading_ref, heading

    def _get_obs(self):
        _, up_axis = self._chassis_axes()
        vel_world = self._base_velocity_world()
        body_acc = (vel_world - self._last_vel_world) / max(self.control_dt, 1e-9)
        wheel_vel = self.data.qvel[self.wheel_dof_adrs].astype(np.float64)
        wheel_acc = (wheel_vel - self._last_wheel_vel) / max(self.control_dt, 1e-9)
        wheel_vel_obs = wheel_vel / max(self.max_wheel_speed, 1e-9)
        wheel_acc_obs = wheel_acc / max(self.max_wheel_speed / max(self.control_dt, 1e-9), 1e-9)
        ex_b, ey_b, heading_error, _, _, _, _ = self._pose_error()
        return np.array(
            [
                ex_b,
                ey_b,
                np.sin(heading_error),
                np.cos(heading_error),
                vel_world[0],
                vel_world[1],
                vel_world[2],
                body_acc[0],
                body_acc[1],
                body_acc[2],
                wheel_vel_obs[0],
                wheel_vel_obs[1],
                wheel_vel_obs[2],
                wheel_vel_obs[3],
                wheel_acc_obs[0],
                wheel_acc_obs[1],
                wheel_acc_obs[2],
                wheel_acc_obs[3],
                up_axis[2],
                self._last_action[0],
                self._last_action[1],
                self._last_action[2],
                self._last_action[3],
                self._steps / max(self.max_steps, 1),
                self.path_amplitude,
                self.path_wavelength,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self.data.qpos[self.wheel_qpos_adrs] = self._rng.uniform(-np.pi, np.pi, size=4)
        self.data.qvel[self.wheel_dof_adrs] = self._rng.normal(0.0, 0.1, size=4)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._last_action[:] = 0.0
        self._last_xpos = self.data.xpos[self.chassis_id].copy()
        self._last_vel_world[:] = 0.0
        self._last_wheel_vel = self.data.qvel[self.wheel_dof_adrs].astype(np.float64)
        return self._get_obs(), {}

    def step(self, action):
        action_value = np.asarray(action, dtype=np.float32).reshape(-1)[:4]
        if action_value.size != 4:
            raise ValueError(f"Expected 4 wheel actions, got {action_value.size}")
        self._last_action = np.clip(action_value, -1.0, 1.0).astype(np.float64)
        target_wheel_vel = self._last_action * self.max_wheel_speed

        start_xpos = self.data.xpos[self.chassis_id].copy()
        start_vel_world = self._base_velocity_world()
        start_wheel_vel = self.data.qvel[self.wheel_dof_adrs].astype(np.float64)
        torque = np.zeros(4, dtype=np.float64)
        for _ in range(self.frame_skip):
            wheel_vel = self.data.qvel[self.wheel_dof_adrs].astype(np.float64)
            torque = self.wheel_velocity_kp * (target_wheel_vel - wheel_vel)
            torque = np.clip(torque, -self.max_torque, self.max_torque)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.wheel_dof_adrs] = torque
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        self._last_xpos = start_xpos
        self._last_vel_world = start_vel_world
        self._last_wheel_vel = start_wheel_vel
        obs = self._get_obs()

        ex_b, ey_b, heading_error, x_ref, y_ref, heading_ref, heading = self._pose_error()
        path_error = float(np.hypot(ex_b, ey_b))
        heading_abs_error = abs(heading_error)
        reward = (
            4.0 * np.exp(-30.0 * path_error * path_error)
            + 1.0 * np.exp(-4.0 * heading_error * heading_error)
            - 8.0 * path_error * path_error
            - 0.5 * heading_error * heading_error
        )

        up_z = float(obs[18])
        chassis_height = float(self.data.xpos[self.chassis_id][2])
        terminated = False
        truncated = self._steps >= self.max_steps
        info = {
            "path_error": path_error,
            "heading_error": heading_error,
            "heading_abs_error": heading_abs_error,
            "target_x": x_ref,
            "target_y": y_ref,
            "target_heading": heading_ref,
            "x": float(self.data.xpos[self.chassis_id][0]),
            "y": float(self.data.xpos[self.chassis_id][1]),
            "heading": heading,
            "up_z": up_z,
            "chassis_height": chassis_height,
            "wheel_torque_fl": float(torque[0]),
            "wheel_torque_fr": float(torque[1]),
            "wheel_torque_rl": float(torque[2]),
            "wheel_torque_rr": float(torque[3]),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass
