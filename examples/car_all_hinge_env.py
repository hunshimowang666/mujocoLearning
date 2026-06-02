import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class CarAllHingeEnv(gym.Env):
    """MuJoCo velocity-tracking task for carAll_hinge.xml."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=None,
        target_speed=0.35,
        frame_skip=10,
        max_steps=250,
        max_torque=0.035,
        max_wheel_speed=60.0,
        wheel_velocity_kp=0.02,
        max_lateral_drift=0.35,
    ):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = xml_path or os.path.join(script_dir, "3dModels", "carAll_hinge.xml")
        self.target_speed = float(target_speed)
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.max_torque = float(max_torque)
        self.max_wheel_speed = float(max_wheel_speed)
        self.wheel_velocity_kp = float(wheel_velocity_kp)
        self.max_lateral_drift = float(max_lateral_drift)

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

        obs_low = np.array(
            [
                -4.0,
                0.0,
                -4.0,
                -4.0,
                -4.0,
                -50.0,
                -50.0,
                -50.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                0.0,
                -self.max_lateral_drift,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                50.0,
                50.0,
                50.0,
                2.0,
                2.0,
                2.0,
                2.0,
                5.0,
                5.0,
                5.0,
                5.0,
                1.0,
                self.max_lateral_drift,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def episode_duration(self):
        return self.max_steps * self.control_dt

    def _chassis_axes(self):
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        forward_axis = xmat[:, 0].copy()
        # carAll_hinge.xml rotates the SolidWorks mesh so local +Y is world +Z.
        up_axis = xmat[:, 1].copy()
        lateral_axis = xmat[:, 2].copy()
        return forward_axis, lateral_axis, up_axis

    def _base_velocity_world(self):
        pos = self.data.xpos[self.chassis_id].copy()
        vel = (pos - self._last_xpos) / max(self.control_dt, 1e-9)
        return vel

    def _get_obs(self):
        _, _, up_axis = self._chassis_axes()
        vel_world = self._base_velocity_world()
        linear_speed = float(np.linalg.norm(vel_world[:2]))
        body_acc = (vel_world - self._last_vel_world) / max(self.control_dt, 1e-9)
        wheel_vel = self.data.qvel[self.wheel_dof_adrs].astype(np.float64)
        wheel_acc = (wheel_vel - self._last_wheel_vel) / max(self.control_dt, 1e-9)
        wheel_vel_obs = wheel_vel / max(self.max_wheel_speed, 1e-9)
        wheel_acc_obs = wheel_acc / max(self.max_wheel_speed / max(self.control_dt, 1e-9), 1e-9)
        lateral_pos = float(self.data.xpos[self.chassis_id][1])
        return np.array(
            [
                self.target_speed - linear_speed,
                linear_speed,
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
                lateral_pos,
                self._last_action[0],
                self._last_action[1],
                self._last_action[2],
                self._last_action[3],
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
        previous_action = self._last_action.copy()
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

        linear_speed = float(obs[1])
        vel_world = self._base_velocity_world()
        forward_axis, lateral_axis, _ = self._chassis_axes()
        forward_speed = float(np.dot(vel_world, forward_axis))
        lateral_speed = float(np.dot(vel_world, lateral_axis))
        vertical_speed = float(vel_world[2])
        up_z = float(obs[16])
        lateral_pos = float(obs[17])
        speed_error = self.target_speed - linear_speed
        action_change = self._last_action - previous_action

        speed_reward = np.exp(-80.0 * speed_error * speed_error)
        forward_alignment = max(forward_speed / max(linear_speed, 1e-6), 0.0)
        reward = (
            3.0 * speed_reward * forward_alignment
            - 8.0 * speed_error * speed_error
            - 2.0 * max(-forward_speed, 0.0)
            - 1.2 * abs(lateral_speed)
            - 0.3 * abs(vertical_speed)
            - 2.0 * lateral_pos * lateral_pos
            - 0.05 * float(np.sum(np.square(self._last_action)))
            - 0.03 * float(np.sum(np.square(action_change)))
        )

        chassis_height = float(self.data.xpos[self.chassis_id][2])
        too_tilted = up_z < 0.45
        too_low = chassis_height < 0.005
        drifted = abs(lateral_pos) > self.max_lateral_drift
        bad_speed = abs(speed_error) > 2.0
        terminated = False
        truncated = self._steps >= self.max_steps

        info = {
            "target_speed": self.target_speed,
            "linear_speed": linear_speed,
            "forward_speed": forward_speed,
            "speed_error": speed_error,
            "forward_alignment": forward_alignment,
            "lateral_speed": lateral_speed,
            "vertical_speed": vertical_speed,
            "lateral_pos": lateral_pos,
            "chassis_height": chassis_height,
            "up_z": up_z,
            "target_wheel_vel_fl": float(target_wheel_vel[0]),
            "target_wheel_vel_fr": float(target_wheel_vel[1]),
            "target_wheel_vel_rl": float(target_wheel_vel[2]),
            "target_wheel_vel_rr": float(target_wheel_vel[3]),
            "wheel_torque_fl": float(torque[0]),
            "wheel_torque_fr": float(torque[1]),
            "wheel_torque_rl": float(torque[2]),
            "wheel_torque_rr": float(torque[3]),
            "too_tilted": bool(too_tilted),
            "too_low": bool(too_low),
            "drifted": bool(drifted),
            "bad_speed": bool(bad_speed),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass
