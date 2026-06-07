from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_XML = THIS_DIR / "inverted_pendulum_model.xml"


def wrap_to_pi(angle: float) -> float:
  return (angle + np.pi) % (2.0 * np.pi) - np.pi


class InvertedPendulumEnv(gym.Env):
  """MuJoCo cart-pole inverted pendulum.

  Action:
    1 value in [-1, 1], mapped to horizontal cart force in newtons.

  Observation:
    [cart_x, cart_v, sin(theta), cos(theta), theta_dot, last_action]
    theta = 0 means the pole is upright.
  """

  metadata = {"render_modes": []}

  def __init__(
    self,
    xml_path: str | Path = DEFAULT_XML,
    frame_skip: int = 10,
    max_force: float = 20.0,
    episode_time: float = 10.0,
    fail_angle_deg: float = 55.0,
    fail_cart_x: float = 1.15,
    random_start: bool = True,
  ):
    super().__init__()
    self.xml_path = Path(xml_path)
    self.frame_skip = frame_skip
    self.max_force = max_force
    self.episode_time = episode_time
    self.fail_angle = np.deg2rad(fail_angle_deg)
    self.fail_cart_x = fail_cart_x
    self.random_start = random_start

    self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
    self.data = mujoco.MjData(self.model)

    self.slider_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
    self.hinge_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    self.slider_qpos = self.model.jnt_qposadr[self.slider_jid]
    self.hinge_qpos = self.model.jnt_qposadr[self.hinge_jid]
    self.slider_qvel = self.model.jnt_dofadr[self.slider_jid]
    self.hinge_qvel = self.model.jnt_dofadr[self.hinge_jid]

    self._rng = np.random.default_rng()
    self._last_action = np.zeros(1, dtype=np.float32)

    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(
      low=np.array([-np.inf, -np.inf, -1.0, -1.0, -np.inf, -1.0], dtype=np.float32),
      high=np.array([np.inf, np.inf, 1.0, 1.0, np.inf, 1.0], dtype=np.float32),
      dtype=np.float32,
    )

  @property
  def control_dt(self) -> float:
    return self.model.opt.timestep * self.frame_skip

  def _theta(self) -> float:
    return wrap_to_pi(float(self.data.qpos[self.hinge_qpos]))

  def _get_obs(self) -> np.ndarray:
    theta = self._theta()
    obs = np.array(
      [
        self.data.qpos[self.slider_qpos],
        self.data.qvel[self.slider_qvel],
        np.sin(theta),
        np.cos(theta),
        self.data.qvel[self.hinge_qvel],
        self._last_action[0],
      ],
      dtype=np.float64,
    )
    return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

  def reset(self, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    if seed is not None:
      self._rng = np.random.default_rng(seed)

    mujoco.mj_resetData(self.model, self.data)
    self._last_action[:] = 0.0

    if self.random_start:
      self.data.qpos[self.slider_qpos] = self._rng.uniform(-0.08, 0.08)
      self.data.qvel[self.slider_qvel] = self._rng.uniform(-0.05, 0.05)
      self.data.qpos[self.hinge_qpos] = self._rng.uniform(-0.12, 0.12)
      self.data.qvel[self.hinge_qvel] = self._rng.uniform(-0.10, 0.10)

    mujoco.mj_forward(self.model, self.data)
    return self._get_obs(), {}

  def step(self, action):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    self._last_action = action.copy()
    self.data.ctrl[0] = float(action[0]) * self.max_force

    for _ in range(self.frame_skip):
      mujoco.mj_step(self.model, self.data)

    theta = self._theta()
    cart_x = float(self.data.qpos[self.slider_qpos])
    cart_v = float(self.data.qvel[self.slider_qvel])
    theta_dot = float(self.data.qvel[self.hinge_qvel])

    upright_reward = np.exp(-8.0 * theta * theta)
    center_reward = np.exp(-2.0 * cart_x * cart_x)
    velocity_cost = 0.01 * (cart_v * cart_v + 0.15 * theta_dot * theta_dot)
    action_cost = 0.001 * float(action[0] * action[0])
    reward = 2.0 * upright_reward + 0.5 * center_reward - velocity_cost - action_cost

    terminated = bool(abs(theta) > self.fail_angle or abs(cart_x) > self.fail_cart_x)
    truncated = bool(self.data.time >= self.episode_time)
    info = {
      "time": float(self.data.time),
      "cart_x": cart_x,
      "theta_deg": float(np.rad2deg(theta)),
      "force_n": float(self.data.ctrl[0]),
    }
    return self._get_obs(), float(reward), terminated, truncated, info

  def close(self):
    pass
