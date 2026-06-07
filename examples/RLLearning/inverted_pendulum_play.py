from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from inverted_pendulum_env import InvertedPendulumEnv


THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / "logs"
MODEL_DIR = LOG_DIR / "models"
BEST_MODEL = LOG_DIR / "best" / "best_model.zip"
FINAL_MODEL = MODEL_DIR / "ppo_inverted_pendulum_final.zip"
VECNORM_PATH = LOG_DIR / "vecnormalize.pkl"

USE_MODEL = "auto"  # "auto", "best", "final", or an explicit .zip path.
DEVICE = "cuda"
PRINT_INTERVAL = 0.25


def find_model() -> Path:
  if USE_MODEL not in ("auto", "best", "final"):
    path = Path(USE_MODEL)
    if not path.exists():
      raise FileNotFoundError(path)
    return path
  if USE_MODEL in ("auto", "best") and BEST_MODEL.exists():
    return BEST_MODEL
  if USE_MODEL in ("auto", "final") and FINAL_MODEL.exists():
    return FINAL_MODEL
  checkpoints = sorted(MODEL_DIR.glob("ppo_inverted_pendulum_*.zip"))
  if checkpoints:
    return checkpoints[-1]
  raise FileNotFoundError(f"No trained model found under {LOG_DIR}. Run inverted_pendulum_train.py first.")


def make_policy_env() -> VecNormalize:
  env = DummyVecEnv([lambda: InvertedPendulumEnv(random_start=False)])
  if VECNORM_PATH.exists():
    env = VecNormalize.load(VECNORM_PATH, env)
  else:
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
  env.training = False
  env.norm_reward = False
  return env


def main() -> None:
  model_path = find_model()
  policy_env = make_policy_env()
  policy = PPO.load(model_path, env=policy_env, device=DEVICE)

  sim_env = InvertedPendulumEnv(random_start=False)
  obs, _ = sim_env.reset()
  print(f"[INFO] Loaded model: {model_path}")
  print("[INFO] Press Esc in the viewer window to quit.")

  last_print = 0.0
  with mujoco.viewer.launch_passive(sim_env.model, sim_env.data) as viewer:
    while viewer.is_running():
      step_start = time.time()

      norm_obs = policy_env.normalize_obs(obs.reshape(1, -1))
      action, _ = policy.predict(norm_obs, deterministic=True)
      obs, reward, terminated, truncated, info = sim_env.step(action[0])

      if sim_env.data.time - last_print >= PRINT_INTERVAL:
        last_print = sim_env.data.time
        print(
          f"t={info['time']:5.2f}s | "
          f"x={info['cart_x']:+.3f} m | "
          f"theta={info['theta_deg']:+.2f} deg | "
          f"force={info['force_n']:+.2f} N | "
          f"reward={reward:+.2f}"
        )

      if terminated or truncated:
        obs, _ = sim_env.reset()

      viewer.sync()
      sleep_time = sim_env.control_dt - (time.time() - step_start)
      if sleep_time > 0.0:
        time.sleep(sleep_time)

  sim_env.close()
  policy_env.close()


if __name__ == "__main__":
  main()
