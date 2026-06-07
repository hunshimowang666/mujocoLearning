from __future__ import annotations

import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from inverted_pendulum_env import InvertedPendulumEnv


THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / "logs"
MODEL_DIR = LOG_DIR / "models"
BEST_DIR = LOG_DIR / "best"
VECNORM_PATH = LOG_DIR / "vecnormalize.pkl"

TOTAL_TIMESTEPS = 300_000
NUM_ENVS = 8
DEVICE = "cuda"
DELETE_OLD_NETWORKS = True
RESUME_TRAINING = False


def make_env():
  return InvertedPendulumEnv(random_start=True)


def latest_checkpoint() -> Path | None:
  checkpoints = sorted(MODEL_DIR.glob("ppo_inverted_pendulum_*.zip"))
  return checkpoints[-1] if checkpoints else None


def main() -> None:
  if DELETE_OLD_NETWORKS and not RESUME_TRAINING and LOG_DIR.exists():
    print(f"[INFO] Removing old logs: {LOG_DIR}")
    shutil.rmtree(LOG_DIR)

  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  BEST_DIR.mkdir(parents=True, exist_ok=True)

  env = make_vec_env(make_env, n_envs=NUM_ENVS, vec_env_cls=DummyVecEnv)
  if RESUME_TRAINING and VECNORM_PATH.exists():
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = True
    env.norm_reward = True
  else:
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  eval_env = make_vec_env(lambda: InvertedPendulumEnv(random_start=False), n_envs=1, vec_env_cls=DummyVecEnv)
  if RESUME_TRAINING and VECNORM_PATH.exists():
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
  else:
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
  eval_env.training = False
  eval_env.norm_reward = False

  resume_model = latest_checkpoint() if RESUME_TRAINING else None
  if resume_model is not None:
    print(f"[INFO] Resuming from: {resume_model}")
    model = PPO.load(resume_model, env=env, device=DEVICE)
  else:
    model = PPO(
      "MlpPolicy",
      env,
      device=DEVICE,
      verbose=1,
      learning_rate=3.0e-4,
      n_steps=1024,
      batch_size=256,
      gamma=0.99,
      gae_lambda=0.95,
      ent_coef=0.002,
      clip_range=0.2,
      policy_kwargs={"net_arch": [64, 64]},
      tensorboard_log=str(LOG_DIR / "tb"),
    )

  checkpoint_cb = CheckpointCallback(
    save_freq=max(10_000 // NUM_ENVS, 1),
    save_path=str(MODEL_DIR),
    name_prefix="ppo_inverted_pendulum",
  )
  eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_DIR),
    log_path=str(LOG_DIR / "eval"),
    eval_freq=max(10_000 // NUM_ENVS, 1),
    deterministic=True,
    render=False,
  )

  model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, eval_cb])
  final_model = MODEL_DIR / "ppo_inverted_pendulum_final.zip"
  model.save(final_model)
  env.save(VECNORM_PATH)
  print(f"[INFO] Saved final model: {final_model}")
  print(f"[INFO] Saved VecNormalize: {VECNORM_PATH}")

  env.close()
  eval_env.close()


if __name__ == "__main__":
  main()
