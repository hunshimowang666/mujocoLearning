"""
06_box_rl_train.py
==================
Train an RL policy to hover simpleBox.xml around a target height.

Usage:
  ./venv/bin/python examples/06_box_rl_train.py
  ./venv/bin/python examples/06_box_rl_train.py --timesteps 200000
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from box_hover_env import SimpleBoxHoverEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "simple_box_hover")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "simple_box_hover")


def make_env():
    return SimpleBoxHoverEnv()


def train(timesteps, resume):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(make_env, n_envs=8, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1, seed=7)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    latest_model = os.path.join(MODEL_DIR, "box_hover_latest")
    if resume and os.path.exists(latest_model + ".zip"):
        print(f"[BoxHover] Resuming from {latest_model}.zip")
        model = PPO.load(latest_model, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=256,
            n_epochs=8,
            gamma=0.98,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            device="cpu",
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix="box_hover_ppo",
        verbose=1,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(latest_model)
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    env.close()
    eval_env.close()
    print(f"[BoxHover] Saved model to {latest_model}.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.timesteps, args.resume)


if __name__ == "__main__":
    main()
