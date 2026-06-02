"""
14_car_all_hinge_rl_train.py
============================
Train an RL controller for carAll_hinge.xml.

Action space:
  [front-left, front-right, rear-left, rear-right] wheel angular velocity commands,
  normalized to [-1, 1].

Usage:
  ./venv/bin/python examples/14_car_all_hinge_rl_train.py
  ./venv/bin/python examples/14_car_all_hinge_rl_train.py --episodes 800
  ./venv/bin/python examples/14_car_all_hinge_rl_train.py --timesteps 200000
  ./venv/bin/python examples/14_car_all_hinge_rl_train.py --target-speed 0.4
  ./venv/bin/python examples/14_car_all_hinge_rl_train.py --resume
"""

import argparse
import glob
import os
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from car_all_hinge_env import CarAllHingeEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "car_all_hinge")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "car_all_hinge")
MAX_EPISODE_STEPS = 250
TARGET_SPEED = 0.3
MAX_TORQUE = 1
MAX_WHEEL_SPEED = 100.0
TOTAL_EPISODES = 10000
RESUME_TRAINING = False
DELETE_OLD_NETWORKS = True
CHECKPOINT_SAVE_FREQ = 10_000


def make_env(target_speed=TARGET_SPEED):
    return CarAllHingeEnv(
        target_speed=target_speed,
        max_steps=MAX_EPISODE_STEPS,
        max_torque=MAX_TORQUE,
        max_wheel_speed=MAX_WHEEL_SPEED,
    )


def delete_old_outputs():
    for path in (MODEL_DIR, LOG_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"[CarAllRL] Deleted old output: {path}")


def get_checkpoint_norm_path(model_path):
    if model_path.endswith("car_all_latest.zip"):
        return os.path.join(MODEL_DIR, "vec_normalize.pkl")
    return model_path[:-4] + "_vecnormalize.pkl"


def find_latest_resume_model():
    candidates = glob.glob(os.path.join(MODEL_DIR, "car_all_latest.zip"))
    candidates += glob.glob(os.path.join(MODEL_DIR, "car_all_ppo_*_steps.zip"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_model(env):
    return PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=8,
        gamma=0.97,
        gae_lambda=0.92,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda",
    )


def train(timesteps, target_speed, resume, delete_old_networks):
    if resume:
        print("[CarAllRL] Resume is enabled; old network deletion is skipped.")
    elif delete_old_networks:
        delete_old_outputs()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(lambda: make_env(target_speed), n_envs=32, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(lambda: make_env(target_speed), n_envs=1, seed=7)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    latest_model = os.path.join(MODEL_DIR, "car_all_latest")
    latest_norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    resume_model = find_latest_resume_model() if resume else None
    if resume_model is not None:
        norm_path = get_checkpoint_norm_path(resume_model)
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            env.norm_reward = True
        print(f"[CarAllRL] Resuming from {resume_model}")
        model = PPO.load(resume_model, env=env, device="cuda")
    elif resume:
        env.close()
        eval_env.close()
        raise FileNotFoundError(
            f"Resume requested, but no saved model was found in {MODEL_DIR}. "
            "Run without --resume to start a new network."
        )
    else:
        model = build_model(env)

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
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="car_all_ppo",
        save_vecnormalize=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(latest_model)
    env.save(latest_norm_path)
    env.close()
    eval_env.close()
    print(f"[CarAllRL] Saved model to {latest_model}.zip")
    print(f"[CarAllRL] Saved VecNormalize to {latest_norm_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    parser.add_argument("--resume", action="store_true", default=RESUME_TRAINING)
    parser.add_argument("--delete-old-networks", action="store_true", default=DELETE_OLD_NETWORKS)
    args = parser.parse_args()

    timesteps = args.timesteps
    if timesteps is None:
        timesteps = args.episodes * MAX_EPISODE_STEPS
    print(
        f"[CarAllRL] episodes={args.episodes}, max_episode_steps={MAX_EPISODE_STEPS}, "
        f"target_speed={args.target_speed:.3f} m/s, max_torque={MAX_TORQUE:.4f} Nm, "
        f"max_wheel_speed={MAX_WHEEL_SPEED:.1f} rad/s, "
        f"total_timesteps={timesteps}"
    )
    print(
        f"[CarAllRL] resume={args.resume}, delete_old_networks={args.delete_old_networks}, "
        f"model_dir={MODEL_DIR}, log_dir={LOG_DIR}"
    )
    train(timesteps, args.target_speed, args.resume, args.delete_old_networks)


if __name__ == "__main__":
    main()
