"""
16_car_all_hinge_sine_path_rl_train.py
======================================
Train an RL controller for sine-path tracking with carAll_hinge.xml.

Action space:
  [front-left, front-right, rear-left, rear-right] wheel angular velocity commands,
  normalized to [-1, 1].

Usage:
  ./venv/bin/python examples/16_car_all_hinge_sine_path_rl_train.py
  ./venv/bin/python examples/16_car_all_hinge_sine_path_rl_train.py --episodes 2000
  ./venv/bin/python examples/16_car_all_hinge_sine_path_rl_train.py --resume
"""

import argparse
import glob
import os
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from car_all_hinge_sine_path_env import CarAllHingeSinePathEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "car_all_hinge_sine_path")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "car_all_hinge_sine_path")
MAX_EPISODE_STEPS = 600
PATH_SPEED = 0.10
PATH_AMPLITUDE = 0.10
PATH_WAVELENGTH = 2.0
MAX_TORQUE = 0.25
MAX_WHEEL_SPEED = 40.0
TOTAL_EPISODES = 3000
RESUME_TRAINING = False
DELETE_OLD_NETWORKS = True
CHECKPOINT_SAVE_FREQ = 10_000
NUM_ENVS = 32


def make_env():
    return CarAllHingeSinePathEnv(
        path_speed=PATH_SPEED,
        path_amplitude=PATH_AMPLITUDE,
        path_wavelength=PATH_WAVELENGTH,
        max_steps=MAX_EPISODE_STEPS,
        max_torque=MAX_TORQUE,
        max_wheel_speed=MAX_WHEEL_SPEED,
    )


def delete_old_outputs():
    for path in (MODEL_DIR, LOG_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"[CarSinePathRL] Deleted old output: {path}")


def get_checkpoint_norm_path(model_path):
    if model_path.endswith("car_sine_path_latest.zip"):
        return os.path.join(MODEL_DIR, "vec_normalize.pkl")
    return model_path[:-4] + "_vecnormalize.pkl"


def find_latest_resume_model():
    candidates = glob.glob(os.path.join(MODEL_DIR, "car_sine_path_latest.zip"))
    candidates += glob.glob(os.path.join(MODEL_DIR, "car_sine_path_ppo_*_steps.zip"))
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
        gamma=0.98,
        gae_lambda=0.94,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        # device="cuda",
        device="cpu",
    )


def train(timesteps, resume, delete_old_networks):
    if resume:
        print("[CarSinePathRL] Resume is enabled; old network deletion is skipped.")
    elif delete_old_networks:
        delete_old_outputs()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(make_env, n_envs=NUM_ENVS, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1, seed=7)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    latest_model = os.path.join(MODEL_DIR, "car_sine_path_latest")
    latest_norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    resume_model = find_latest_resume_model() if resume else None
    if resume_model is not None:
        norm_path = get_checkpoint_norm_path(resume_model)
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            env.norm_reward = True
        print(f"[CarSinePathRL] Resuming from {resume_model}")
        # model = PPO.load(resume_model, env=env, device="cuda")
        model = PPO.load(resume_model, env=env, device="cpu")
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
        name_prefix="car_sine_path_ppo",
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
    print(f"[CarSinePathRL] Saved model to {latest_model}.zip")
    print(f"[CarSinePathRL] Saved VecNormalize to {latest_norm_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=RESUME_TRAINING)
    parser.add_argument("--delete-old-networks", action="store_true", default=DELETE_OLD_NETWORKS)
    args = parser.parse_args()

    timesteps = args.timesteps
    if timesteps is None:
        timesteps = args.episodes * MAX_EPISODE_STEPS
    print(
        f"[CarSinePathRL] episodes={args.episodes}, max_episode_steps={MAX_EPISODE_STEPS}, "
        f"path_speed={PATH_SPEED:.3f} m/s, amplitude={PATH_AMPLITUDE:.3f} m, "
        f"wavelength={PATH_WAVELENGTH:.3f} m, total_timesteps={timesteps}"
    )
    print(
        f"[CarSinePathRL] resume={args.resume}, delete_old_networks={args.delete_old_networks}, "
        f"model_dir={MODEL_DIR}, log_dir={LOG_DIR}"
    )
    train(timesteps, args.resume, args.delete_old_networks)


if __name__ == "__main__":
    main()
