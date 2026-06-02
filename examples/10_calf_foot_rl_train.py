"""
10_calf_foot_rl_train.py
========================
Train an RL torque controller for calf_foot_hinge.xml.

Usage:
  ./venv/bin/python examples/10_calf_foot_rl_train.py
  ./venv/bin/python examples/10_calf_foot_rl_train.py --episodes 1000
  ./venv/bin/python examples/10_calf_foot_rl_train.py --timesteps 300000
  ./venv/bin/python examples/10_calf_foot_rl_train.py --resume
"""

import argparse
import glob
import os
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from calf_foot_hinge_env import CalfFootHingeEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "calf_foot_hinge")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "calf_foot_hinge")
TARGET_ANGLE_TABLE = os.path.join(SCRIPT_DIR, "calf_foot_target_angles.csv")
MAX_EPISODE_STEPS = 500
MAX_TRACKING_ERROR_DEG = 10.0
TOTAL_EPISODES = 4000
RESUME_TRAINING = False
DELETE_OLD_NETWORKS = True
CHECKPOINT_SAVE_FREQ = 10_000


def make_env():
    return CalfFootHingeEnv(
        max_steps=MAX_EPISODE_STEPS,
        max_tracking_error_deg=MAX_TRACKING_ERROR_DEG,
        target_table_path=TARGET_ANGLE_TABLE,
    )


def delete_old_outputs():
    for path in (MODEL_DIR, LOG_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"[CalfFootRL] Deleted old output: {path}")


def get_checkpoint_norm_path(model_path):
    if model_path.endswith("calf_foot_latest.zip"):
        return os.path.join(MODEL_DIR, "vec_normalize.pkl")
    return model_path[:-4] + "_vecnormalize.pkl"


def find_latest_resume_model():
    candidates = glob.glob(os.path.join(MODEL_DIR, "calf_foot_latest.zip"))
    candidates += glob.glob(os.path.join(MODEL_DIR, "calf_foot_ppo_*_steps.zip"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def train(timesteps, resume, delete_old_networks):
    if delete_old_networks and resume:
        print("[CalfFootRL] Resume is enabled, so old network deletion is skipped.")
    elif delete_old_networks:
        delete_old_outputs()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(make_env, n_envs=8, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1, seed=7)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    latest_model = os.path.join(MODEL_DIR, "calf_foot_latest")
    latest_norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    resume_model = find_latest_resume_model() if resume else None
    if resume_model is not None:
        norm_path = get_checkpoint_norm_path(resume_model)
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            env.norm_reward = True
        print(f"[CalfFootRL] Resuming from {resume_model}")
        model = PPO.load(resume_model, env=env, device="cuda")
    elif resume:
        print("[CalfFootRL] Resume requested, but no checkpoint was found. Starting new training.")
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
            device="cuda",
        )
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
            device="cuda",
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
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="calf_foot_ppo",
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
    print(f"[CalfFootRL] Saved model to {latest_model}.zip")


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
        f"[CalfFootRL] episodes={args.episodes}, max_episode_steps={MAX_EPISODE_STEPS}, "
        f"max_tracking_error={MAX_TRACKING_ERROR_DEG} deg, total_timesteps={timesteps}"
    )
    print(
        f"[CalfFootRL] resume={args.resume}, delete_old_networks={args.delete_old_networks}, "
        f"model_dir={MODEL_DIR}, log_dir={LOG_DIR}"
    )
    print(f"[CalfFootRL] target_angle_table={TARGET_ANGLE_TABLE}")
    train(timesteps, args.resume, args.delete_old_networks)


if __name__ == "__main__":
    main()
