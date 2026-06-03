"""
18_go2_squat_rl_train.py
========================
Train an RL torque controller that makes Unitree Go2 squat.

Observation:
  12 joint position errors followed by 12 joint velocities.

Action:
  12 joint torques normalized to [-1, 1].

Usage:
  ./venv/bin/python examples/18_go2_squat_rl_train.py
  ./venv/bin/python examples/18_go2_squat_rl_train.py --episodes 3000
  ./venv/bin/python examples/18_go2_squat_rl_train.py --timesteps 1000000
  ./venv/bin/python examples/18_go2_squat_rl_train.py --resume
"""

import argparse
import glob
import os
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from go2_squat_env import Go2SquatEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "go2_squat")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "go2_squat")

# 直接在这里改参数，然后运行本文件即可。
TARGET_HIP_DEG = 0.0
TARGET_THIGH_DEG = 68.8
TARGET_CALF_DEG = -131.8
MAX_EPISODE_STEPS = 500
TOTAL_EPISODES = 6000
# This task uses Stable-Baselines3 with a small MLP policy. MuJoCo simulation is
# CPU-side here, so parallel CPU environments are usually faster than CUDA.
NUM_ENVS = 12
RESUME_TRAINING = False
DELETE_OLD_NETWORKS = True
CHECKPOINT_SAVE_FREQ = 20_000
DEVICE = "cpu"


class EvalCallbackWithVecNormalize(EvalCallback):
    def _on_step(self) -> bool:
        previous_best = self.best_mean_reward
        continue_training = super()._on_step()
        if self.best_mean_reward > previous_best and self.best_model_save_path is not None:
            norm_path = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
            self.training_env.save(norm_path)
            print(f"[Go2SquatRL] Saved best VecNormalize to {norm_path}")
        return continue_training


def make_env():
    return Go2SquatEnv(
        target_hip_deg=TARGET_HIP_DEG,
        target_thigh_deg=TARGET_THIGH_DEG,
        target_calf_deg=TARGET_CALF_DEG,
        max_steps=MAX_EPISODE_STEPS,
    )


def delete_old_outputs():
    for path in (MODEL_DIR, LOG_DIR):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"[Go2SquatRL] Deleted old output: {path}")


def get_checkpoint_norm_path(model_path):
    if model_path.endswith("go2_squat_latest.zip"):
        return os.path.join(MODEL_DIR, "vec_normalize.pkl")
    filename = os.path.basename(model_path)
    prefix = "go2_squat_ppo_"
    suffix = "_steps.zip"
    if filename.startswith(prefix) and filename.endswith(suffix):
        steps = filename[len(prefix) : -len(suffix)]
        return os.path.join(MODEL_DIR, f"go2_squat_ppo_vecnormalize_{steps}_steps.pkl")
    return model_path[:-4] + "_vecnormalize.pkl"


def find_latest_resume_model():
    candidates = glob.glob(os.path.join(MODEL_DIR, "go2_squat_latest.zip"))
    candidates += glob.glob(os.path.join(MODEL_DIR, "go2_squat_ppo_*_steps.zip"))
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
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        device=DEVICE,
    )


def train(timesteps, resume, delete_old_networks):
    if resume:
        print("[Go2SquatRL] Resume is enabled; old network deletion is skipped.")
    elif delete_old_networks:
        delete_old_outputs()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(make_env, n_envs=NUM_ENVS, seed=42, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env, n_envs=1, seed=7)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    latest_model = os.path.join(MODEL_DIR, "go2_squat_latest")
    latest_norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    resume_model = find_latest_resume_model() if resume else None
    if resume_model is not None:
        norm_path = get_checkpoint_norm_path(resume_model)
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            env.training = True
            env.norm_reward = True
        print(f"[Go2SquatRL] Resuming from {resume_model}")
        model = PPO.load(resume_model, env=env, device=DEVICE)
    elif resume:
        env.close()
        eval_env.close()
        raise FileNotFoundError(
            f"Resume requested, but no saved model was found in {MODEL_DIR}. "
            "Run without --resume to start a new network."
        )
    else:
        model = build_model(env)

    eval_cb = EvalCallbackWithVecNormalize(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="go2_squat_ppo",
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
    print(f"[Go2SquatRL] Saved model to {latest_model}.zip")
    print(f"[Go2SquatRL] Saved VecNormalize to {latest_norm_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=RESUME_TRAINING)
    parser.add_argument("--delete-old-networks", action="store_true", default=DELETE_OLD_NETWORKS)
    parser.add_argument("--keep-old-networks", action="store_true")
    args = parser.parse_args()

    delete_old_networks = args.delete_old_networks and not args.keep_old_networks
    timesteps = args.timesteps
    if timesteps is None:
        timesteps = args.episodes * MAX_EPISODE_STEPS

    print(
        f"[Go2SquatRL] episodes={args.episodes}, max_episode_steps={MAX_EPISODE_STEPS}, "
        f"num_envs={NUM_ENVS}, device={DEVICE}, total_timesteps={timesteps}"
    )
    print(
        f"[Go2SquatRL] target hip/thigh/calf = "
        f"{TARGET_HIP_DEG:.1f}, {TARGET_THIGH_DEG:.1f}, {TARGET_CALF_DEG:.1f} deg"
    )
    print(
        f"[Go2SquatRL] resume={args.resume}, delete_old_networks={delete_old_networks}, "
        f"model_dir={MODEL_DIR}, log_dir={LOG_DIR}"
    )
    train(timesteps, args.resume, delete_old_networks)


if __name__ == "__main__":
    main()
