"""
03_car_cheetah_train.py
=======================
高速小车/猎豹机器人 (HalfCheetah-v5) 强化学习训练示例
使用 TD3 算法训练猎豹机器人高速奔跑

环境说明:
  - HalfCheetah: 17维观测, 6维动作 (2条腿各3关节)
  - 奖励: 前进速度 - 控制代价 (无存活限制)
  - 最高奖励可达 10000+

同时提供 Car Racing (连续控制版本) 选项

用法:
  python 03_car_cheetah_train.py              # 训练 HalfCheetah (TD3)
  python 03_car_cheetah_train.py --env hopper # 训练 Hopper 单脚跳
  python 03_car_cheetah_train.py --eval       # 评估
"""

import argparse
import os
import gymnasium as gym
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# 支持的环境列表
ENVS = {
    "cheetah": "HalfCheetah-v5",  # 猎豹/高速小车
    "hopper":  "Hopper-v5",        # 单脚跳机器人
    "walker":  "Walker2d-v5",      # 双脚行走
    "pusher":  "Pusher-v5",        # 机械臂推物
}


def get_paths(env_key):
    model_dir = os.path.join(os.path.dirname(__file__), "models", env_key)
    log_dir   = os.path.join(os.path.dirname(__file__), "logs",   env_key)
    return model_dir, log_dir


def train(env_key="cheetah"):
    env_id = ENVS.get(env_key, ENVS["cheetah"])
    model_dir, log_dir = get_paths(env_key)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    env = make_vec_env(env_id, n_envs=1, seed=42)
    eval_env = make_vec_env(env_id, n_envs=1, seed=0)

    # TD3 探索噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix=f"{env_key}_td3",
    )

    model = TD3(
        "MlpPolicy", env,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        device="cpu",
        verbose=1,
    )

    print(f"\n[Car/Cheetah] 开始训练，环境: {env_id}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}\n")

    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, f"{env_key}_td3_final"))
    print(f"\n[Car/Cheetah] 训练完成，模型保存至 {model_dir}")


def evaluate(env_key="cheetah"):
    import time
    env_id = ENVS.get(env_key, ENVS["cheetah"])
    model_dir, _ = get_paths(env_key)
    model_path = os.path.join(model_dir, "best_model")

    if not os.path.exists(model_path + ".zip"):
        print(f"[{env_key}] 未找到模型，请先训练。")
        return

    env = gym.make(env_id, render_mode="human")
    model = TD3.load(model_path, env=env)
    print(f"[{env_key}] 加载模型: {model_path}")

    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            time.sleep(0.005)
        print(f"  Episode {ep+1}: reward = {ep_reward:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",  default="cheetah", choices=list(ENVS.keys()))
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.env)
    else:
        train(args.env)
