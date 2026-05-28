"""
02_ant_dog_train.py
===================
机械狗/四足机器人 (Ant-v5) 强化学习训练示例
使用 SAC (Soft Actor-Critic) 算法训练 Ant 四足行走

环境说明:
  - 27维观测空间 (8个关节角度+速度、躯干姿态、接触力)
  - 8维连续动作空间 (4条腿各2个关节力矩)
  - 奖励: 前进速度 + 存活 - 控制代价 - 接触代价

SAC 对连续控制效果好，样本效率高于 PPO

用法:
  python 02_ant_dog_train.py           # 训练
  python 02_ant_dog_train.py --eval    # 评估
  python 02_ant_dog_train.py --algo ppo  # 使用PPO替代SAC
"""

import argparse
import os
import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "ant_dog")
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs",   "ant_dog")
ENV_ID    = "Ant-v5"

ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}


def train(algo_name="sac"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    AlgoCls = ALGO_MAP[algo_name.lower()]

    # SAC/TD3 使用单环境 (off-policy); PPO 使用并行环境
    n_envs = 1 if algo_name in ("sac", "td3") else 4
    env = make_vec_env(ENV_ID, n_envs=n_envs, seed=42)

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=0)
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix=f"ant_{algo_name}",
    )

    if algo_name == "sac":
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            device="auto",
            verbose=1,
        )
    elif algo_name == "td3":
        model = TD3(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            gamma=0.99,
            device="auto",
            verbose=1,
        )
    else:  # ppo
        model = PPO(
            "MlpPolicy", env,
            n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99,
            gae_lambda=0.95, clip_range=0.2,
            learning_rate=3e-4,
            device="auto",
            verbose=1,
        )

    print(f"\n[Ant-Dog] 开始训练，环境: {ENV_ID}, 算法: {algo_name.upper()}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}\n")

    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, f"ant_{algo_name}_final"))
    print(f"\n[Ant-Dog] 训练完成，模型保存至 {MODEL_DIR}")


def evaluate(algo_name="sac"):
    import time, numpy as np
    AlgoCls = ALGO_MAP[algo_name.lower()]
    model_path = os.path.join(MODEL_DIR, "best_model")

    if not os.path.exists(model_path + ".zip"):
        print("[Ant-Dog] 未找到已训练模型，请先训练。")
        return

    env = gym.make(ENV_ID, render_mode="human")
    model = AlgoCls.load(model_path, env=env)
    print(f"[Ant-Dog] 加载模型: {model_path}")

    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            time.sleep(0.01)
        print(f"  Episode {ep+1}: reward = {ep_reward:.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",  action="store_true")
    parser.add_argument("--algo",  default="sac", choices=["sac", "ppo", "td3"])
    args = parser.parse_args()

    if args.eval:
        evaluate(args.algo)
    else:
        train(args.algo)