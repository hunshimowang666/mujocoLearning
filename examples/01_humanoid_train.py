"""
01_humanoid_train.py
====================
人形机器人 (Humanoid-v5) 强化学习训练示例
使用 PPO 算法训练 MuJoCo Humanoid 直立行走

支持 Ctrl+C 优雅暂停 + --resume 断点续训

用法:
  python 01_humanoid_train.py                # 从头训练
  python 01_humanoid_train.py --resume       # 从最新检查点续训
  python 01_humanoid_train.py --eval         # 评估已训练模型
"""

import argparse
import os
import signal
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "humanoid")
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs",   "humanoid")
ENV_ID    = "Humanoid-v5"
TOTAL_TIMESTEPS = 10_000_000


class GracefulExitCallback(BaseCallback):
    """Ctrl+C 优雅退出：保存当前模型后停止训练。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._stop = False

    def notify(self):
        self._stop = True

    def _on_step(self) -> bool:
        return not self._stop


# 全局变量，让 signal handler 能访问回调
_exit_cb = None


def _signal_handler(sig, frame):
    """Ctrl+C 触发优雅退出。"""
    print("\n\n[Humanoid] Ctrl+C 收到，正在保存当前进度...")
    if _exit_cb is not None:
        _exit_cb.notify()
    else:
        print("[Humanoid] 未在训练中，直接退出。")
        raise KeyboardInterrupt


signal.signal(signal.SIGINT, _signal_handler)


def _find_latest_checkpoint(model_dir):
    """找到最新的 checkpoint 文件（步数最大的）。"""
    checkpoints = []
    for f in os.listdir(model_dir):
        if f.startswith("humanoid_ppo_") and f.endswith(".zip"):
            # 文件名格式: humanoid_ppo_XXXXX_steps.zip
            try:
                steps = int(f.replace("humanoid_ppo_", "").replace("_steps.zip", ""))
                checkpoints.append((steps, os.path.join(model_dir, f)))
            except ValueError:
                continue
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1], checkpoints[0][0]


def _get_remaining_timesteps(checkpoint_steps):
    """计算剩余需要训练的步数。"""
    return max(0, TOTAL_TIMESTEPS - checkpoint_steps)


def train(resume=False):
    global _exit_cb

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # 初始化环境和回调
    vec_env = make_vec_env(ENV_ID, n_envs=4, seed=42)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    exit_cb = GracefulExitCallback()
    _exit_cb = exit_cb

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="humanoid_ppo",
        verbose=1,
    )

    # 创建或加载模型
    if resume:
        ckpt_path, ckpt_steps = _find_latest_checkpoint(MODEL_DIR)
        if ckpt_path is None:
            print("[Humanoid] 未找到检查点，从头开始训练。")
            model = PPO(
                "MlpPolicy", vec_env,
                n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                learning_rate=3e-4, device="cpu", verbose=1,
            )
            remaining = TOTAL_TIMESTEPS
        else:
            print(f"[Humanoid] 从检查点续训: {ckpt_path}")
            print(f"  已完成步数: {ckpt_steps:,} / {TOTAL_TIMESTEPS:,}")
            model = PPO.load(ckpt_path, env=vec_env, device="cpu")
            remaining = _get_remaining_timesteps(ckpt_steps)
            if remaining == 0:
                print("[Humanoid] 训练已完成，无需续训。")
                vec_env.close()
                eval_env.close()
                return
            print(f"  剩余步数: {remaining:,}")
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            learning_rate=3e-4, device="cpu", verbose=1,
        )
        remaining = TOTAL_TIMESTEPS

    print(f"\n[Humanoid] 开始训练，环境: {ENV_ID}")
    print(f"  观测空间: {vec_env.observation_space}")
    print(f"  动作空间: {vec_env.action_space}")
    print(f"  目标步数: {remaining:,}")
    print(f"  Ctrl+C 可随时暂停并保存\n")

    start_time = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=[eval_cb, checkpoint_cb, exit_cb],
        progress_bar=True,
        reset_num_timesteps=not resume,
    )
    elapsed = time.time() - start_time

    # 保存最终模型和归一化参数
    model.save(os.path.join(MODEL_DIR, "humanoid_final"))
    vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

    print(f"\n[Humanoid] 训练结束，耗时 {elapsed:.0f}s")
    print(f"  模型已保存至: {MODEL_DIR}")
    if exit_cb._stop:
        print("  [提示] 可用 --resume 从最新检查点续训:")
        print("    python 01_humanoid_train.py --resume")

    vec_env.close()
    eval_env.close()
    _exit_cb = None


def evaluate():
    import numpy as np

    model_path = os.path.join(MODEL_DIR, "best_model")
    norm_path  = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    if not os.path.exists(model_path + ".zip"):
        print("[Humanoid] 未找到已训练模型，请先运行训练。")
        return

    env = make_vec_env(ENV_ID, n_envs=1)
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)
    print(f"[Humanoid] 已加载模型: {model_path}")

    obs = env.reset()
    total_rewards = []
    episode_reward = 0.0
    episode_count = 0
    max_episodes = 3

    while episode_count < max_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        if done[0]:
            episode_count += 1
            total_rewards.append(episode_reward)
            print(f"  Episode {episode_count}: reward = {episode_reward:.1f}")
            episode_reward = 0.0
            obs = env.reset()
        time.sleep(0.01)

    env.close()
    print(f"\n平均奖励: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanoid PPO Training")
    parser.add_argument("--eval", action="store_true", help="评估已训练的模型")
    parser.add_argument("--resume", action="store_true", help="从最新检查点续训")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train(resume=args.resume)
