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
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np


class StableRewardWrapper(Wrapper):
    """包装器：修改奖励函数，鼓励平稳运动（单环境版本）"""
    
    def __init__(self, env):
        super().__init__(env)
        self.torso_body_id = -1
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.torso_body_id >= 0:
            try:
                torso_quat = self.env.unwrapped.data.body(self.torso_body_id).xquat
                w, x, y, z = torso_quat
                
                pitch = np.arcsin(2 * (w * y - x * z)) * (180 / np.pi)
                roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)) * (180 / np.pi)
                
                pitch_penalty = abs(pitch) * 0.05
                roll_penalty = abs(roll) * 0.05
                reward = reward - pitch_penalty - roll_penalty
                
                if abs(pitch) > 60 or abs(roll) > 60:
                    terminated = True
                    
            except:
                pass
                
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.torso_body_id == -1:
            for i in range(self.env.unwrapped.model.nbody):
                name = self.env.unwrapped.model.body(i).name
                if name == "torso":
                    self.torso_body_id = i
                    break
                    
            if self.torso_body_id == -1 and self.env.unwrapped.model.nbody > 1:
                self.torso_body_id = 1
                
        return obs, info


class VecStableRewardWrapper(VecEnvWrapper):
    """包装器：修改奖励函数，鼓励平稳运动（向量化环境版本）"""
    
    def __init__(self, venv):
        super().__init__(venv)
        self.torso_body_ids = []
        self.debug_count = 0
        
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        
        # 确保 done 是 numpy array
        import numpy as np
        if not isinstance(done, np.ndarray):
            done = np.array([done])
        
        # 遍历每个环境
        for env_idx in range(self.num_envs):
            if env_idx >= len(self.torso_body_ids):
                self.torso_body_ids.append(-1)
                
            body_id = self.torso_body_ids[env_idx]
            if body_id >= 0:
                try:
                    env = self.venv.envs[env_idx]
                    torso_quat = env.unwrapped.data.body(body_id).xquat
                    w, x, y, z = torso_quat
                    
                    # 计算俯仰角和横滚角（度）
                    pitch = np.arcsin(2 * (w * y - x * z)) * (180 / np.pi)
                    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)) * (180 / np.pi)
                    
                    # 姿态惩罚
                    pitch_penalty = abs(pitch) * 0.05
                    roll_penalty = abs(roll) * 0.05
                    reward[env_idx] = float(reward[env_idx]) - pitch_penalty - roll_penalty
                    
                    # 过度倾斜时终止回合
                    if abs(pitch) > 45 or abs(roll) > 45:
                        done[env_idx] = True
                        if self.debug_count % 100 == 0:
                            print(f"[DEBUG] Terminated - Pitch: {pitch:.1f}°, Roll: {roll:.1f}°")
                        
                except Exception as e:
                    if self.debug_count % 100 == 0:
                        print(f"[DEBUG] Error: {e}")
                        
        self.debug_count += 1
        return obs, reward, done, info
        
    def reset(self):
        obs = self.venv.reset()
        
        # 初始化每个环境的body id
        for env_idx in range(self.num_envs):
            if env_idx >= len(self.torso_body_ids):
                self.torso_body_ids.append(-1)
                
            if self.torso_body_ids[env_idx] == -1:
                env = self.venv.envs[env_idx]
                for i in range(env.unwrapped.model.nbody):
                    name = env.unwrapped.model.body(i).name
                    if name == "torso":
                        self.torso_body_ids[env_idx] = i
                        break
                        
                if self.torso_body_ids[env_idx] == -1 and env.unwrapped.model.nbody > 1:
                    self.torso_body_ids[env_idx] = 1
                    
        return obs


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
    
    # 对需要平稳运动的环境应用奖励包装器
    if env_id in ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]:
        env = VecStableRewardWrapper(env)
        eval_env = VecStableRewardWrapper(eval_env)
        print(f"[{env_key}] Applied VecStableRewardWrapper for stable movement")

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
        device="auto",
        verbose=0,  # 降低打印频率
    )

    print(f"\n[Car/Cheetah] 开始训练，环境: {env_id}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}\n")

    model.learn(
        total_timesteps=100_000,
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