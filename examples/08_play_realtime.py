"""
08_play_realtime.py
===================
打开 MuJoCo 可视化窗口，用训练好的模型实时控制机器人。

用法:
  python 08_play_realtime.py --model humanoid
  python 08_play_realtime.py --model ant
  python 08_play_realtime.py --model cheetah

快捷键（在 MuJoCo 窗口中）:
  Space  - 暂停/继续
  R      - 重置
  Esc    - 退出
"""

import argparse
import os
import numpy as np
import warnings

import gymnasium as gym
import mujoco
from gymnasium import Wrapper

# 将 GLFW 相关 warning 升级为 exception，
# 这样按 ESC 关闭窗口后 env.render() 会抛异常，被下方 try/except 捕获退出
warnings.filterwarnings("error", message=".*GLFW.*")
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


class HalfCheetahRewardWrapper(Wrapper):
    """包装器：修改HalfCheetah奖励函数，鼓励平稳前进"""
    
    def __init__(self, env):
        super().__init__(env)
        self.torso_body_id = -1
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 获取躯干位置和姿态
        if self.torso_body_id >= 0:
            try:
                # 获取躯干的旋转角度（quaternion）
                torso_quat = self.env.unwrapped.data.body(self.torso_body_id).xquat
                
                # 从四元数计算俯仰角（pitch）和横滚角（roll）
                w, x, y, z = torso_quat
                pitch = np.arcsin(2 * (w * y - x * z)) * (180 / np.pi)
                roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)) * (180 / np.pi)
                
                # 计算姿态惩罚（角度越大惩罚越大）
                pitch_penalty = abs(pitch) * 0.1
                roll_penalty = abs(roll) * 0.1
                
                # 修改奖励：减去姿态惩罚
                reward = reward - pitch_penalty - roll_penalty
                
                # 如果身体过度倾斜，终止回合
                if abs(pitch) > 60 or abs(roll) > 60:
                    terminated = True
                    print(f"[Terminated] Pitch: {pitch:.1f}°, Roll: {roll:.1f}°")
                    
            except Exception as e:
                pass
                
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 查找躯干body
        if self.torso_body_id == -1:
            for i in range(self.env.unwrapped.model.nbody):
                name = self.env.unwrapped.model.body(i).name
                if name == "torso":
                    self.torso_body_id = i
                    break
                    
            if self.torso_body_id == -1 and self.env.unwrapped.model.nbody > 1:
                self.torso_body_id = 1
                
        return obs, info


class CameraFollowWrapper(Wrapper):
    """包装器：让相机跟随机器人运动"""
    
    def __init__(self, env, follow_body_name="torso"):
        super().__init__(env)
        self.follow_body_name = follow_body_name
        self.follow_body_id = -1
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 获取viewer和相机
        renderer = self.env.unwrapped.mujoco_renderer
        if renderer and renderer.viewer:
            cam = renderer.viewer.cam
            
            # 获取机器人躯干位置
            if self.follow_body_id >= 0:
                try:
                    body_pos = self.env.unwrapped.data.body(self.follow_body_id).xpos
                    
                    # 设置相机看向机器人
                    cam.lookat[0] = body_pos[0]  # X位置
                    cam.lookat[1] = body_pos[1]  # Y位置
                    cam.lookat[2] = body_pos[2] + 0.5  # Z位置（胸部高度）
                    
                except:
                    pass
            
            # 固定相机角度和距离
            cam.elevation = -30.0  # 仰角
            cam.distance = 8.0      # 距离
            cam.azimuth = 90.0      # 固定方位角（从侧面看）
                
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 查找要跟随的body
        if self.follow_body_id == -1:
            for i in range(self.env.unwrapped.model.nbody):
                name = self.env.unwrapped.model.body(i).name
                if name == self.follow_body_name:
                    self.follow_body_id = i
                    print(f"[Camera] Following body: {name}")
                    break
                    
            # 如果没找到，使用第一个非world body
            if self.follow_body_id == -1 and self.env.unwrapped.model.nbody > 1:
                self.follow_body_id = 1
                print(f"[Camera] Following body[{self.follow_body_id}]: {self.env.unwrapped.model.body(self.follow_body_id).name}")
                
        return obs, info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "humanoid": ("models/humanoid", "Humanoid-v5", PPO),
    "ant":      ("models/ant",      "Ant-v5",      SAC),
    "ant_dog":  ("models/ant_dog",  "Ant-v5",      SAC),
    "cheetah":  ("models/cheetah",  "HalfCheetah-v5", TD3),
}


def play(model_name):
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
        return

    model_dir, env_id, algo_class = MODEL_CONFIGS[model_name]
    model_path = os.path.join(SCRIPT_DIR, model_dir, "best_model")
    norm_path  = os.path.join(SCRIPT_DIR, model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found: {model_path}.zip")
        print("Please train first!")
        return

    # Load model with normalization
    vec_env = make_vec_env(env_id, n_envs=1, seed=42)
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = algo_class.load(model_path, env=vec_env)
    print(f"[{model_name}] Model loaded: {model_path}")

    # Create a rendering env
    env = gym.make(env_id, render_mode="human")
    
    # For HalfCheetah, apply reward wrapper to encourage stable running
    if env_id == "HalfCheetah-v5":
        env = HalfCheetahRewardWrapper(env)
        print(f"[{model_name}] Applied HalfCheetahRewardWrapper for stable running")
    
    # Wrap with camera follow
    env = CameraFollowWrapper(env, follow_body_name="torso")

    obs_v = vec_env.reset()
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    episode = 0

    print(f"[{model_name}] Simulation started! Close the MuJoCo window to stop.")
    print(f"[{model_name}] Controls: Space=Pause, R=Reset, Esc=Quit")
    print(f"[{model_name}] Camera: Following robot automatically")

    try:
        while True:
            # Try to render - this will fail when window is closed
            try:
                env.render()
            except Exception as e:
                print(f"\nWindow closed, stopping...")
                break

            action, _ = model.predict(obs_v, deterministic=True)

            # Step both envs in sync
            obs_v, reward_v, done_v, info_v = vec_env.step(action)
            obs, reward, terminated, truncated, info = env.step(action[0])

            total_reward += reward
            steps += 1

            if terminated or truncated:
                episode += 1
                print(f"  Episode {episode}: reward={total_reward:.1f}, steps={steps}")
                total_reward = 0.0
                steps = 0
                obs_v = vec_env.reset()
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trained MuJoCo model in real-time")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which trained model to play")
    args = parser.parse_args()
    play(args.model)