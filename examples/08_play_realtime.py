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

# 将 GLFW 相关 warning 升级为 exception，
# 这样按 ESC 关闭窗口后 env.render() 会抛异常，被下方 try/except 捕获退出
warnings.filterwarnings("error", message=".*GLFW.*")
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "humanoid": ("models/humanoid", "Humanoid-v5", PPO),
    "ant":      ("models/ant",      "Ant-v5",      SAC),
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

    # Create a rendering env (human mode = popup window)
    env = gym.make(env_id, render_mode="human")

    obs_v = vec_env.reset()
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    episode = 0

    print(f"[{model_name}] Simulation started! Close the MuJoCo window to stop.")
    print(f"[{model_name}] Controls: Space=Pause, R=Reset, Esc=Quit")

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
