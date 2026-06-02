"""
15_car_all_hinge_rl_play.py
===========================
Play the trained RL controller for carAll_hinge.xml.

Usage:
  ./venv/bin/python examples/15_car_all_hinge_rl_play.py
  ./venv/bin/python examples/15_car_all_hinge_rl_play.py --model examples/models/car_all_hinge/car_all_latest.zip
"""

import argparse
import importlib
import os
import time

from mujoco import viewer
from mujoco.glfw import glfw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from car_all_hinge_env import CarAllHingeEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "car_all_hinge")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "best_model.zip")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "car_all_latest.zip")
DEFAULT_NORM = os.path.join(MODEL_DIR, "vec_normalize.pkl")

TRAIN_CFG = importlib.import_module("14_car_all_hinge_rl_train")
MAX_EPISODE_STEPS = TRAIN_CFG.MAX_EPISODE_STEPS
TARGET_SPEED = TRAIN_CFG.TARGET_SPEED
MAX_TORQUE = TRAIN_CFG.MAX_TORQUE
MAX_WHEEL_SPEED = TRAIN_CFG.MAX_WHEEL_SPEED


def make_policy_env(norm_path, target_speed):
    env = DummyVecEnv([
        lambda: CarAllHingeEnv(
            target_speed=target_speed,
            max_steps=MAX_EPISODE_STEPS,
            max_torque=MAX_TORQUE,
            max_wheel_speed=MAX_WHEEL_SPEED,
        )
    ])
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    return env


def normalize_obs(policy_env, obs):
    obs_v = obs.reshape(1, -1)
    if isinstance(policy_env, VecNormalize):
        obs_v = policy_env.normalize_obs(obs_v)
    return obs_v


def play(model_path, norm_path, target_speed):
    if not os.path.exists(model_path):
        if model_path == DEFAULT_MODEL and os.path.exists(FALLBACK_MODEL):
            model_path = FALLBACK_MODEL
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: ./venv/bin/python examples/14_car_all_hinge_rl_train.py"
            )

    policy_env = make_policy_env(norm_path, target_speed)
    policy = PPO.load(model_path, env=policy_env, device="cuda")

    env = CarAllHingeEnv(
        target_speed=target_speed,
        max_steps=MAX_EPISODE_STEPS,
        max_torque=MAX_TORQUE,
        max_wheel_speed=MAX_WHEEL_SPEED,
    )
    obs, _ = env.reset(seed=0)
    obs_v = normalize_obs(policy_env, obs)
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded policy: {model_path}")
    print(f"Target speed: {target_speed:.3f} m/s")
    print("Controls: R reset, Q/Esc quit")

    with viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        wall_start = time.perf_counter()
        sim_start = env.data.time
        t_print = 0.0
        info = {}
        reward = 0.0

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed = env.data.time - sim_start
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_R:
                    obs, _ = env.reset(seed=0)
                    obs_v = normalize_obs(policy_env, obs)
                    wall_start = time.perf_counter()
                    sim_start = env.data.time
                    t_print = 0.0
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            action, _ = policy.predict(obs_v, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs_v = normalize_obs(policy_env, obs)

            if terminated or truncated:
                if terminated:
                    reasons = []
                    for key, label in (
                        ("too_tilted", "tilted"),
                        ("too_low", "too low"),
                        ("drifted", "drifted"),
                        ("bad_speed", "speed unstable"),
                    ):
                        if info.get(key):
                            reasons.append(label)
                    print(f"[reset] terminated: {', '.join(reasons) or 'unknown'}")
                elif truncated:
                    print("[reset] episode complete")
                obs, _ = env.reset()
                obs_v = normalize_obs(policy_env, obs)
                wall_start = time.perf_counter()
                sim_start = env.data.time
                t_print = 0.0

            if env.data.time - t_print >= 0.5 and info:
                print(
                    f"t={env.data.time:5.2f}s | speed={info['linear_speed']:+.3f} m/s | "
                    f"forward={info['forward_speed']:+.3f} | "
                    f"target={info['target_speed']:+.3f} | err={info['speed_error']:+.3f} | "
                    f"align={info['forward_alignment']:+.2f} | y={info['lateral_pos']:+.3f} | "
                    f"up={info['up_z']:+.2f} | reward={reward:+.2f}"
                )
                t_print = env.data.time

            v.sync()

    policy_env.close()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--norm", default=DEFAULT_NORM)
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    args = parser.parse_args()
    play(args.model, args.norm, args.target_speed)


if __name__ == "__main__":
    main()
