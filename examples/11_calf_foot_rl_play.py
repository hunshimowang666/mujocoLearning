"""
11_calf_foot_rl_play.py
=======================
Play the trained RL torque controller for calf_foot_hinge.xml.

Usage:
  ./venv/bin/python examples/11_calf_foot_rl_play.py
"""

import argparse
import importlib
import os
import time

import mujoco
import numpy as np
from mujoco import viewer
from mujoco.glfw import glfw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from calf_foot_hinge_env import CalfFootHingeEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "calf_foot_hinge")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "best_model.zip")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "calf_foot_latest.zip")
DEFAULT_NORM = os.path.join(MODEL_DIR, "vec_normalize.pkl")

MAX_EPISODE_STEPS = importlib.import_module("10_calf_foot_rl_train").MAX_EPISODE_STEPS
IMPACT_DQ_DEG_PER_SEC = 286.0
DISTURBANCE_TORQUE = 0.7
DISTURBANCE_DURATION = 0.12


def make_policy_env(norm_path):
    env = DummyVecEnv([lambda: CalfFootHingeEnv(max_steps=MAX_EPISODE_STEPS)])
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


def play(model_path, norm_path):
    if not os.path.exists(model_path):
        if model_path == DEFAULT_MODEL and os.path.exists(FALLBACK_MODEL):
            model_path = FALLBACK_MODEL
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\nRun: ./venv/bin/python examples/10_calf_foot_rl_train.py"
            )

    policy_env = make_policy_env(norm_path)
    policy = PPO.load(model_path, env=policy_env, device="cpu")

    env = CalfFootHingeEnv(max_steps=MAX_EPISODE_STEPS)
    obs, _ = env.reset(seed=0)
    obs_v = normalize_obs(policy_env, obs)
    impact_dq = np.deg2rad(IMPACT_DQ_DEG_PER_SEC)
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded policy: {model_path}")
    print("Controls: A/Z disturbance, R reset, Q/Esc quit")

    with viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        wall_start = time.perf_counter()
        sim_start = env.data.time
        t_print = 0.0
        disturbance_until = 0.0
        disturbance_torque = 0.0

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed = env.data.time - sim_start
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_A:
                    env.data.qvel[env.dof_adr] -= impact_dq
                    disturbance_torque = -DISTURBANCE_TORQUE
                    disturbance_until = env.data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[A] disturbance: qvel={np.rad2deg(env.data.qvel[env.dof_adr]):+.1f} deg/s")
                elif key == glfw.KEY_Z:
                    env.data.qvel[env.dof_adr] += impact_dq
                    disturbance_torque = DISTURBANCE_TORQUE
                    disturbance_until = env.data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[Z] disturbance: qvel={np.rad2deg(env.data.qvel[env.dof_adr]):+.1f} deg/s")
                elif key == glfw.KEY_R:
                    obs, _ = env.reset(seed=0)
                    obs_v = normalize_obs(policy_env, obs)
                    disturbance_until = 0.0
                    disturbance_torque = 0.0
                    env.external_torque = 0.0
                    wall_start = time.perf_counter()
                    sim_start = env.data.time
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            action, _ = policy.predict(obs_v, deterministic=True)
            env.external_torque = disturbance_torque if env.data.time < disturbance_until else 0.0
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs_v = normalize_obs(policy_env, obs)

            if terminated or truncated:
                obs, _ = env.reset()
                obs_v = normalize_obs(policy_env, obs)
                disturbance_until = 0.0
                disturbance_torque = 0.0
                env.external_torque = 0.0
                wall_start = time.perf_counter()
                sim_start = env.data.time

            if env.data.time - t_print >= 0.5:
                print(
                    f"t={env.data.time:5.2f}s | q={info['q_deg']:+.2f} deg | "
                    f"dq={info['dq_deg_s']:+.1f} deg/s | tau={info['torque']:+.3f} Nm | "
                    f"reward={reward:+.2f}"
                )
                t_print = env.data.time

            v.sync()

    policy_env.close()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--norm", default=DEFAULT_NORM)
    args = parser.parse_args()
    play(args.model, args.norm)


if __name__ == "__main__":
    main()
