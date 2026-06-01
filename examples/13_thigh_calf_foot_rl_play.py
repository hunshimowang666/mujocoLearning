"""
13_thigh_calf_foot_rl_play.py
=============================
Play the trained RL torque controller for thigh_calf_foot_hinge.xml.

Usage:
  ./venv/bin/python examples/13_thigh_calf_foot_rl_play.py
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

from thigh_calf_foot_hinge_env import ThighCalfFootHingeEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "thigh_calf_foot_hinge")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "best_model.zip")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "thigh_calf_foot_latest.zip")
DEFAULT_NORM = os.path.join(MODEL_DIR, "vec_normalize.pkl")

TRAIN_CFG = importlib.import_module("12_thigh_calf_foot_rl_train")
MAX_EPISODE_STEPS = TRAIN_CFG.MAX_EPISODE_STEPS
MAX_TRACKING_ERROR_DEG = TRAIN_CFG.MAX_TRACKING_ERROR_DEG
TARGET_ANGLE_TABLE = TRAIN_CFG.TARGET_ANGLE_TABLE
IMPACT_DQ_DEG_PER_SEC = 286.0
DISTURBANCE_TORQUE = 0.7
DISTURBANCE_DURATION = 0.12


def make_env():
    return ThighCalfFootHingeEnv(
        max_steps=MAX_EPISODE_STEPS,
        max_tracking_error_deg=MAX_TRACKING_ERROR_DEG,
        target_table_path=TARGET_ANGLE_TABLE,
    )


def make_policy_env(norm_path):
    env = DummyVecEnv([make_env])
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
                f"Model not found: {model_path}\nRun: ./venv/bin/python examples/12_thigh_calf_foot_rl_train.py"
            )

    policy_env = make_policy_env(norm_path)
    policy = PPO.load(model_path, env=policy_env, device="cpu")

    env = make_env()
    obs, _ = env.reset(seed=0)
    obs_v = normalize_obs(policy_env, obs)
    impact_dq = np.deg2rad(IMPACT_DQ_DEG_PER_SEC)
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded policy: {model_path}")
    print(f"Target angle table: {TARGET_ANGLE_TABLE}")
    print(f"Max tracking error before reset: {MAX_TRACKING_ERROR_DEG:.1f} deg")
    print("Controls: A/Z knee disturbance, S/X ankle disturbance, R reset, Q/Esc quit")

    with viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        wall_start = time.perf_counter()
        sim_start = env.data.time
        t_print = 0.0
        disturbance_until = 0.0
        disturbance_torque = np.zeros(2, dtype=np.float64)

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed = env.data.time - sim_start
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_A:
                    env.data.qvel[env.dof_adrs[0]] -= impact_dq
                    disturbance_torque[:] = [-DISTURBANCE_TORQUE, 0.0]
                    disturbance_until = env.data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[A] knee disturbance: dq={np.rad2deg(env.data.qvel[env.dof_adrs[0]]):+.1f} deg/s")
                elif key == glfw.KEY_Z:
                    env.data.qvel[env.dof_adrs[0]] += impact_dq
                    disturbance_torque[:] = [DISTURBANCE_TORQUE, 0.0]
                    disturbance_until = env.data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[Z] knee disturbance: dq={np.rad2deg(env.data.qvel[env.dof_adrs[0]]):+.1f} deg/s")
                elif key == glfw.KEY_S:
                    env.data.qvel[env.dof_adrs[1]] -= impact_dq
                    disturbance_torque[:] = [0.0, -DISTURBANCE_TORQUE]
                    disturbance_until = env.data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[S] ankle disturbance: dq={np.rad2deg(env.data.qvel[env.dof_adrs[1]]):+.1f} deg/s")
                elif key == glfw.KEY_X:
                    env.data.qvel[env.dof_adrs[1]] += impact_dq
                    disturbance_torque[:] = [0.0, DISTURBANCE_TORQUE]
                    disturbance_until = env.data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[X] ankle disturbance: dq={np.rad2deg(env.data.qvel[env.dof_adrs[1]]):+.1f} deg/s")
                elif key == glfw.KEY_R:
                    obs, _ = env.reset(seed=0)
                    obs_v = normalize_obs(policy_env, obs)
                    disturbance_until = 0.0
                    disturbance_torque[:] = 0.0
                    env.external_torque[:] = 0.0
                    wall_start = time.perf_counter()
                    sim_start = env.data.time
                    t_print = 0.0
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            action, _ = policy.predict(obs_v, deterministic=True)
            env.external_torque[:] = disturbance_torque if env.data.time < disturbance_until else 0.0
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs_v = normalize_obs(policy_env, obs)

            if terminated or truncated:
                if terminated:
                    reasons = []
                    if info.get("error_too_large"):
                        reasons.append("tracking error too large")
                    if info.get("foot_too_low"):
                        reasons.append("foot too low")
                    print(f"[reset] terminated: {', '.join(reasons) or 'unknown'}")
                elif truncated:
                    print("[reset] episode complete")
                obs, _ = env.reset()
                obs_v = normalize_obs(policy_env, obs)
                disturbance_until = 0.0
                disturbance_torque[:] = 0.0
                env.external_torque[:] = 0.0
                wall_start = time.perf_counter()
                sim_start = env.data.time
                t_print = 0.0

            if env.data.time - t_print >= 0.5:
                print(
                    f"t={env.data.time:5.2f}s | "
                    f"knee q={info['knee_q_deg']:+.2f}, target={info['knee_target_deg']:+.2f}, "
                    f"err={info['knee_error_deg']:+.2f}, tau={info['knee_torque']:+.3f} | "
                    f"ankle q={info['ankle_q_deg']:+.2f}, target={info['ankle_target_deg']:+.2f}, "
                    f"err={info['ankle_error_deg']:+.2f}, tau={info['ankle_torque']:+.3f} | "
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
