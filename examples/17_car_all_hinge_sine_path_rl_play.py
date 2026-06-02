"""
17_car_all_hinge_sine_path_rl_play.py
=====================================
Play the trained sine-path tracking controller for carAll_hinge.xml.

Usage:
  ./venv/bin/python examples/17_car_all_hinge_sine_path_rl_play.py
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

from car_all_hinge_sine_path_env import CarAllHingeSinePathEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "car_all_hinge_sine_path")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "best_model.zip")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "car_sine_path_latest.zip")
DEFAULT_NORM = os.path.join(MODEL_DIR, "vec_normalize.pkl")

TRAIN_CFG = importlib.import_module("16_car_all_hinge_sine_path_rl_train")


def make_env():
    return CarAllHingeSinePathEnv(
        path_speed=TRAIN_CFG.PATH_SPEED,
        path_amplitude=TRAIN_CFG.PATH_AMPLITUDE,
        path_wavelength=TRAIN_CFG.PATH_WAVELENGTH,
        max_steps=TRAIN_CFG.MAX_EPISODE_STEPS,
        max_torque=TRAIN_CFG.MAX_TORQUE,
        max_wheel_speed=TRAIN_CFG.MAX_WHEEL_SPEED,
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


def sine_path_point(x):
    k = 2.0 * np.pi / TRAIN_CFG.PATH_WAVELENGTH
    y = TRAIN_CFG.PATH_AMPLITUDE * np.sin(k * x)
    return float(y)


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_sine_path(scene, target_x=None, target_y=None):
    scene.ngeom = 0
    control_dt = 0.002 * 10
    x_end = TRAIN_CFG.PATH_SPEED * TRAIN_CFG.MAX_EPISODE_STEPS * control_dt
    x_end = max(x_end, TRAIN_CFG.PATH_WAVELENGTH)
    xs = np.linspace(0.0, x_end, 140)
    for x in xs:
        add_sphere(
            scene,
            (x, sine_path_point(x), 0.006),
            0.008,
            (0.1, 0.75, 0.35, 0.55),
        )
    if target_x is not None and target_y is not None:
        add_sphere(
            scene,
            (target_x, target_y, 0.025),
            0.018,
            (1.0, 0.15, 0.05, 0.9),
        )


def play(model_path, norm_path):
    if not os.path.exists(model_path):
        if model_path == DEFAULT_MODEL and os.path.exists(FALLBACK_MODEL):
            model_path = FALLBACK_MODEL
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: ./venv/bin/python examples/16_car_all_hinge_sine_path_rl_train.py"
            )

    policy_env = make_policy_env(norm_path)
    policy = PPO.load(model_path, env=policy_env, device="cuda")

    env = make_env()
    obs, _ = env.reset(seed=0)
    obs_v = normalize_obs(policy_env, obs)
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded policy: {model_path}")
    print(
        f"Sine path: speed={TRAIN_CFG.PATH_SPEED:.3f} m/s, "
        f"amplitude={TRAIN_CFG.PATH_AMPLITUDE:.3f} m, "
        f"wavelength={TRAIN_CFG.PATH_WAVELENGTH:.3f} m"
    )
    print("Controls: R reset, Q/Esc quit")

    with viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        wall_start = time.perf_counter()
        sim_start = env.data.time
        t_print = 0.0
        info = {}
        reward = 0.0
        draw_sine_path(v.user_scn)

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
                print("[reset] episode complete")
                obs, _ = env.reset()
                obs_v = normalize_obs(policy_env, obs)
                wall_start = time.perf_counter()
                sim_start = env.data.time
                t_print = 0.0

            if env.data.time - t_print >= 0.5 and info:
                print(
                    f"t={env.data.time:5.2f}s | path_err={info['path_error']:.3f} m | "
                    f"head_err={info['heading_error']:+.3f} rad | "
                    f"pos=({info['x']:+.3f},{info['y']:+.3f}) | "
                    f"target=({info['target_x']:+.3f},{info['target_y']:+.3f}) | "
                    f"reward={reward:+.2f}"
                )
                t_print = env.data.time

            if info:
                draw_sine_path(v.user_scn, info["target_x"], info["target_y"])
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
