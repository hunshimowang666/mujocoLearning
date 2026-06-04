"""
25_go2_backflip_rl_play.py
==========================
Play the trained MuJoCo Go2 backflip RL policy.

Usage:
  ./venv/bin/python examples/25_go2_backflip_rl_play.py
"""

import argparse
import glob
import importlib
import os
import time

import mujoco
from mujoco import viewer
from mujoco.glfw import glfw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from go2_backflip_env import Go2BackflipEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "go2_backflip")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "best_model.zip")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "go2_backflip_latest.zip")
DEFAULT_NORM = os.path.join(MODEL_DIR, "best_model_vecnormalize.pkl")
FALLBACK_NORM = os.path.join(MODEL_DIR, "vec_normalize.pkl")

TRAIN_CFG = importlib.import_module("24_go2_backflip_rl_train")
MAX_EPISODE_STEPS = TRAIN_CFG.MAX_EPISODE_STEPS
SCRIPTED_TORQUE_SCALE = TRAIN_CFG.SCRIPTED_TORQUE_SCALE
DEBUG_ASSIST_TORQUE_Y = TRAIN_CFG.DEBUG_ASSIST_TORQUE_Y
CAMERA_LOOKAT = (0.0, 0.0, 0.32)
CAMERA_DISTANCE = 2.1
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -18.0


def configure_viewer_camera(v):
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = CAMERA_LOOKAT
    v.cam.distance = CAMERA_DISTANCE
    v.cam.azimuth = CAMERA_AZIMUTH
    v.cam.elevation = CAMERA_ELEVATION


def make_policy_env(norm_path):
    env = DummyVecEnv(
        [
            lambda: Go2BackflipEnv(
                max_steps=MAX_EPISODE_STEPS,
                scripted_torque_scale=SCRIPTED_TORQUE_SCALE,
                debug_assist_torque_y=DEBUG_ASSIST_TORQUE_Y,
            )
        ]
    )
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    return env


def checkpoint_norm_path(model_path):
    filename = os.path.basename(model_path)
    prefix = "go2_backflip_ppo_"
    suffix = "_steps.zip"
    if filename.startswith(prefix) and filename.endswith(suffix):
        steps = filename[len(prefix) : -len(suffix)]
        return os.path.join(MODEL_DIR, f"go2_backflip_ppo_vecnormalize_{steps}_steps.pkl")
    return model_path[:-4] + "_vecnormalize.pkl"


def latest_checkpoint_pair():
    models = glob.glob(os.path.join(MODEL_DIR, "go2_backflip_ppo_*_steps.zip"))
    if not models:
        return None, None
    models = sorted(
        models,
        key=lambda path: int(os.path.basename(path).split("_")[-2]),
    )
    for model_path in reversed(models):
        norm_path = checkpoint_norm_path(model_path)
        if os.path.exists(norm_path):
            return model_path, norm_path
    return models[-1], None


def resolve_model_and_norm(model_path, norm_path):
    if os.path.exists(model_path) and os.path.exists(norm_path):
        return model_path, norm_path

    if model_path == DEFAULT_MODEL and os.path.exists(DEFAULT_MODEL):
        if os.path.exists(DEFAULT_NORM):
            return DEFAULT_MODEL, DEFAULT_NORM
        if os.path.exists(FALLBACK_NORM):
            return DEFAULT_MODEL, FALLBACK_NORM

    checkpoint_model, checkpoint_norm = latest_checkpoint_pair()
    if checkpoint_model is not None and checkpoint_norm is not None:
        print(
            "[Go2BackflipRL] Default model/norm pair was not complete; "
            f"using checkpoint {checkpoint_model}"
        )
        return checkpoint_model, checkpoint_norm

    if model_path == DEFAULT_MODEL and os.path.exists(FALLBACK_MODEL):
        if os.path.exists(FALLBACK_NORM):
            return FALLBACK_MODEL, FALLBACK_NORM
        return FALLBACK_MODEL, norm_path

    return model_path, norm_path


def normalize_obs(policy_env, obs):
    obs_v = obs.reshape(1, -1)
    if isinstance(policy_env, VecNormalize):
        obs_v = policy_env.normalize_obs(obs_v)
    return obs_v


def play(model_path, norm_path):
    model_path, norm_path = resolve_model_and_norm(model_path, norm_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: ./venv/bin/python examples/24_go2_backflip_rl_train.py"
        )
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"VecNormalize file not found: {norm_path}\n"
            "Use a checkpoint that has its matching *_vecnormalize_*.pkl file, "
            "or finish training once to create vec_normalize.pkl."
        )

    policy_env = make_policy_env(norm_path)
    policy = PPO.load(model_path, env=policy_env, device=TRAIN_CFG.DEVICE)

    env = Go2BackflipEnv(
        max_steps=MAX_EPISODE_STEPS,
        initial_joint_noise_deg=0.0,
        initial_joint_vel_noise=0.0,
        scripted_torque_scale=SCRIPTED_TORQUE_SCALE,
        debug_assist_torque_y=DEBUG_ASSIST_TORQUE_Y,
    )
    obs, _ = env.reset(seed=0)
    obs_v = normalize_obs(policy_env, obs)
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded policy: {model_path}")
    print(f"Loaded VecNormalize: {norm_path}")
    print(
        f"scripted_torque_scale={SCRIPTED_TORQUE_SCALE}, "
        f"debug_assist_torque_y={DEBUG_ASSIST_TORQUE_Y}"
    )
    print("Controls: R reset, Q/Esc quit")

    mujoco.mj_forward(env.model, env.data)

    with viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        configure_viewer_camera(v)
        v.sync()
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
                    configure_viewer_camera(v)
                    v.sync()
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            action, _ = policy.predict(obs_v, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs_v = normalize_obs(policy_env, obs)

            if env.data.time - t_print >= 0.2 and info:
                print(
                    f"t={info['elapsed']:5.2f}s | phase={info['phase_name']:>11s} | "
                    f"flip={info['flip_angle_deg']:+7.1f}/{info['desired_flip_angle_deg']:+6.1f} deg | "
                    f"final_err={info['final_flip_error_deg']:+6.1f} deg | "
                    f"z={info['base_z']:.3f} max_z={info['max_base_z']:.3f} | "
                    f"up={info['base_up_z']:+.2f} | feet={info['foot_contacts']:.0f} | "
                    f"body_hit={info['hard_nonfoot_contacts']} | "
                    f"tau={info['torque_rms']:.1f} | reward={reward:+.2f}"
                )
                t_print = env.data.time

            if terminated or truncated:
                if info.get("success"):
                    print("[reset] success")
                elif terminated:
                    reasons = []
                    if info.get("too_low"):
                        reasons.append("base too low")
                    if info.get("body_hit_ground"):
                        reasons.append("body hit ground")
                    if info.get("nonfinite"):
                        reasons.append("nonfinite")
                    print(f"[reset] terminated: {', '.join(reasons) or 'unknown'}")
                elif truncated:
                    print("[reset] episode complete")
                obs, _ = env.reset()
                obs_v = normalize_obs(policy_env, obs)
                wall_start = time.perf_counter()
                sim_start = env.data.time
                t_print = 0.0
                configure_viewer_camera(v)
                v.sync()

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
