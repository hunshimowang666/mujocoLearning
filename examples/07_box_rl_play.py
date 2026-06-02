"""
07_box_rl_play.py
=================
Play the trained RL hover policy with MuJoCo rendering and camera PIP.

Usage:
  ./venv/bin/python examples/07_box_rl_play.py
"""

import argparse
import os
import time

import mujoco
import numpy as np
from mujoco.glfw import glfw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from box_hover_env import SimpleBoxHoverEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "simple_box_hover")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "best_model.zip")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "box_hover_latest.zip")
DEFAULT_NORM = os.path.join(MODEL_DIR, "vec_normalize.pkl")

BUTTON_LEFT = False
BUTTON_MIDDLE = False
BUTTON_RIGHT = False
LAST_X = 0.0
LAST_Y = 0.0
SHOW_COORDINATE_FRAMES = True
IMPACT_DELTA_VZ = 1.0


def make_policy_env(norm_path):
    env = DummyVecEnv([lambda: SimpleBoxHoverEnv()])
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    return env


def add_axes_to_scene(mujoco_module, scene, origin, rotation, length):
    origin = np.asarray(origin, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    size = np.array([0.02, 0.02, 0.08], dtype=np.float64)
    axes = (
        (rotation[:, 0], np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        (rotation[:, 1], np.array([0.0, 0.8, 0.0, 1.0], dtype=np.float32)),
        (rotation[:, 2], np.array([0.1, 0.3, 1.0, 1.0], dtype=np.float32)),
    )

    if scene.ngeom + 3 > scene.maxgeom:
        return

    for direction, rgba in axes:
        geom = scene.geoms[scene.ngeom]
        endpoint = origin + length * direction
        mujoco_module.mjv_initGeom(
            geom,
            mujoco_module.mjtGeom.mjGEOM_ARROW,
            size,
            origin,
            identity,
            rgba,
        )
        mujoco_module.mjv_connector(
            geom,
            mujoco_module.mjtGeom.mjGEOM_ARROW,
            0.025,
            origin,
            endpoint,
        )
        scene.ngeom += 1


def play(model_path, norm_path):
    if not os.path.exists(model_path):
        if model_path == DEFAULT_MODEL and os.path.exists(FALLBACK_MODEL):
            model_path = FALLBACK_MODEL
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\nRun: ./venv/bin/python examples/06_box_rl_train.py"
            )

    policy_env = make_policy_env(norm_path)
    policy = PPO.load(model_path, env=policy_env, device="cuda")

    env = SimpleBoxHoverEnv()
    obs, _ = env.reset(seed=0)
    obs_v = policy_env.normalize_obs(obs.reshape(1, -1)) if isinstance(policy_env, VecNormalize) else obs.reshape(1, -1)

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    window = glfw.create_window(1280, 800, "MuJoCo RL box hover with camera PIP", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    main_cam = mujoco.MjvCamera()
    pip_cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=10000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    show_coordinate_frames = SHOW_COORDINATE_FRAMES

    def apply_visual_options():
        opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = int(show_coordinate_frames)
        opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = int(show_coordinate_frames)

    def add_coordinate_frames_to_scene():
        if not show_coordinate_frames:
            return
        add_axes_to_scene(mujoco, scene, np.zeros(3), np.eye(3), 0.6)
        add_axes_to_scene(mujoco, scene, env.data.xpos[env.box_id], env.data.xmat[env.box_id], 0.35)

    apply_visual_options()

    mujoco.mjv_defaultCamera(main_cam)
    main_cam.distance = 2.0
    main_cam.elevation = -25.0
    main_cam.azimuth = 135.0
    main_cam.lookat[:] = [0.0, 0.0, env.target_z]

    camera_name = "box_camera"
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    show_pip = camera_id >= 0
    if show_pip:
        mujoco.mjv_defaultCamera(pip_cam)
        pip_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        pip_cam.fixedcamid = camera_id

    pressed_keys = []

    def keyboard(window, key, scancode, act, mods):
        del scancode, mods
        if act != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        else:
            pressed_keys.append(key)

    def mouse_button(window, button, act, mods):
        del button, act, mods
        global BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, LAST_X, LAST_Y
        BUTTON_LEFT = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        BUTTON_MIDDLE = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        BUTTON_RIGHT = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        LAST_X, LAST_Y = glfw.get_cursor_pos(window)

    def mouse_move(window, xpos, ypos):
        global LAST_X, LAST_Y
        if not (BUTTON_LEFT or BUTTON_MIDDLE or BUTTON_RIGHT):
            LAST_X, LAST_Y = xpos, ypos
            return

        _, height = glfw.get_window_size(window)
        dx = xpos - LAST_X
        dy = ypos - LAST_Y
        LAST_X, LAST_Y = xpos, ypos
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if BUTTON_RIGHT:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif BUTTON_LEFT:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(env.model, action, dx / height, dy / height, scene, main_cam)

    def scroll(window, xoffset, yoffset):
        del window, xoffset
        mujoco.mjv_moveCamera(env.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, main_cam)

    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)

    print("Controls: A gentle downward impact, Z gentle upward impact, R reset, C coordinate/inertia display, Q/Esc quit")
    wall_start = time.perf_counter()
    sim_start = env.data.time
    t_print = 0.0

    try:
        while not glfw.window_should_close(window):
            elapsed = time.perf_counter() - wall_start
            sim_elapsed = env.data.time - sim_start
            if sim_elapsed > elapsed:
                time.sleep(sim_elapsed - elapsed)

            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_R:
                    obs, _ = env.reset(seed=0)
                    obs_v = policy_env.normalize_obs(obs.reshape(1, -1)) if isinstance(policy_env, VecNormalize) else obs.reshape(1, -1)
                    wall_start = time.perf_counter()
                    sim_start = env.data.time
                    print("[R] reset")
                elif key == glfw.KEY_A:
                    env.data.qvel[2] -= IMPACT_DELTA_VZ
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[A] gentle downward impact: vz={env.data.qvel[2]:+.2f}")
                elif key == glfw.KEY_Z:
                    env.data.qvel[2] += IMPACT_DELTA_VZ
                    mujoco.mj_forward(env.model, env.data)
                    print(f"[Z] gentle upward impact: vz={env.data.qvel[2]:+.2f}")
                elif key == glfw.KEY_C:
                    show_coordinate_frames = not show_coordinate_frames
                    apply_visual_options()
                    print(f"[C] coordinate/inertia display: {show_coordinate_frames}")
                elif key == glfw.KEY_Q:
                    glfw.set_window_should_close(window, True)

            action, _ = policy.predict(obs_v, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs_v = policy_env.normalize_obs(obs.reshape(1, -1)) if isinstance(policy_env, VecNormalize) else obs.reshape(1, -1)

            if terminated or truncated:
                obs, _ = env.reset()
                obs_v = policy_env.normalize_obs(obs.reshape(1, -1)) if isinstance(policy_env, VecNormalize) else obs.reshape(1, -1)
                wall_start = time.perf_counter()
                sim_start = env.data.time

            if env.data.time - t_print >= 1.0:
                print(f"t={env.data.time:4.1f}s | z={info['z']:.3f}m | force={info['force']:+6.1f}N | reward={reward:+.2f}")
                t_print = env.data.time

            fb_width, fb_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
            mujoco.mjv_updateScene(env.model, env.data, opt, None, main_cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            add_coordinate_frames_to_scene()
            mujoco.mjr_render(viewport, scene, context)

            if show_pip:
                margin = max(12, fb_width // 100)
                pip_width = max(260, fb_width // 4)
                pip_height = max(180, fb_height // 4)
                pip_rect = mujoco.MjrRect(
                    fb_width - pip_width - margin,
                    margin,
                    pip_width,
                    pip_height,
                )
                mujoco.mjv_updateScene(env.model, env.data, opt, None, pip_cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                add_coordinate_frames_to_scene()
                mujoco.mjr_render(pip_rect, scene, context)

            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        glfw.terminate()
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
