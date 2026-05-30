"""
10_camera_pip.py
================
Load an MJCF model and show a fixed model camera as a picture-in-picture view.

Usage:
  ./venv/bin/python examples/10_camera_pip.py
  ./venv/bin/python examples/10_camera_pip.py --model examples/3dModels/simpleBox.xml --camera box_camera

Controls:
  Space  pause / resume
  R      reset
  Esc    quit
  Mouse  rotate / pan / zoom the main view
"""

import argparse
import os
import sys

try:
    import mujoco
    from mujoco.glfw import glfw
except ModuleNotFoundError as exc:
    print("Missing dependency:", exc)
    print("Install with:")
    print("  ./venv/bin/python -m pip install mujoco")
    sys.exit(1)


BUTTON_LEFT = False
BUTTON_MIDDLE = False
BUTTON_RIGHT = False
LAST_X = 0.0
LAST_Y = 0.0
PAUSED = False


def make_callbacks(model, data, scene, main_cam):
    def keyboard(window, key, scancode, act, mods):
        del scancode, mods
        global PAUSED

        if act != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            PAUSED = not PAUSED
        elif key == glfw.KEY_R:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

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

        width, height = glfw.get_window_size(window)
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

        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, main_cam)

    def scroll(window, xoffset, yoffset):
        del window, xoffset
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, main_cam)

    return keyboard, mouse_button, mouse_move, scroll


def draw_overlay_label(rect, context, text):
    if hasattr(mujoco, "mjr_overlay"):
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect,
            text,
            "",
            context,
        )


def run(model_path, camera_name, width, height, frames):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        raise ValueError(f"Camera '{camera_name}' not found. Available cameras: {names}")

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    window = glfw.create_window(width, height, "MuJoCo camera picture-in-picture", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    main_cam = mujoco.MjvCamera()
    pip_cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_defaultCamera(main_cam)
    main_cam.distance = 2.0
    main_cam.elevation = -25.0
    main_cam.azimuth = 135.0
    main_cam.lookat[:] = [0.0, 0.0, 0.5]

    mujoco.mjv_defaultCamera(pip_cam)
    pip_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    pip_cam.fixedcamid = camera_id

    keyboard, mouse_button, mouse_move, scroll = make_callbacks(model, data, scene, main_cam)
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)

    print(f"Loaded: {model_path}")
    print(f"Picture-in-picture camera: {camera_name}")
    print("Controls: Space pause/resume, R reset, Esc quit")

    rendered_frames = 0
    try:
        while not glfw.window_should_close(window):
            if not PAUSED:
                sim_start = data.time
                while data.time - sim_start < 1.0 / 60.0:
                    mujoco.mj_step(model, data)

            fb_width, fb_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)

            mujoco.mjv_updateScene(
                model, data, opt, None, main_cam, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            mujoco.mjr_render(viewport, scene, context)

            margin = max(12, fb_width // 100)
            pip_width = max(260, fb_width // 4)
            pip_height = max(180, fb_height // 4)
            pip_x = fb_width - pip_width - margin
            pip_y = margin

            if hasattr(mujoco, "mjr_rectangle"):
                border = mujoco.MjrRect(pip_x - 4, pip_y - 4, pip_width + 8, pip_height + 8)
                mujoco.mjr_rectangle(border, 0.02, 0.02, 0.02, 1.0)

            pip_rect = mujoco.MjrRect(pip_x, pip_y, pip_width, pip_height)
            mujoco.mjv_updateScene(
                model, data, opt, None, pip_cam, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            mujoco.mjr_render(pip_rect, scene, context)
            draw_overlay_label(pip_rect, context, camera_name)

            glfw.swap_buffers(window)
            glfw.poll_events()
            rendered_frames += 1
            if frames and rendered_frames >= frames:
                break
    finally:
        glfw.terminate()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    default_model = os.path.join(repo_dir, "examples", "3dModels", "simpleBox.xml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model, help="Path to the MJCF XML model")
    parser.add_argument("--camera", default="box_camera", help="Fixed camera name for the small view")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=800, help="Window height")
    parser.add_argument("--frames", type=int, default=0, help="Render N frames then exit; 0 means run until closed")
    args = parser.parse_args()

    run(os.path.abspath(args.model), args.camera, args.width, args.height, args.frames)


if __name__ == "__main__":
    main()
