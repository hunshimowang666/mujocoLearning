"""
09_panda_arm_sim.py
===================
Interactive Franka Emika Panda arm demo from MuJoCo Menagerie.

Controls:
  1-7    select arm joint
  A/Z    decrease/increase selected joint target
  O/C    open/close gripper
  H/R    reset to home keyframe
  Q/Esc  quit
"""

import os
import time

import mujoco
import numpy as np
from mujoco import viewer
from mujoco.glfw import glfw


JOINT_STEP = 0.08
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(
        script_dir,
        "mujoco_menagerie",
        "franka_emika_panda",
        "scene.xml",
    )
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"找不到 Panda 模型文件: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError("Panda 模型里没有找到 keyframe 'home'")

    arm_actuators = [f"actuator{i}" for i in range(1, 8)]
    arm_act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in arm_actuators
    ]
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if any(act_id < 0 for act_id in arm_act_ids) or gripper_act_id < 0:
        raise RuntimeError("Panda 模型 actuator 名称与脚本预期不一致")

    ctrl_ranges = model.actuator_ctrlrange.copy()
    selected = 0
    pressed_keys = []

    def reset_home():
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)

    def clamp_ctrl(act_id):
        lo, hi = ctrl_ranges[act_id]
        data.ctrl[act_id] = np.clip(data.ctrl[act_id], lo, hi)

    def print_status():
        act_id = arm_act_ids[selected]
        lo, hi = ctrl_ranges[act_id]
        print(
            f"Selected joint {selected + 1}: target={data.ctrl[act_id]:+.3f} rad "
            f"range=[{lo:+.3f}, {hi:+.3f}]"
        )

    def on_key(keycode):
        pressed_keys.append(keycode)

    reset_home()

    print(f"Loaded: {xml_path}")
    print("Controls: 1-7 select joint, A/Z move, O/C gripper, H/R home, Q/Esc quit")
    print_status()

    with viewer.launch_passive(model, data, key_callback=on_key) as v:
        wall_start = time.perf_counter()

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            if data.time > wall_elapsed:
                time.sleep(data.time - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if glfw.KEY_1 <= key <= glfw.KEY_7:
                    selected = key - glfw.KEY_1
                    print_status()
                elif key == glfw.KEY_A:
                    act_id = arm_act_ids[selected]
                    data.ctrl[act_id] -= JOINT_STEP
                    clamp_ctrl(act_id)
                    print_status()
                elif key == glfw.KEY_Z:
                    act_id = arm_act_ids[selected]
                    data.ctrl[act_id] += JOINT_STEP
                    clamp_ctrl(act_id)
                    print_status()
                elif key == glfw.KEY_O:
                    data.ctrl[gripper_act_id] = GRIPPER_OPEN
                    print("[O] gripper open")
                elif key == glfw.KEY_C:
                    data.ctrl[gripper_act_id] = GRIPPER_CLOSED
                    print("[C] gripper closed")
                elif key in (glfw.KEY_H, glfw.KEY_R):
                    reset_home()
                    wall_start = time.perf_counter()
                    print("[H/R] reset to home")
                    print_status()
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
