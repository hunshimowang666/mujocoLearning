"""
26_water_snake_pid.py
=====================
PID torque control for sw2urdfWS2d/view.xml.

The controller drives J1 and J2 to the same target angle.
Angles exposed in this file are in degrees; MuJoCo state/control math uses radians.

Controls:
  R      reset
  Q/Esc  quit
"""

import os
import time

import mujoco
import numpy as np
from mujoco import viewer
from mujoco.glfw import glfw


TARGET_JOINT_DEG = {
    "J1": 30.0,
    "J2": -30.0,
}

KP = 8.0
KI = 0.05
KD = 1.2
INTEGRAL_LIMIT = 1.5
TORQUE_LIMIT = 20.0


class JointPID:
    def __init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(self, error, qvel, dt):
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
        torque = KP * error + KI * self.integral - KD * qvel
        return float(np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "3dModels", "sw2urdfWS2d", "view.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    joints = []
    for name, target_deg in TARGET_JOINT_DEG.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Joint '{name}' not found")
        joints.append(
            {
                "name": name,
                "joint_id": joint_id,
                "qpos_adr": model.jnt_qposadr[joint_id],
                "dof_adr": model.jnt_dofadr[joint_id],
                "target": np.deg2rad(target_deg),
                "pid": JointPID(),
            }
        )

    def reset_pose():
        mujoco.mj_resetData(model, data)
        for joint in joints:
            data.qpos[joint["qpos_adr"]] = 0.0
            data.qvel[joint["dof_adr"]] = 0.0
            joint["pid"].reset()
        mujoco.mj_forward(model, data)

    reset_pose()
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded: {xml_path}")
    print(f"Gravity: {model.opt.gravity}")
    print(f"PID: kp={KP}, ki={KI}, kd={KD}, torque_limit={TORQUE_LIMIT} Nm")
    for joint in joints:
        axis = model.jnt_axis[joint["joint_id"]]
        print(
            f"Joint {joint['name']}: target={np.rad2deg(joint['target']):.1f} deg, "
            f"axis(local)={axis}, qpos_adr={joint['qpos_adr']}, dof_adr={joint['dof_adr']}"
        )
    print("Controls: R reset, Q/Esc quit")

    with viewer.launch_passive(model, data, key_callback=on_key) as v:
        wall_start = time.perf_counter()
        t_print = 0.0

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            if data.time > wall_elapsed:
                time.sleep(data.time - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_R:
                    reset_pose()
                    wall_start = time.perf_counter()
                    t_print = 0.0
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            data.qfrc_applied[:] = 0.0
            rows = []
            for joint in joints:
                q = float(data.qpos[joint["qpos_adr"]])
                dq = float(data.qvel[joint["dof_adr"]])
                error = joint["target"] - q
                torque = joint["pid"].compute(error, dq, model.opt.timestep)
                data.qfrc_applied[joint["dof_adr"]] = torque
                rows.append((joint["name"], q, dq, error, torque))

            mujoco.mj_step(model, data)

            if data.time - t_print >= 0.5:
                status = " | ".join(
                    f"{name}: q={np.rad2deg(q):+.2f} deg, "
                    f"err={np.rad2deg(error):+.2f} deg, "
                    f"dq={np.rad2deg(dq):+.1f} deg/s, "
                    f"tau={torque:+.3f} Nm"
                    for name, q, dq, error, torque in rows
                )
                print(f"t={data.time:5.2f}s | {status}")
                t_print = data.time

            v.sync()


if __name__ == "__main__":
    main()
