"""
08_calf_foot_pid.py
===================
PID torque control for calf_foot_hinge.xml.

The controller keeps ankle_hinge near q=0, which is the upright pose in the XML.

Controls:
  A      add a gentle negative angular-velocity disturbance
  Z      add a gentle positive angular-velocity disturbance
  R      reset
  Q/Esc  quit
"""

import os
import time

import mujoco
import numpy as np
from mujoco import viewer
from mujoco.glfw import glfw


INITIAL_Q = 0.12
TARGET_Q = INITIAL_Q
KP = 1.8
KI = 0.08
KD = 0.18
TORQUE_LIMIT = 2.0
IMPACT_DQ = 5.0
DISTURBANCE_TORQUE = 0.7
DISTURBANCE_DURATION = 0.12


class JointPID:
    def __init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(self, error, qvel, dt):
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -2.0, 2.0))
        torque = KP * error + KI * self.integral - KD * qvel
        return float(np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "3dModels", "calf_foot_hinge.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_hinge")
    if joint_id < 0:
        raise RuntimeError("Joint 'ankle_hinge' not found")

    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    joint_axis = model.jnt_axis[joint_id].copy()
    pid = JointPID()

    def reset_pose():
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr] = INITIAL_Q
        data.qvel[dof_adr] = 0.0
        mujoco.mj_forward(model, data)
        pid.reset()

    reset_pose()

    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded: {xml_path}")
    print(f"Joint: ankle_hinge, axis={joint_axis}")
    print(f"Initial joint angle: {INITIAL_Q:.3f} rad")
    print(f"Target joint angle: {TARGET_Q:.3f} rad")
    print(f"PID: kp={KP}, ki={KI}, kd={KD}, torque_limit={TORQUE_LIMIT} Nm")
    print("Controls: A/Z disturbance, R reset, Q/Esc quit")

    with viewer.launch_passive(model, data, key_callback=on_key) as v:
        wall_start = time.perf_counter()
        t_print = 0.0
        disturbance_until = 0.0
        disturbance_torque = 0.0

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            if data.time > wall_elapsed:
                time.sleep(data.time - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_A:
                    data.qvel[dof_adr] -= IMPACT_DQ
                    disturbance_torque = -DISTURBANCE_TORQUE
                    disturbance_until = data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(model, data)
                    print(
                        f"[A] disturbance: qvel={data.qvel[dof_adr]:+.2f} rad/s, "
                        f"pulse={disturbance_torque:+.2f} Nm"
                    )
                elif key == glfw.KEY_Z:
                    data.qvel[dof_adr] += IMPACT_DQ
                    disturbance_torque = DISTURBANCE_TORQUE
                    disturbance_until = data.time + DISTURBANCE_DURATION
                    mujoco.mj_forward(model, data)
                    print(
                        f"[Z] disturbance: qvel={data.qvel[dof_adr]:+.2f} rad/s, "
                        f"pulse={disturbance_torque:+.2f} Nm"
                    )
                elif key == glfw.KEY_R:
                    reset_pose()
                    disturbance_until = 0.0
                    disturbance_torque = 0.0
                    wall_start = time.perf_counter()
                    t_print = 0.0
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            q = float(data.qpos[qpos_adr])
            dq = float(data.qvel[dof_adr])
            error = TARGET_Q - q
            torque = pid.compute(error, dq, model.opt.timestep)
            if data.time < disturbance_until:
                torque += disturbance_torque

            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_adr] = torque

            mujoco.mj_step(model, data)

            if data.time - t_print >= 0.5:
                print(
                    f"t={data.time:5.2f}s | q={q:+.3f} rad | dq={dq:+.3f} rad/s | "
                    f"err={error:+.3f} | tau={torque:+.3f} Nm"
                )
                t_print = data.time

            v.sync()


if __name__ == "__main__":
    main()
