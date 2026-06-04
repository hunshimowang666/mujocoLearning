"""
23_go2_backflip_mujoco_attempt.py
=================================
Scripted MuJoCo attempt at a Unitree Go2 backflip.

Important:
  Unitree's official SDK BackFlip() is a high-level firmware command. The
  low-level joint targets/torques are not public, so this file is not a direct
  port of that controller. It is a MuJoCo-only torque-control experiment that
  tries a crouch -> rear-biased thrust -> tuck -> extend -> recover sequence.

Usage:
  ./venv/bin/python examples/23_go2_backflip_mujoco_attempt.py
  ./venv/bin/python examples/23_go2_backflip_mujoco_attempt.py --no-viewer

Controls:
  R  reset
  Q/Esc quit
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np
from mujoco import viewer
from mujoco.glfw import glfw


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(
    SCRIPT_DIR,
    "unitree_rl_mjlab",
    "src",
    "assets",
    "robots",
    "unitree_go2",
    "xmls",
    "scene_go2_torque.xml",
)

JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

HIP = (0, 3, 6, 9)
THIGH = (1, 4, 7, 10)
CALF = (2, 5, 8, 11)
FRONT_THIGH = (1, 4)
FRONT_CALF = (2, 5)
REAR_THIGH = (7, 10)
REAR_CALF = (8, 11)

HOME_DEG = np.array([0.0, 51.6, -103.1] * 4, dtype=np.float64)

# GO2 torque limits from the original XML motor ctrlrange.
TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float64)

# This scripted attempt deliberately uses a stronger rear-leg extension than a
# static pose controller. Tune these first before attempting RL imitation.
KP = np.array([32.0, 72.0, 86.0] * 4, dtype=np.float64)
KD = np.array([1.1, 3.0, 3.4] * 4, dtype=np.float64)

def deg_pose(hip=0.0, thigh=51.6, calf=-103.1):
    return np.array([hip, thigh, calf] * 4, dtype=np.float64)


def make_pose(**updates):
    pose = deg_pose()
    for indices, value in updates.items():
        if indices == "front_thigh":
            pose[list(FRONT_THIGH)] = value
        elif indices == "front_calf":
            pose[list(FRONT_CALF)] = value
        elif indices == "rear_thigh":
            pose[list(REAR_THIGH)] = value
        elif indices == "rear_calf":
            pose[list(REAR_CALF)] = value
        elif indices == "all_thigh":
            pose[list(THIGH)] = value
        elif indices == "all_calf":
            pose[list(CALF)] = value
        elif indices == "all_hip":
            pose[list(HIP)] = value
        else:
            raise ValueError(f"Unknown pose field: {indices}")
    return pose


PHASES = (
    # name, end_time, target pose in degrees
    ("settle", 0.25, deg_pose(0.0, 51.6, -103.1)),
    ("crouch", 0.62, deg_pose(0.0, 72.0, -138.0)),
    # Rear legs extend first to create backward pitch impulse.
    (
        "rear_thrust",
        0.78,
        make_pose(front_thigh=76.0, front_calf=-142.0, rear_thigh=15.0, rear_calf=-86.0),
    ),
    # Front legs extend shortly after to help lift the front of the body.
    (
        "full_thrust",
        0.92,
        make_pose(front_thigh=10.0, front_calf=-86.0, rear_thigh=3.0, rear_calf=-86.0),
    ),
    ("tuck", 1.32, deg_pose(0.0, 96.0, -157.0)),
    ("open", 1.62, deg_pose(0.0, 35.0, -94.0)),
    ("recover", 2.15, deg_pose(0.0, 51.6, -103.1)),
    ("hold", 10.0, deg_pose(0.0, 51.6, -103.1)),
)


class Go2BackflipAttempt:
    def __init__(self, torque_scale=1.0, assist_torque_y=0.0):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.torque_limits = TORQUE_LIMITS * float(torque_scale)
        self.assist_torque_y = float(assist_torque_y)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        )
        if np.any(self.joint_ids < 0):
            missing = [name for name, jid in zip(JOINT_NAMES, self.joint_ids) if jid < 0]
            raise RuntimeError(f"Missing joints: {missing}")
        self.qpos_adrs = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids])
        self.dof_adrs = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids])
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = 0.285
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.start_time = self.data.time
        self.max_z = float(self.data.qpos[2])
        self.min_z = float(self.data.qpos[2])
        self.max_abs_flip_angle = 0.0
        self._last_raw_flip_angle = self.raw_flip_angle()
        self._unwrapped_flip_angle = 0.0
        self.last_phase = None

    def raw_flip_angle(self):
        """Body pitch-like angle from the body x-axis projected into world x-z."""
        xmat = self.data.xmat[self.base_id].reshape(3, 3)
        body_x_axis = xmat[:, 0]
        return float(np.arctan2(-body_x_axis[2], body_x_axis[0]))

    def update_unwrapped_flip_angle(self):
        raw = self.raw_flip_angle()
        delta = raw - self._last_raw_flip_angle
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        self._unwrapped_flip_angle += delta
        self._last_raw_flip_angle = raw
        self.max_abs_flip_angle = max(self.max_abs_flip_angle, abs(self._unwrapped_flip_angle))
        return self._unwrapped_flip_angle

    def target_for_time(self, t):
        previous_time = 0.0
        previous_pose = HOME_DEG
        for name, end_time, pose_deg in PHASES:
            if t <= end_time:
                alpha = (t - previous_time) / max(end_time - previous_time, 1e-9)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                # Smooth interpolation avoids an unrealistic torque step.
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                target_deg = (1.0 - alpha) * previous_pose + alpha * pose_deg
                return name, np.deg2rad(target_deg)
            previous_time = end_time
            previous_pose = pose_deg
        return PHASES[-1][0], np.deg2rad(PHASES[-1][2])

    def step(self):
        t = self.data.time - self.start_time
        phase, target = self.target_for_time(t)
        target = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        q = self.data.qpos[self.qpos_adrs]
        dq = self.data.qvel[self.dof_adrs]
        torque = KP * (target - q) - KD * dq
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)

        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_adrs] = torque

        if self.assist_torque_y != 0.0 and 0.75 <= t <= 1.05:
            # Debug-only body torque around the freejoint y angular DOF. This is
            # not a robot-realistic controller, but it helps separate "trajectory
            # is weak" from "MuJoCo model cannot rotate".
            self.data.qfrc_applied[4] += self.assist_torque_y

        mujoco.mj_step(self.model, self.data)

        base_z = float(self.data.qpos[2])
        flip_angle = self.update_unwrapped_flip_angle()
        self.max_z = max(self.max_z, base_z)
        self.min_z = min(self.min_z, base_z)
        return {
            "t": t,
            "phase": phase,
            "base_z": base_z,
            "flip_angle_deg": float(np.rad2deg(flip_angle)),
            "max_z": self.max_z,
            "max_abs_flip_angle_deg": float(np.rad2deg(self.max_abs_flip_angle)),
            "torque_rms": float(np.sqrt(np.mean(np.square(torque)))),
            "contacts": int(self.data.ncon),
        }


def run_no_viewer(sim, duration):
    next_print = 0.0
    info = {}
    while sim.data.time - sim.start_time < duration:
        info = sim.step()
        if info["t"] >= next_print:
            print_info(info)
            next_print += 0.1
    print("Final summary:")
    print_info(info)


def print_info(info):
    print(
        f"t={info['t']:5.2f}s | phase={info['phase']:>11s} | "
        f"z={info['base_z']:.3f} m | flip={info['flip_angle_deg']:+7.1f} deg | "
        f"max_z={info['max_z']:.3f} | max|flip|={info['max_abs_flip_angle_deg']:.1f} | "
        f"tau_rms={info['torque_rms']:.1f} Nm | contacts={info['contacts']}"
    )


def run_viewer(sim, duration):
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print("MuJoCo scripted Go2 backflip attempt")
    print("This is not Unitree's firmware controller; it is a tunable PD trajectory experiment.")
    print("Controls: R reset, Q/Esc quit")

    with viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as v:
        v.cam.distance = 2.1
        v.cam.azimuth = 135
        v.cam.elevation = -18
        v.cam.lookat[:] = [0.0, 0.0, 0.25]

        wall_start = time.perf_counter()
        sim_start = sim.data.time
        next_print = 0.0

        while v.is_running():
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed = sim.data.time - sim_start
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            should_quit = False
            while pressed_keys:
                key = pressed_keys.pop(0)
                if key == glfw.KEY_R:
                    sim.reset()
                    wall_start = time.perf_counter()
                    sim_start = sim.data.time
                    next_print = 0.0
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            info = sim.step()
            if info["t"] >= next_print:
                print_info(info)
                next_print += 0.2

            if info["t"] >= duration:
                sim.reset()
                wall_start = time.perf_counter()
                sim_start = sim.data.time
                next_print = 0.0
                print("[auto] reset")

            v.sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=1.0,
        help="Scale original GO2 torque limits. Values >1 are non-realistic but useful for debugging.",
    )
    parser.add_argument(
        "--assist-torque-y",
        type=float,
        default=0.0,
        help="Debug-only free-body pitch torque during takeoff. 0 keeps joint-torque-only control.",
    )
    args = parser.parse_args()

    sim = Go2BackflipAttempt(torque_scale=args.torque_scale, assist_torque_y=args.assist_torque_y)
    if args.no_viewer:
        run_no_viewer(sim, args.duration)
    else:
        run_viewer(sim, args.duration)


if __name__ == "__main__":
    main()
