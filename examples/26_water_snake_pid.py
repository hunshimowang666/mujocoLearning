"""
26_water_snake_pid.py
=====================
PID torque control for sw2urdfWS2d/view.xml.

The controller reads newSnakeExplore2D.txt as a target sequence:
  col 1      time (s)
  col 2-4    backDrivenCabin x, y, yaw
  col 5-6    J1, J2 target angles for this 3-cabin model
Every PATH_ROW_UPDATE_INTERVAL seconds, the next row becomes the new target.
The episode resets after the final timestamp in the path file.
The root body is moved directly to the row pose; J1 and J2 are still PID-controlled.
Angles exposed in this file are in degrees; MuJoCo state/control math uses radians.

The path file is interpreted as a right-handed, Z-down inertial frame:
  planner +X -> MuJoCo +X
  planner +Y -> MuJoCo -Y
  planner +Z -> MuJoCo -Z

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
    "J1": 0.0,
    "J2": 0.0,
}

USE_FIRST_PATH_ROW_AS_INITIAL = True
PATH_FILE = "newSnakeExplore2D.txt"
PATH_ROW_UPDATE_INTERVAL = 0.1
ENABLE_KINEMATIC_PATH_MOVE = True
SHOW_WORLD_AXES = True
SHOW_PATH_CURVE = True
WORLD_AXES_LENGTH = 0.45
ZERO_DEPTH_WORLD_Z = 0.45
PATH_CURVE_DEPTH = 0.0

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


def add_arrow(scene, start, end, rgba, radius=0.018):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([radius, radius, radius * 4.0], dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def add_capsule(scene, start, end, rgba, radius=0.01):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


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


def planner_position_to_mujoco(x, y, z_down=0.0):
    return np.array([x, -y, ZERO_DEPTH_WORLD_Z - z_down], dtype=np.float64)


def planner_yaw_to_mujoco_root_yaw(yaw):
    # The cabin chain points along body-local -Y, so yaw=0 in the planner
    # must rotate local -Y onto MuJoCo +X.
    return 0.5 * np.pi - yaw


def planner_joint_deg_to_mujoco(joint_deg):
    # Planner joint angles rotate around +Z in the Z-down planning frame.
    # That axis maps to MuJoCo -Z, so hinge signs are opposite in MuJoCo.
    return -joint_deg


def draw_world_axes(scene):
    origin = np.zeros(3, dtype=np.float64)
    add_arrow(
        scene,
        origin,
        np.array([WORLD_AXES_LENGTH, 0.0, 0.0], dtype=np.float64),
        (1.0, 0.0, 0.0, 1.0),
    )
    add_arrow(
        scene,
        origin,
        np.array([0.0, -WORLD_AXES_LENGTH, 0.0], dtype=np.float64),
        (0.0, 0.8, 0.0, 1.0),
    )
    add_arrow(
        scene,
        origin,
        np.array([0.0, 0.0, -WORLD_AXES_LENGTH], dtype=np.float64),
        (0.1, 0.3, 1.0, 1.0),
    )


def read_path_table(path):
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 6:
        raise RuntimeError(
            f"{path} needs at least 6 columns: time, x, y, yaw, J1, J2"
        )
    return table


def draw_path_curve(scene, path_table):
    if path_table is None or len(path_table) == 0:
        return
    points = np.array(
        [
            planner_position_to_mujoco(row[1], row[2], PATH_CURVE_DEPTH)
            for row in path_table
        ],
        dtype=np.float64,
    )
    for start, end in zip(points[:-1], points[1:]):
        add_capsule(scene, start, end, (0.0, 0.9, 1.0, 0.85), radius=0.008)
    add_sphere(scene, points[0], 0.025, (0.0, 1.0, 0.0, 1.0))
    add_sphere(scene, points[-1], 0.025, (1.0, 0.2, 0.0, 1.0))


def draw_debug_scene(scene, path_table):
    scene.ngeom = 0
    if SHOW_WORLD_AXES:
        draw_world_axes(scene)
    if SHOW_PATH_CURVE:
        draw_path_curve(scene, path_table)


def path_row_to_target(row):
    if row.shape[0] < 6:
        raise RuntimeError("Path row needs at least 6 columns: time, x, y, yaw, J1, J2")
    return {
        "time": float(row[0]),
        "x": float(row[1]),
        "y": float(row[2]),
        "yaw": np.deg2rad(float(row[3])),
        "joints": {
            "J1": float(row[4]),
            "J2": float(row[5]),
        },
    }


def read_first_path_target(path):
    return path_row_to_target(read_path_table(path)[0])


def path_row_index_from_time(sim_time, row_interval, num_rows):
    return min(int(np.floor((sim_time + 1.0e-9) / row_interval)), num_rows - 1)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "3dModels", "sw2urdfWS2d", "view.xml")
    path_file = os.path.join(script_dir, PATH_FILE)
    path_table = read_path_table(path_file)
    path_targets = [path_row_to_target(row) for row in path_table]
    path_total_time = float(path_table[-1, 0])
    path_target = path_targets[0] if USE_FIRST_PATH_ROW_AS_INITIAL else None
    if path_target is not None:
        for name, target_deg in path_target["joints"].items():
            TARGET_JOINT_DEG[name] = planner_joint_deg_to_mujoco(target_deg)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    free_joint_ids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if len(free_joint_ids) == 0:
        raise RuntimeError("No freejoint found for backDrivenCabin root pose")
    root_qpos_adr = model.jnt_qposadr[free_joint_ids[0]]
    root_dof_adr = model.jnt_dofadr[free_joint_ids[0]]

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

    def apply_root_pose(target):
        root_pos = planner_position_to_mujoco(target["x"], target["y"])
        yaw = planner_yaw_to_mujoco_root_yaw(target["yaw"])
        data.qpos[root_qpos_adr + 0 : root_qpos_adr + 3] = root_pos
        data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = [
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        ]
        data.qvel[root_dof_adr : root_dof_adr + 6] = 0.0

    def set_joint_targets(target):
        for joint in joints:
            planner_deg = target["joints"][joint["name"]]
            joint["target"] = np.deg2rad(planner_joint_deg_to_mujoco(planner_deg))

    def print_path_target(prefix, row_index, target):
        print(
            f"{prefix} row={row_index + 1}/{len(path_targets)}, "
            f"file_t={target['time']:.3f}s, "
            f"x={target['x']:.3f} m, y={target['y']:.3f} m, "
            f"yaw={np.rad2deg(target['yaw']):.3f} deg, "
            f"mujoco_yaw={np.rad2deg(planner_yaw_to_mujoco_root_yaw(target['yaw'])):.3f} deg, "
            f"J1={target['joints']['J1']:.3f} deg -> {planner_joint_deg_to_mujoco(target['joints']['J1']):.3f} deg, "
            f"J2={target['joints']['J2']:.3f} deg -> {planner_joint_deg_to_mujoco(target['joints']['J2']):.3f} deg"
        )

    current_row_index = 0

    def reset_pose():
        nonlocal current_row_index, path_target
        mujoco.mj_resetData(model, data)
        current_row_index = 0
        path_target = path_targets[current_row_index] if path_targets else None
        if path_target is not None:
            apply_root_pose(path_target)
            set_joint_targets(path_target)
        for joint in joints:
            data.qpos[joint["qpos_adr"]] = joint["target"]
            data.qvel[joint["dof_adr"]] = 0.0
            joint["pid"].reset()
        mujoco.mj_forward(model, data)

    reset_pose()
    pressed_keys = []

    def on_key(keycode):
        pressed_keys.append(keycode)

    print(f"Loaded: {xml_path}")
    if path_target is not None:
        print_path_target("Initial path target:", current_row_index, path_target)
    print(f"Gravity: {model.opt.gravity}")
    print(
        f"Fluid: density={model.opt.density:g} kg/m^3, "
        f"viscosity={model.opt.viscosity:g} Pa*s"
    )
    print(f"PID: kp={KP}, ki={KI}, kd={KD}, torque_limit={TORQUE_LIMIT} Nm")
    print(
        f"Debug draw: world_axes={SHOW_WORLD_AXES}, "
        f"path_curve={SHOW_PATH_CURVE}, path_points={len(path_table)}, "
        "frame=right-handed Z-down"
    )
    print(
        f"Path stepping: update_interval={PATH_ROW_UPDATE_INTERVAL:g}s, "
        f"total_time={path_total_time:g}s, "
        f"kinematic_root_move={ENABLE_KINEMATIC_PATH_MOVE}, "
        "joints are PID-controlled"
    )
    for joint in joints:
        axis = model.jnt_axis[joint["joint_id"]]
        print(
            f"Joint {joint['name']}: target={np.rad2deg(joint['target']):.1f} deg, "
            f"axis(local)={axis}, qpos_adr={joint['qpos_adr']}, dof_adr={joint['dof_adr']}"
        )
    print("Controls: R reset, Q/Esc quit")

    with viewer.launch_passive(model, data, key_callback=on_key) as v:
        draw_debug_scene(v.user_scn, path_table)
        wall_start = time.perf_counter()
        t_print = 0.0
        episode_count = 1

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
                    episode_count = 1
                    print("[R] reset")
                elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    should_quit = True

            if should_quit:
                break

            if path_total_time > 0.0 and data.time >= path_total_time:
                episode_count += 1
                reset_pose()
                wall_start = time.perf_counter()
                t_print = 0.0
                print(f"[path complete] reset, episode={episode_count}")
                draw_debug_scene(v.user_scn, path_table)
                v.sync()
                continue

            next_row_index = path_row_index_from_time(
                data.time,
                PATH_ROW_UPDATE_INTERVAL,
                len(path_targets),
            )
            if next_row_index != current_row_index:
                current_row_index = next_row_index
                path_target = path_targets[current_row_index]
                set_joint_targets(path_target)
                for joint in joints:
                    joint["pid"].reset()
                print_path_target("[path]", current_row_index, path_target)

            if path_target is not None and ENABLE_KINEMATIC_PATH_MOVE:
                apply_root_pose(path_target)
                mujoco.mj_forward(model, data)

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
            if path_target is not None and ENABLE_KINEMATIC_PATH_MOVE:
                apply_root_pose(path_target)
                mujoco.mj_forward(model, data)
            draw_debug_scene(v.user_scn, path_table)

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
