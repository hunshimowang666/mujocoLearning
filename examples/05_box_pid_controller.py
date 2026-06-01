"""
05_box_pid_controller.py
========================
纯 PID 控制器：在重力场中将立方体控制在目标高度 0.5m

无重力前馈 —— 靠积分项逐步"学习"抵消重力所需的上推力。
运行时可以看到：
  1. 物体从 0.5m 开始被重力拉下
  2. 误差增大，P项立即响应，I项缓慢积累
  3. 积分力追上重力，物体回升
  4. 短暂震荡后稳定悬停在 0.5m
"""

import numpy as np
import os
import time


BUTTON_LEFT = False
BUTTON_MIDDLE = False
BUTTON_RIGHT = False
LAST_X = 0.0
LAST_Y = 0.0
SHOW_COORDINATE_FRAMES = True
USE_PID_CONTROL = True


class PIDController:
    """位置式 PID，带积分限幅和输出限幅"""

    def __init__(self, kp: float, ki: float, kd: float, out_max: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_max = out_max

        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def compute(self, error: float, dt: float):
        if dt <= 0:
            dt = 0.001

        p = self.kp * error

        # 积分 + 限幅
        self._integral += error * dt
        self._integral = np.clip(self._integral, -20.0, 20.0)
        i = self.ki * self._integral

        # 微分（首次跳过）
        if self._first:
            d = 0.0
            self._first = False
        else:
            d = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        out = p + i + d

        if self.out_max is not None:
            out = np.clip(out, -self.out_max, self.out_max)

        return out, p, i, d          # 返回各分量便于调试


def main():
    import mujoco
    from mujoco.glfw import glfw

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "3dModels", "simpleBox.xml")
    if not os.path.exists(xml_path):
        print(f"错误：找不到 {xml_path}")
        return

    # -- 加载模型（不修改 qpos0，保留 XML 默认位置 0,0,1） --
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    mass   = model.body_mass[box_id]
    g      = abs(model.opt.gravity[2])

    # -- 控制参数 --
    target_z      = 0.5
    control_rate  = 50            # Hz
    physics_dt    = model.opt.timestep
    control_dt    = 1.0 / control_rate
    steps_per_ctrl = int(control_dt / physics_dt)

    # 创建控制器的工厂函数（reset 时完整重建）
    def make_pid():
        return PIDController(kp=80.0, ki=35.0, kd=12.0, out_max=50.0)

    pid = make_pid()

    print(f"目标高度  : {target_z} m")
    print(f"物体质量  : {mass} kg,  重力 g = {g} m/s²")
    print(f"平衡重力需: {mass * g:.0f} N  （全部由积分项提供）")
    print(f"物理步长  : {physics_dt*1000:.0f} ms ({1/physics_dt:.0f} Hz)")
    print(f"控制频率  : {control_rate} Hz  (每 {steps_per_ctrl} 个物理步更新一次)")
    print(f"PID 控制  : {'开启' if USE_PID_CONTROL else '关闭'}")
    if USE_PID_CONTROL:
        print(f"PID 参数  : Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")
    print("=" * 60)
    print("  [A] 向下冲击  [Z] 向上冲击  [R] 重置  [C] 坐标显示  [Q/Esc] 退出")
    print("  鼠标拖拽/滚轮控制主视角，右下角显示 box_camera 画面")
    print("=" * 60)

    if not glfw.init():
        raise RuntimeError("无法初始化 GLFW")

    window = glfw.create_window(1280, 800, "MuJoCo PID controller with camera PIP", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("无法创建 GLFW 窗口")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    main_cam = mujoco.MjvCamera()
    pip_cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    show_coordinate_frames = SHOW_COORDINATE_FRAMES

    def apply_visual_options():
        # 自己手动画 STL/body 坐标轴；MuJoCo 内置 frame 可能显示编译后的辅助坐标。
        opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = int(show_coordinate_frames)
        opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = int(show_coordinate_frames)

    def add_axes_to_scene(origin, rotation, length):
        if not show_coordinate_frames or scene.ngeom + 3 > scene.maxgeom:
            return

        origin = np.asarray(origin, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        identity = np.eye(3, dtype=np.float64).reshape(-1)
        size = np.array([0.02, 0.02, 0.08], dtype=np.float64)
        axes = (
            (rotation[:, 0], np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)),
            (rotation[:, 1], np.array([0.0, 0.8, 0.0, 1.0], dtype=np.float32)),
            (rotation[:, 2], np.array([0.1, 0.3, 1.0, 1.0], dtype=np.float32)),
        )

        for direction, rgba in axes:
            geom = scene.geoms[scene.ngeom]
            endpoint = origin + length * direction
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                size,
                origin,
                identity,
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.025,
                origin,
                endpoint,
            )
            scene.ngeom += 1

    def add_coordinate_frames_to_scene():
        add_axes_to_scene(np.zeros(3), np.eye(3), 0.6)
        add_axes_to_scene(data.xpos[box_id], data.xmat[box_id], 0.35)

    apply_visual_options()

    mujoco.mjv_defaultCamera(main_cam)
    main_cam.distance = 2.0
    main_cam.elevation = -25.0
    main_cam.azimuth = 135.0
    main_cam.lookat[:] = [0.0, 0.0, target_z]

    camera_name = "box_camera"
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    show_pip = camera_id >= 0
    if show_pip:
        mujoco.mjv_defaultCamera(pip_cam)
        pip_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        pip_cam.fixedcamid = camera_id
    else:
        print(f"[Camera] 未找到 {camera_name}，右下角画面关闭")

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

        width, height = glfw.get_window_size(window)
        del width
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
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, main_cam
        )

    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)

    t_print = 0.0
    wall_start = time.perf_counter()
    step_cnt = 0
    force = 0.0
    p_part = i_part = d_part = 0.0

    try:
        while not glfw.window_should_close(window):
            # ---- 实时同步 ----
            wall_elapsed = time.perf_counter() - wall_start
            if data.time > wall_elapsed:
                time.sleep(data.time - wall_elapsed)

            # ---- 处理按键 ----
            while pressed_keys:
                k = pressed_keys.pop(0)
                if k == glfw.KEY_A:
                    data.qvel[2] -= 8.0
                    print(f"[A] vz={data.qvel[2]:+.1f}")
                elif k == glfw.KEY_Z:
                    data.qvel[2] += 8.0
                    print(f"[Z] vz={data.qvel[2]:+.1f}")
                elif k == glfw.KEY_R:
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    pid = make_pid()
                    wall_start = time.perf_counter()
                    t_print = 0.0
                    step_cnt = 0
                    force = p_part = i_part = d_part = 0.0
                    print("[R] 环境 & 控制器已重新加载")
                elif k == glfw.KEY_C:
                    show_coordinate_frames = not show_coordinate_frames
                    apply_visual_options()
                    state = "开启" if show_coordinate_frames else "关闭"
                    print(f"[C] 坐标/惯性显示已{state}")
                elif k == glfw.KEY_Q:
                    print("[Q] 退出")
                    glfw.set_window_should_close(window, True)

            z = data.body("box").xpos[2]
            vz = data.qvel[2]
            err = target_z - z

            # ---- PID 仅在控制周期到来时更新 ----
            if USE_PID_CONTROL and step_cnt % steps_per_ctrl == 0:
                force, p_part, i_part, d_part = pid.compute(err, control_dt)
            elif not USE_PID_CONTROL:
                force = p_part = i_part = d_part = 0.0

            # ---- 施加力（作用在质心，避免扭矩导致乱转） ----
            f_world = np.array([0.0, 0.0, force], dtype=np.float64)
            qfrc = np.zeros(model.nv, dtype=np.float64)
            com_pos = data.xipos[box_id]          # 质心世界坐标
            mujoco.mj_applyFT(model, data, f_world, np.zeros(3), com_pos, box_id, qfrc)
            data.qfrc_applied[:] = qfrc

            mujoco.mj_step(model, data)
            step_cnt += 1

            # ---- 打印 ----
            if data.time - t_print >= 1.0:
                print(f"t={data.time:4.1f}s | z={z:.3f}m | vz={vz:+6.2f} | "
                      f"err={err:+6.3f}m | "
                      f"P={p_part:+6.1f} I={i_part:+6.1f} D={d_part:+6.1f} | "
                      f"F={force:+6.1f}N")
                t_print = data.time

            # ---- 主视角 ----
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)
            mujoco.mjv_updateScene(
                model, data, opt, None, main_cam, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            add_coordinate_frames_to_scene()
            mujoco.mjr_render(viewport, scene, context)

            # ---- 右下角相机画面 ----
            if show_pip:
                margin = max(12, fb_width // 100)
                pip_width = max(260, fb_width // 4)
                pip_height = max(180, fb_height // 4)
                pip_x = fb_width - pip_width - margin
                pip_y = margin

                if hasattr(mujoco, "mjr_rectangle"):
                    border = mujoco.MjrRect(
                        pip_x - 4, pip_y - 4, pip_width + 8, pip_height + 8
                    )
                    mujoco.mjr_rectangle(border, 0.02, 0.02, 0.02, 1.0)

                pip_rect = mujoco.MjrRect(pip_x, pip_y, pip_width, pip_height)
                mujoco.mjv_updateScene(
                    model, data, opt, None, pip_cam, mujoco.mjtCatBit.mjCAT_ALL, scene
                )
                add_coordinate_frames_to_scene()
                mujoco.mjr_render(pip_rect, scene, context)
                if hasattr(mujoco, "mjr_overlay"):
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        pip_rect,
                        camera_name,
                        "",
                        context,
                    )

            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()
