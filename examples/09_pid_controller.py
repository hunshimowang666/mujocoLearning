"""
09_pid_controller.py
====================
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "3dModels", "simpleBox.xml")
    if not os.path.exists(xml_path):
        print(f"错误：找不到 {xml_path}")
        return

    # -- 加载模型（不修改 qpos0，保留 XML 默认位置 0,0,1） --
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

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
    print(f"PID 参数  : Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")
    print("=" * 60)
    print("  [A] 向下冲击  [Z] 向上冲击  [R] 重置  [Q] 退出")
    print("=" * 60)

    try:
        from mujoco import viewer

        # ---- 通过 MuJoCo Viewer 回调捕获按键（msvcrt 只监听控制台，不适用） ----
        pressed_keys = []

        def on_key(keycode: int):
            pressed_keys.append(keycode)

        with viewer.launch_passive(model, data, key_callback=on_key) as v:
            t_print    = 0.0
            wall_start = time.perf_counter()
            step_cnt   = 0
            force      = 0.0
            p_part = i_part = d_part = 0.0

            # GLFW 键码常量
            K_A, K_Z, K_R, K_Q = 65, 90, 82, 81

            while v.is_running():
                # ---- 实时同步 ----
                wall_elapsed = time.perf_counter() - wall_start
                if data.time > wall_elapsed:
                    time.sleep(data.time - wall_elapsed)

                z   = data.body("box").xpos[2]
                vz  = data.qvel[2]
                err = target_z - z

                # ---- PID 仅在控制周期到来时更新 ----
                if step_cnt % steps_per_ctrl == 0:
                    force, p_part, i_part, d_part = pid.compute(err, control_dt)

                # ---- 处理 Viewer 按键 ----
                should_quit = False
                while pressed_keys:
                    k = pressed_keys.pop(0)
                    if k == K_A:
                        data.qvel[2] -= 8.0
                        print(f"[A] vz={data.qvel[2]:+.1f}")
                    elif k == K_Z:
                        data.qvel[2] += 8.0
                        print(f"[Z] vz={data.qvel[2]:+.1f}")
                    elif k == K_R:
                        mujoco.mj_resetData(model, data)
                        pid = make_pid()
                        wall_start = time.perf_counter()
                        t_print    = 0.0
                        step_cnt   = 0
                        force = p_part = i_part = d_part = 0.0
                        print("[R] 环境 & 控制器已重新加载")
                    elif k == K_Q:
                        print("[Q] 退出")
                        should_quit = True
                if should_quit:
                    break

                # ---- 施加力（作用在质心，避免扭矩导致乱转） ----
                f_world  = np.array([0.0, 0.0, force], dtype=np.float64)
                qfrc     = np.zeros(model.nv, dtype=np.float64)
                com_pos  = data.xipos[box_id]          # 质心世界坐标
                mujoco.mj_applyFT(model, data,
                                  f_world, np.zeros(3),
                                  com_pos, box_id, qfrc)
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

                v.sync()

    except ImportError:
        print("mujoco viewer 不可用")


if __name__ == "__main__":
    main()
