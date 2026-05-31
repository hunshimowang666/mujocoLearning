"""Run a MuJoCo Menagerie Franka Panda arm simulation.

Headless smoke test:
  python examples/11_panda_arm_sim.py

Interactive viewer:
  python examples/11_panda_arm_sim.py --viewer
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "examples" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"


def make_arm_target(t: float, home_ctrl: np.ndarray) -> np.ndarray:
  target = home_ctrl.copy()
  target[0] = 0.45 * math.sin(0.7 * t)
  target[1] = 0.35 * math.sin(0.5 * t)
  target[2] = 0.35 * math.sin(0.9 * t)
  target[3] = -1.65 + 0.25 * math.sin(0.6 * t)
  target[4] = 0.25 * math.sin(0.8 * t)
  target[5] = 1.55 + 0.2 * math.sin(0.4 * t)
  target[6] = -0.8 + 0.25 * math.sin(0.7 * t)
  target[7] = 180.0 + 60.0 * math.sin(1.2 * t)
  return target


def step_controller(model: mujoco.MjModel, data: mujoco.MjData, home_ctrl: np.ndarray) -> None:
  data.ctrl[:] = make_arm_target(data.time, home_ctrl)
  mujoco.mj_step(model, data)


def run_headless(
  model: mujoco.MjModel, data: mujoco.MjData, duration: float, home_ctrl: np.ndarray
) -> None:
  hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
  steps = int(duration / model.opt.timestep)
  for _ in range(steps):
    step_controller(model, data, home_ctrl)
  hand_pos = data.xpos[hand_id]
  print(
    "Panda simulation OK | "
    f"time={data.time:.2f}s | "
    f"hand_pos=({hand_pos[0]:+.3f}, {hand_pos[1]:+.3f}, {hand_pos[2]:+.3f})"
  )


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, duration: float) -> None:
  import mujoco.viewer

  home_ctrl = data.ctrl.copy()
  end_time = data.time + duration if duration > 0 else math.inf
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < end_time:
      step_controller(model, data, home_ctrl)
      viewer.sync()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer.")
  parser.add_argument("--duration", type=float, default=5.0, help="Simulation seconds.")
  args = parser.parse_args()

  if not MODEL_XML.exists():
    raise FileNotFoundError(f"Missing model XML: {MODEL_XML}")

  model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  home_ctrl = data.ctrl.copy()

  if args.viewer:
    run_viewer(model, data, args.duration)
  else:
    run_headless(model, data, args.duration, home_ctrl)


if __name__ == "__main__":
  main()
