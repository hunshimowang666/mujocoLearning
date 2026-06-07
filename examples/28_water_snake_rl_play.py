"""Play the mjlab-trained water-snake policy."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "unitree_rl_mjlab"
sys.path.insert(0, str(PROJECT_ROOT))

DIRECT_EXPERIMENT_NAME = "water_snake_path_mjlab"
DIRECT_LATEST_CHECKPOINT = True
DIRECT_CHECKPOINT_FILE: str | None = None
DIRECT_DEVICE = "cuda:0"
DIRECT_VIEWER: Literal["native", "viser"] = "native"
DIRECT_RENDER_FRAME_RATE = 30.0
DIRECT_DRAW_PATH = True


def configure_runtime_environment() -> None:
  os.environ.pop("PYTHONNOUSERSITE", None)
  os.environ.setdefault("WANDB_MODE", "disabled")
  os.environ.setdefault("MUJOCO_GL", "glfw")
  wsl_lib = "/usr/lib/wsl/lib"
  if Path(wsl_lib).exists():
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if wsl_lib not in ld_library_path.split(":"):
      os.environ["LD_LIBRARY_PATH"] = (
        wsl_lib if not ld_library_path else f"{wsl_lib}:{ld_library_path}"
      )


def find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name
  checkpoints = sorted(
    log_root.glob("*/model_*.pt"),
    key=lambda path: (path.parent.name, int(path.stem.split("_")[-1])),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return checkpoints[-1]


@dataclass(frozen=True)
class WaterSnakeMjlabPlayConfig:
  checkpoint_file: str | None = DIRECT_CHECKPOINT_FILE
  latest_checkpoint: bool = DIRECT_LATEST_CHECKPOINT
  device: str = DIRECT_DEVICE
  viewer: Literal["native", "viser"] = DIRECT_VIEWER
  render_frame_rate: float = DIRECT_RENDER_FRAME_RATE
  draw_path: bool = DIRECT_DRAW_PATH


def direct_play_config() -> WaterSnakeMjlabPlayConfig:
  return WaterSnakeMjlabPlayConfig()


def play(cfg: WaterSnakeMjlabPlayConfig) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  from src.water_snake_task.config import (
    water_snake_path_env_cfg,
    water_snake_path_ppo_runner_cfg,
  )

  configure_torch_backends()

  checkpoint = Path(cfg.checkpoint_file) if cfg.checkpoint_file else None
  if checkpoint is None and cfg.latest_checkpoint:
    checkpoint = find_latest_checkpoint(DIRECT_EXPERIMENT_NAME)
  if checkpoint is None:
    raise ValueError("Set checkpoint_file or latest_checkpoint=True.")
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  env_cfg = water_snake_path_env_cfg(play=True, draw_path=cfg.draw_path)
  env_cfg.scene.num_envs = 1
  agent_cfg = water_snake_path_ppo_runner_cfg()

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=cfg.device,
  )
  policy = runner.get_inference_policy(device=cfg.device)

  print(f"Loaded checkpoint: {checkpoint}")
  print("Action space: 8 thruster forces + 2 joint angle targets; no joint PID target tracking.")
  print(f"Path drawing: {'enabled' if cfg.draw_path else 'disabled'}")

  if cfg.viewer == "native":
    viewer = NativeMujocoViewer(env, policy, frame_rate=cfg.render_frame_rate)
  else:
    viewer = ViserPlayViewer(env, policy, frame_rate=cfg.render_frame_rate)
  viewer.run()
  env.close()


def main() -> None:
  configure_runtime_environment()
  os.chdir(PROJECT_ROOT)
  if len(sys.argv) == 1:
    cfg = direct_play_config()
    print("[INFO] Running 28_water_snake_rl_play.py with direct-run settings")
  else:
    cfg = tyro.cli(WaterSnakeMjlabPlayConfig)
  play(cfg)


if __name__ == "__main__":
  main()
