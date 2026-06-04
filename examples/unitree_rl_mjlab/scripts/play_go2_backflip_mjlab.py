"""Play the mjlab-trained Go2 backflip policy in the native MuJoCo viewer."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from train_go2_backflip_mjlab import DIRECT_EXPERIMENT_NAME, DIRECT_TASK


DIRECT_PLAY_TASK = DIRECT_TASK
DIRECT_PLAY_EXPERIMENT_NAME = DIRECT_EXPERIMENT_NAME
DIRECT_PLAY_LATEST_CHECKPOINT = True
DIRECT_PLAY_CHECKPOINT_FILE: str | None = None
DIRECT_PLAY_DEVICE = "cuda:0"
DIRECT_PLAY_VIEWER: Literal["native", "viser"] = "native"
DIRECT_RENDER_FRAME_RATE = 30.0


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
class PlayBackflipMjlabConfig:
  checkpoint_file: str | None = DIRECT_PLAY_CHECKPOINT_FILE
  latest_checkpoint: bool = DIRECT_PLAY_LATEST_CHECKPOINT
  device: str = DIRECT_PLAY_DEVICE
  viewer: Literal["native", "viser"] = DIRECT_PLAY_VIEWER
  render_frame_rate: float = DIRECT_RENDER_FRAME_RATE


def direct_play_config() -> PlayBackflipMjlabConfig:
  return PlayBackflipMjlabConfig()


def play(cfg: PlayBackflipMjlabConfig) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  from src.backflip_task.config.go2 import (
    unitree_go2_backflip_env_cfg,
    unitree_go2_backflip_ppo_runner_cfg,
  )

  configure_torch_backends()

  checkpoint = Path(cfg.checkpoint_file) if cfg.checkpoint_file else None
  if checkpoint is None and cfg.latest_checkpoint:
    checkpoint = find_latest_checkpoint(DIRECT_PLAY_EXPERIMENT_NAME)
  if checkpoint is None:
    raise ValueError("Set checkpoint_file or latest_checkpoint=True.")
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  env_cfg = unitree_go2_backflip_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = unitree_go2_backflip_ppo_runner_cfg()

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
  print("Viewer controls are the standard mjlab/MuJoCo controls.")

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
    print("[INFO] Running play_go2_backflip_mjlab.py with direct-run settings")
  else:
    cfg = tyro.cli(PlayBackflipMjlabConfig)
  play(cfg)


if __name__ == "__main__":
  main()
