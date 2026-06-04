"""Train Unitree Go2 backflip with mjlab GPU-parallel simulation."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


DIRECT_TASK = "Unitree-Go2-Backflip"
DIRECT_EXPERIMENT_NAME = "go2_backflip_mjlab"
DIRECT_NUM_ENVS = 4096
DIRECT_MAX_ITERATIONS = 3000
DIRECT_GPU_IDS: list[int] | Literal["all"] | None = [0]
DIRECT_RESUME_LATEST = True
DIRECT_DELETE_OLD_NETWORKS = True


def configure_runtime_environment() -> None:
  os.environ.pop("PYTHONNOUSERSITE", None)
  os.environ.setdefault("WANDB_MODE", "disabled")
  os.environ.setdefault("MUJOCO_GL", "egl")
  wsl_lib = "/usr/lib/wsl/lib"
  if Path(wsl_lib).exists():
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if wsl_lib not in ld_library_path.split(":"):
      os.environ["LD_LIBRARY_PATH"] = (
        wsl_lib if not ld_library_path else f"{wsl_lib}:{ld_library_path}"
      )


def find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = Path("logs") / "rsl_rl" / experiment_name
  checkpoints = sorted(
    log_root.glob("*/model_*.pt"),
    key=lambda path: (path.parent.name, int(path.stem.split("_")[-1])),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return checkpoints[-1]


@dataclass(frozen=True)
class Go2BackflipMjlabTrainConfig:
  task: str = DIRECT_TASK
  num_envs: int | None = DIRECT_NUM_ENVS
  max_iterations: int | None = DIRECT_MAX_ITERATIONS
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  delete_old_networks: bool = DIRECT_DELETE_OLD_NETWORKS
  resume_latest: bool = DIRECT_RESUME_LATEST
  resume_run: str | None = None
  resume_checkpoint: str = "model_.*.pt"


def direct_train_config() -> Go2BackflipMjlabTrainConfig:
  return Go2BackflipMjlabTrainConfig()


def _select_device(gpu_ids: list[int] | Literal["all"] | None) -> str:
  if gpu_ids is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu"
  if gpu_ids == "all":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    return "cuda:0"
  if len(gpu_ids) == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu"
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
  os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
  return "cuda:0"


def install_compact_training_console(runner) -> None:
  original_log = runner.logger.log

  def log_compact(*args, **kwargs):
    it = kwargs.get("it", args[0] if len(args) > 0 else None)
    total_it = kwargs.get("total_it", args[2] if len(args) > 2 else None)
    collect_time = kwargs.get("collect_time", args[3] if len(args) > 3 else None)
    learn_time = kwargs.get("learn_time", args[4] if len(args) > 4 else None)
    with contextlib.redirect_stdout(io.StringIO()):
      original_log(*args, **kwargs)

    if it is None or total_it is None or collect_time is None or learn_time is None:
      return

    iteration_time = collect_time + learn_time
    collection_size = runner.logger.cfg["num_steps_per_env"] * runner.logger.num_envs
    steps_per_second = int(collection_size / iteration_time) if iteration_time > 0.0 else 0
    mean_reward = (
      statistics.mean(runner.logger.rewbuffer)
      if len(runner.logger.rewbuffer) > 0
      else None
    )
    mean_episode_length = (
      statistics.mean(runner.logger.lenbuffer)
      if len(runner.logger.lenbuffer) > 0
      else None
    )
    reward_text = f"{mean_reward:.2f}" if mean_reward is not None else "n/a"
    length_text = (
      f"{mean_episode_length:.2f}" if mean_episode_length is not None else "n/a"
    )
    print(
      f"iter {it}/{total_it} | "
      f"steps/s {steps_per_second} | "
      f"mean reward {reward_text} | "
      f"mean episode length {length_text}",
      flush=True,
    )

  runner.logger.log = log_compact
  print("[INFO] Compact training console enabled")


def train(args: Go2BackflipMjlabTrainConfig) -> None:
  if args.task != DIRECT_TASK:
    raise ValueError(f"This script only supports {DIRECT_TASK}.")

  from src.backflip_task.config.go2 import (
    unitree_go2_backflip_env_cfg,
    unitree_go2_backflip_ppo_runner_cfg,
  )

  env_cfg = unitree_go2_backflip_env_cfg()
  agent_cfg = unitree_go2_backflip_ppo_runner_cfg()
  agent_cfg.experiment_name = DIRECT_EXPERIMENT_NAME

  if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs

  if args.resume_latest and args.resume_run is not None:
    raise ValueError("Use either --resume-latest or --resume-run, not both.")

  resume = args.resume_latest or args.resume_run is not None
  resume_path: Path | None = None
  if resume:
    agent_cfg.resume = True
    if args.resume_latest:
      resume_path = find_latest_checkpoint(agent_cfg.experiment_name)
      agent_cfg.load_run = resume_path.parent.name
      agent_cfg.load_checkpoint = resume_path.name
    else:
      agent_cfg.load_run = args.resume_run
      agent_cfg.load_checkpoint = args.resume_checkpoint
      from mjlab.utils.os import get_checkpoint_path

      resume_path = get_checkpoint_path(
        Path("logs") / "rsl_rl" / agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
      )
    print(
      "[INFO] Resuming backflip mjlab training from "
      f"run={agent_cfg.load_run}, checkpoint={agent_cfg.load_checkpoint}"
    )

  log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  if args.delete_old_networks and not resume and log_root_path.exists():
    print(f"[INFO] Removing old training logs: {log_root_path}")
    shutil.rmtree(log_root_path)

  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_dir_name += f"_{agent_cfg.run_name}"
  log_dir = log_root_path / log_dir_name

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  device = _select_device(args.gpu_ids)
  configure_torch_backends()
  env_cfg.seed = agent_cfg.seed
  print(f"[INFO] Training with: device={device}, seed={agent_cfg.seed}")
  print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  runner.add_git_repo_to_log(__file__)
  install_compact_training_console(runner)

  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  num_learning_iterations = agent_cfg.max_iterations
  if resume_path is not None:
    num_learning_iterations = max(
      agent_cfg.max_iterations - runner.current_learning_iteration,
      0,
    )
    print(
      "[INFO] Resume target: "
      f"current_iteration={runner.current_learning_iteration}, "
      f"target_iteration={agent_cfg.max_iterations}, "
      f"remaining_iterations={num_learning_iterations}"
    )

  runner.learn(
    num_learning_iterations=num_learning_iterations,
    init_at_random_ep_len=True,
  )
  env.close()


def main() -> None:
  configure_runtime_environment()

  if len(sys.argv) == 1:
    args = direct_train_config()
    print("[INFO] Running train_go2_backflip_mjlab.py with direct-run settings")
  else:
    args = tyro.cli(Go2BackflipMjlabTrainConfig)
  os.chdir(PROJECT_ROOT)
  train(args)


if __name__ == "__main__":
  main()
