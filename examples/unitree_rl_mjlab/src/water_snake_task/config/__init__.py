"""Water snake path-following task config."""

from .env_cfgs import TASK_ID, water_snake_path_env_cfg
from .rl_cfg import water_snake_path_ppo_runner_cfg

__all__ = [
  "TASK_ID",
  "water_snake_path_env_cfg",
  "water_snake_path_ppo_runner_cfg",
]
