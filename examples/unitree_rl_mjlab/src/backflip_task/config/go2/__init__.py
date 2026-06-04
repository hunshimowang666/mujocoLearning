"""Standalone Go2 backflip task config.

This package intentionally avoids the global mjlab task registry because importing
``mjlab.tasks`` auto-scans many installed tasks. The backflip train/play scripts
load these config factories directly.
"""

from .env_cfgs import TASK_ID, unitree_go2_backflip_env_cfg
from .rl_cfg import unitree_go2_backflip_ppo_runner_cfg

__all__ = [
  "TASK_ID",
  "unitree_go2_backflip_env_cfg",
  "unitree_go2_backflip_ppo_runner_cfg",
]
