"""Gymnasium Humanoid-v5 MJLab task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .config.env_cfgs import humanoid_v5_flat_env_cfg
from .config.rl_cfg import humanoid_v5_ppo_runner_cfg


register_mjlab_task(
  task_id="Gym-HumanoidV5-Flat",
  env_cfg=humanoid_v5_flat_env_cfg(),
  play_env_cfg=humanoid_v5_flat_env_cfg(play=True),
  rl_cfg=humanoid_v5_ppo_runner_cfg(),
)
