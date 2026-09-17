"""Wheel-legged recovery tasks; public Gym IDs are kept checkpoint-compatible."""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import (
    WF_TRON1AGetUpPPORunnerCfg,
    WF_TRON1AGetUpRecoveryPPORunnerCfg,
    WF_TRON1AGetUpAutoPPORunnerCfg,
    WF_TRON1AGetUpBoundedPPORunnerCfg,
)
from .getup_env_cfg import (
    WFGetUpEnvCfg, WFGetUpEnvCfg_PLAY,
    WFGetUpRecoveryEnvCfg, WFGetUpRecoveryEnvCfg_PLAY,
    WFGetUpAutoEnvCfg, WFGetUpAutoEnvCfg_PLAY,
)

for task_id, env_cfg in (
    ("Isaac-Limx-WF-GetUp-v0", WFGetUpEnvCfg),
    ("Isaac-Limx-WF-GetUp-Play-v0", WFGetUpEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": WF_TRON1AGetUpPPORunnerCfg},
    )

for task_id, env_cfg in (
    ("Isaac-Limx-WF-GetUp-Recovery-v0", WFGetUpRecoveryEnvCfg),
    ("Isaac-Limx-WF-GetUp-Recovery-Play-v0", WFGetUpRecoveryEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": WF_TRON1AGetUpRecoveryPPORunnerCfg},
    )

for task_id, env_cfg in (
    ("Isaac-Limx-WF-GetUp-Auto-v0", WFGetUpAutoEnvCfg),
    ("Isaac-Limx-WF-GetUp-Auto-Play-v0", WFGetUpAutoEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": WF_TRON1AGetUpAutoPPORunnerCfg},
    )

for task_id, env_cfg in (
    ("Isaac-Limx-WF-GetUp-Bounded-v0", WFGetUpAutoEnvCfg),
    ("Isaac-Limx-WF-GetUp-Bounded-Play-v0", WFGetUpAutoEnvCfg_PLAY),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": WF_TRON1AGetUpBoundedPPORunnerCfg},
    )
