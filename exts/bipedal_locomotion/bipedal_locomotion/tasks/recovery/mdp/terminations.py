"""Ground contact is expected during recovery and is never a failure by itself."""

import math

import torch

from .rewards import stable_mask
from .state import get_state
from .limits import outside_joint_limits


def sustained_success(env, wheel_cfg, body_cfg, leg_cfg):
    state = get_state(env)
    if state.last_step != env.common_step_counter:
        state.steps += 1
        stable = stable_mask(env, wheel_cfg, body_cfg, leg_cfg)
        stable &= ~outside_recovery_area(env)
        stable &= ~leg_limit_violation(env, leg_cfg)
        stable &= state.steps * env.step_dt >= env.cfg.getup.min_episode_time
        state.hold.copy_(torch.where(stable, state.hold + 1, 0))
        state.success.copy_(state.hold >= math.ceil(env.cfg.getup.hold_time / env.step_dt))
        state.last_step = env.common_step_counter
    return state.success


def outside_recovery_area(env):
    data = env.scene["robot"].data
    drift = (data.root_pos_w[:, :2] - get_state(env).start_xy).norm(dim=1)
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (drift > env.cfg.getup.max_drift) | (height < -0.25) | (height > 2.0)


def leg_limit_violation(env, leg_cfg):
    data = env.scene[leg_cfg.name].data
    return outside_joint_limits(data.joint_pos[:, leg_cfg.joint_ids],
                                data.joint_pos_limits[:, leg_cfg.joint_ids],
                                env.cfg.getup.joint_limit_tolerance)
