"""Simulator integration checks for resets, observation contracts and finite rollouts."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.tasks.recovery.mdp.events import reset_fallen
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply


def main():
    cfg = parse_env_cfg("Isaac-Limx-WF-GetUp-Play-v0", device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 1.0  # exercise automatic reset / timeout for every stage
    env = gym.make("Isaac-Limx-WF-GetUp-Play-v0", cfg=cfg)
    try:
        raw = env.unwrapped
        for level in range(4):
            raw.task_state.level = level
            obs, _ = env.reset()
            assert obs["policy"].shape == (args.num_envs, 28)
            assert obs["obsHistory"].shape == (args.num_envs, 10, 28)
            assert torch.count_nonzero(obs["commands"]) == 0
            asset = raw.scene["robot"]
            leg_ids = raw._getup_leg_joint_ids
            limits = asset.data.joint_pos_limits[:, leg_ids]
            assert torch.isfinite(limits).all() and (limits.abs() < torch.pi).all()
            leg_action = raw.action_manager.get_term("joint_pos")
            leg_action.process_actions(torch.full((args.num_envs, 6), 1e6, device=raw.device))
            assert (leg_action.processed_actions <= asset.data.joint_pos_limits[:, leg_action._joint_ids, 1]).all()
            leg_action.process_actions(torch.full((args.num_envs, 6), -1e6, device=raw.device))
            assert (leg_action.processed_actions >= asset.data.joint_pos_limits[:, leg_action._joint_ids, 0]).all()
            root = asset.data.root_state_w.clone()
            corners = raw._getup_collision_corners
            rotation = root[:, None, 3:7].expand(-1, len(corners), -1).reshape(-1, 4)
            points = corners[None].expand(args.num_envs, -1, -1).reshape(-1, 3)
            minimum = quat_apply(rotation, points).reshape(args.num_envs, -1, 3)[:, :, 2].amin(dim=1)
            minimum += root[:, 2] - raw.scene.env_origins[:, 2]
            assert (minimum >= cfg.getup.reset_clearance - 1e-5).all(), minimum
            # A subset reset must leave every other physical state untouched.
            reset_fallen(raw, torch.tensor([0], device=raw.device))
            torch.testing.assert_close(asset.data.root_state_w[1:], root[1:])
            assert (raw.task_state.stage == level).all()
            assert not raw.task_state.success.any()
            terminations = 0
            for _ in range(60):
                actions = torch.randn(args.num_envs, 8, device=raw.device) * 0.15
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                terminations += int((terminated | truncated).sum())
            assert terminations >= args.num_envs, "Timeout / automatic reset did not run."
            print(f"[GetUp check] stage={level}: collision clearance, subset reset, observations, rollout passed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
