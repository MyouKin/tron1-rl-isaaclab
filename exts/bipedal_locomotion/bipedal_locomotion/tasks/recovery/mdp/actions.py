"""Position action with a final clamp to the articulation's physical joint limits."""

from isaaclab.envs.mdp.actions import JointPositionAction


class LimitedLegPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])
