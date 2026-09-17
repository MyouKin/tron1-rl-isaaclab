"""Check real bound methods without constructing the simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


loss = load_method("rsl_rl/rsl_rl/algorithm/ppo.py", "PPO", "action_bound_loss")
resolve = load_method("exts/bipedal_locomotion/bipedal_locomotion/utils/wrappers/rsl_rl/vecenv_wrapper.py",
                      "RslRlVecEnvWrapper", "get_action_mean_bounds")


class TestActionBounds(unittest.TestCase):
    def test_gradient_brings_saturated_mean_back_without_touching_legal_mean(self):
        state = SimpleNamespace(action_mean_bounds=torch.tensor([[-1., 1.]] * 3))
        mean = torch.tensor([[-25., 0., 23.]], requires_grad=True)
        loss(state, mean).backward()
        self.assertLess(mean.grad[0, 0], 0)
        self.assertEqual(mean.grad[0, 1], 0)
        self.assertGreater(mean.grad[0, 2], 0)
        self.assertEqual(float(loss(state, torch.zeros(1, 3))), 0.)
        self.assertEqual(float(loss(SimpleNamespace(action_mean_bounds=None), mean)), 0.)

    def test_action_bounds_account_for_offset_scale_and_actual_joint_order(self):
        position = SimpleNamespace(_clip=torch.tensor([[[-2., 2.], [-3., 3.]]]),
                                   _scale=2., _offset=torch.tensor([[0.5, -0.5]]),
                                   _joint_ids=[1, 0],
                                   _asset=SimpleNamespace(data=SimpleNamespace(
                                       joint_pos_limits=torch.tensor([[[-1., 1.], [-0.5, 0.5]]]))))
        velocity = SimpleNamespace(_clip=torch.tensor([[[-15., 15.]]]), _scale=3., _offset=0.)
        terms = {'joint_pos': position, 'joint_vel': velocity}
        env = SimpleNamespace(cfg=SimpleNamespace(getup=True), device='cpu',
                              action_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.get))
        actual = resolve(SimpleNamespace(unwrapped=env))
        torch.testing.assert_close(torch.tensor(actual), torch.tensor([[-0.5, 0.], [-0.25, 0.75], [-5., 5.]]))


if __name__ == '__main__':
    unittest.main()
