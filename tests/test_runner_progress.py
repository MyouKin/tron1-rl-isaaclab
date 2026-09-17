"""Exercise the real logger without importing the simulator or constructing PPO."""

import ast
import contextlib
import io
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


source = Path(__file__).resolve().parents[1] / "rsl_rl/rsl_rl/runner/on_policy_runner.py"
runner = next(node for node in ast.parse(source.read_text()).body
              if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunner")
log_method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "log")
namespace = {"torch": torch, "statistics": statistics}
exec(compile(ast.Module(body=[log_method], type_ignores=[]), str(source), "exec"), namespace)


class TestRunnerProgress(unittest.TestCase):
    def check_progress(self, start, count, prior_time=0.0):
        state = SimpleNamespace(
            num_steps_per_env=24, env=SimpleNamespace(num_envs=4096),
            tot_timesteps=0, tot_time=prior_time, writer=Mock(),
            alg=SimpleNamespace(actor_critic=SimpleNamespace(logstd=torch.zeros(8)), learning_rate=1e-4),
        )
        for offset in range(count):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                namespace["log"](state, dict(
                    collection_time=0.6, learn_time=0.4, ep_infos=[],
                    mean_value_loss=0.0, mean_extra_loss=0.0, mean_surrogate_loss=0.0,
                    mean_kl=0.0, rewbuffer=[], it=start + offset,
                    start_iteration=start, start_total_time=prior_time, tot_iter=start + count,
                ))
            rendered = output.getvalue()
            self.assertIn(f"Learning iteration {start + offset + 1}/{start + count}", rendered)
            eta = rendered.split("ETA:")[1].strip()
            self.assertEqual(eta, f"{count - offset - 1:.1f}s")
        self.assertEqual(state.tot_timesteps, count * 24 * 4096)

    def test_fresh_training(self):
        self.check_progress(0, 3)

    def test_resumed_training(self):
        self.check_progress(6800, 500)

    def test_repeated_learn_excludes_previous_elapsed_time(self):
        self.check_progress(7300, 3, prior_time=600.0)


if __name__ == "__main__":
    unittest.main()
