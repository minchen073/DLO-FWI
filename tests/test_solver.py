"""Algorithm checks use a small differentiable operator and frozen toy prior."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlo import METHODS, invert
from dlo.prior import DiffusionPrior
from dlo.metrics import evaluate


class ToyPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return SimpleNamespace(sample=0.1 * torch.tanh(x))


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / "configs/openfwi.json").read_text())
        self.initial = torch.linspace(2300, 3400, 144).reshape(12, 12)
        self.forward = lambda v: ((v - 3000) / 1500).square()
        self.obs = self.forward(self.initial + 100)
        self.prior = DiffusionPrior(ToyPrior().eval(), torch.cumprod(1-torch.linspace(0.0001, 0.02, 1000), 0))

    def run_method(self, method, truth=None, reuse=True):
        params = {**self.config["methods"][method], "n_iters": 4}
        torch.manual_seed(42)
        return invert(method, self.obs, self.initial, self.forward, params,
                      prior=self.prior, velocity_true=truth, reuse_generator=reuse)

    def test_reference_is_evaluation_only(self):
        for method in METHODS:
            with self.subTest(method=method):
                a = self.run_method(method)
                b = self.run_method(method, self.initial + 500)
                np.testing.assert_array_equal(a["velocity_pred"], b["velocity_pred"])
                self.assertNotIn("mae", a["history"][-1])
                self.assertIn("mae", b["history"][-1])

    def test_cached_generator_matches_recomputed_generator(self):
        a = self.run_method("dlo", reuse=True)
        cached_calls = self.prior.wrapper.calls
        b = self.run_method("dlo", reuse=False)
        self.assertEqual(cached_calls, 4 * 3)
        self.assertEqual(self.prior.wrapper.calls - cached_calls, 4 * 6)
        np.testing.assert_array_equal(a["velocity_pred"], b["velocity_pred"])
        self.assertEqual(a["history"], b["history"])

    def test_metrics_identity(self):
        x = torch.linspace(-1, 1, 144).reshape(12, 12)
        metrics = evaluate(x, x)
        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["rmse"], 0)
        self.assertAlmostEqual(metrics["ssim"], 1, places=6)

    def test_initial_state_and_finite_outputs(self):
        initial = self.initial.clone()
        for method in METHODS:
            result = self.run_method(method)
            self.assertEqual(len(result["history"]), 5)
            self.assertTrue(np.isfinite(result["velocity_pred"]).all())
            self.assertFalse(np.array_equal(result["velocity_pred"], initial.numpy()))
        torch.testing.assert_close(self.initial, initial, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
