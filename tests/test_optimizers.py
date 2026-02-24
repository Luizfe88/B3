import unittest
import numpy as np
import pandas as pd
import os
from importlib import import_module

opt = import_module("optimizer")


def make_df(n=600):
    rng = np.random.default_rng(42)
    prices = np.cumprod(1 + rng.normal(0, 0.002, size=n)) * 100
    df = pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.0005, size=n)),
            "high": prices * (1 + abs(rng.normal(0, 0.001, size=n))),
            "low": prices * (1 - abs(rng.normal(0, 0.001, size=n))),
            "close": prices,
            "tick_volume": rng.integers(100, 10000, size=n),
        }
    )
    return df


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def test_gd(self):
        res = opt.optimize_gd(
            self.df, opt.DEFAULT_PARAMS.copy(), steps=10, lr=1.0, seed=42
        )
        self.assertIn("score", res)
        self.assertTrue(np.isfinite(res["score"]))

    def test_ga(self):
        res = opt.optimize_ga(self.df, pop_size=10, generations=5, seed=42)
        self.assertIn("score", res)
        self.assertTrue(np.isfinite(res["score"]))

    def test_sa(self):
        res = opt.optimize_sa(self.df, opt.DEFAULT_PARAMS.copy(), iters=50, seed=42)
        self.assertIn("score", res)
        self.assertTrue(np.isfinite(res["score"]))

    def test_train_ml_model(self):
        out_dir = getattr(opt, "OPT_OUTPUT_DIR", "optimizer_output")
        os.makedirs(out_dir, exist_ok=True)
        res = opt.train_ml_model("TEST", self.df, base_dir=out_dir)
        self.assertIsNotNone(res)
        self.assertIn("cv_scores", res)


if __name__ == "__main__":
    unittest.main()
