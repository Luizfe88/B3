import logging
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger("StatisticalValidator")


class PermutationValidator:
    def __init__(self, n_permutations: int = 5000, p_value_threshold: float = 0.05):
        self.n_permutations = int(n_permutations)
        self.p_value_threshold = float(p_value_threshold)

    def run_test(self, real_returns: "pd.Series") -> bool:
        """
        Runs a permutation test over the provided returns series.

        Returns True when the observed metric is statistically significant
        (p-value <= threshold) meaning the strategy is unlikely to be noise.
        """
        if not isinstance(real_returns, pd.Series):
            real_returns = pd.Series(real_returns)

        if len(real_returns) < 30:
            logger.warning("📊 Dados insuficientes para teste de permutação sólido.")
            return True

        real_metric = self._calculate_metric(real_returns.values)

        null_distribution = Parallel(n_jobs=-1)(
            delayed(self._permute_and_score)(real_returns.values)
            for _ in range(self.n_permutations)
        )

        hits = np.sum(np.array(null_distribution) >= real_metric)
        p_value = hits / float(self.n_permutations)

        logger.info(
            f"🧪 Permutation Test: Metric={real_metric:.6f}, p-value={p_value:.6f}"
        )

        return p_value <= self.p_value_threshold

    def _permute_and_score(self, returns: Sequence[float]) -> float:
        shuffled = np.random.permutation(returns)
        return self._calculate_metric(shuffled)

    def _calculate_metric(self, returns: Sequence[float]) -> float:
        arr = np.asarray(returns, dtype=float)
        if arr.size == 0:
            return 0.0
        std = np.std(arr, ddof=0)
        if std == 0:
            return 0.0
        return float(np.mean(arr) / std)
