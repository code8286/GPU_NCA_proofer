import math
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_transfer_v4_frequency import count_parts, conditional_patch_nll


class CausalFrequencyDiagnosticChecks(unittest.TestCase):
    def test_closed_form_is_causal_sequential_predictive_likelihood(self):
        c = {'model': {'context': 17}, 'nca': {'grid': 4, 'states': 2}}
        row = np.array([[16, 1, 2, 1, 4, 17, 16, 1, 3, 2, 1, 17, 16, 4, 3, 2, 2, 17]], dtype=np.uint16)
        initial, target = count_parts(row, c)
        p = np.arange(1, 17, dtype=float)
        p /= p.sum()
        for alpha in (.1, 10., 1000.):
            counts = initial[0].copy()
            total, logp, n = counts.sum(), 0., 0
            for token in row[0, 6:18]:
                if token >= 16:
                    continue
                logp += math.log((counts[token]+alpha*p[token])/(total+alpha))
                counts[token] += 1
                total += 1
                n += 1
            self.assertAlmostEqual(conditional_patch_nll(initial, target, p, alpha), -logp/n, places=11)

    def test_markers_and_unused_future_tokens_do_not_enter_the_baseline(self):
        c = {'model': {'context': 11}, 'nca': {'grid': 4, 'states': 2}}
        a = np.array([[16, 1, 2, 1, 4, 17, 16, 1, 3, 2, 1, 17, 16, 4, 3, 2, 2, 17]], dtype=np.uint16)
        b = a.copy()
        b[:, 12:] = 15
        b[b >= 16] = 17
        ai, at = count_parts(a, c)
        bi, bt = count_parts(b, c)
        np.testing.assert_array_equal(ai, bi)
        np.testing.assert_array_equal(at, bt)
        self.assertEqual(ai.sum(), 4)
        self.assertEqual(at.sum(), 4)


if __name__ == '__main__':
    unittest.main()
