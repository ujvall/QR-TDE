"""Tests for benchmark metrics, threshold calibration and the tampering geometry."""

import unittest

import numpy as np

from experiments.common import binary_metrics, bootstrap_ci, threshold_at_specificity


class TestMetrics(unittest.TestCase):
    def test_threshold_keeps_target_specificity(self):
        genuine = np.arange(100, dtype=float)
        t = threshold_at_specificity(genuine, 0.95)
        self.assertGreaterEqual(np.mean(genuine <= t), 0.95)
        self.assertLess(np.mean(genuine <= t - 1), 0.95)   # lowest such threshold

    def test_unavailable_scores_never_alarm(self):
        genuine = np.array([-np.inf] * 50 + list(range(50)), float)
        t = threshold_at_specificity(genuine, 0.95)
        self.assertFalse(-np.inf > t)

    def test_binary_metrics(self):
        m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.2, 0.1, 0.8])
        self.assertEqual((m["tp"], m["fn"], m["tn"], m["fp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["specificity"], 0.5)
        self.assertAlmostEqual(m["roc_auc"], 0.75)

    def test_bootstrap_ci_brackets_estimate(self):
        rng = np.random.default_rng(0)
        y = np.ones(200, int)
        pred = (rng.random(200) < 0.7).astype(int)
        lo, hi = bootstrap_ci(y, pred, np.arange(200), "recall", n=500)
        self.assertLess(lo, pred.mean())
        self.assertGreater(hi, pred.mean())


class TestTamperGeometry(unittest.TestCase):
    def test_marker_edit_lands_on_intended_modules(self):
        """A marker edit in module space must show up as RS errors after a simulated capture."""
        import cv2
        from dataprep.build_benchmark import capture_params, qr_matrix, simulate, tamper_layer
        from ecc.rs_generic import analyze_matrix
        rng = np.random.default_rng(3)
        m, version, level = qr_matrix("https://example.org/t", rng, version=4, level="H")
        p = capture_params(rng)
        p.update(blur=0.0, noise=1.0, jpeg=95, persp=0.02, scale=0.6)
        g, _ = simulate(m, p, None, np.random.default_rng(1))
        layer, info = tamper_layer("marker_edit", m, version, level, np.random.default_rng(5))
        t, _ = simulate(m, p, layer, np.random.default_rng(1))
        det = cv2.QRCodeDetector()
        errs = []
        for img in (g, t):
            data, _, straight = det.detectAndDecode(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            self.assertTrue(data)
            errs.append(analyze_matrix(straight < 128)["total_errors"])
        self.assertEqual(errs[0], 0)
        self.assertGreater(errs[1], 0)


if __name__ == "__main__":
    unittest.main()
