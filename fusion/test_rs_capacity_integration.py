"""
Integration tests for the version-generic Reed-Solomon capacity component in the active
pipeline (fusion.decision_engine.rs_capacity_evidence + app.inspector).
"""

import unittest

import cv2
import numpy as np
import qrcode

from app.inspector import QRInspector
from ecc.rs_generic import _layout
from fusion.decision_engine import RS_CAPACITY_THRESHOLD, rs_capacity_evidence


def render(version, level, payload, flip_codewords=0, box=10):
    q = qrcode.QRCode(version=version, error_correction=level, box_size=box, border=4)
    q.add_data(payload)
    q.make(fit=False)
    m = np.array(q.get_matrix(), bool)
    _, coords, _ = _layout(version)
    for cw in range(flip_codewords):              # one module in each of k consecutive codewords
        r, c = coords[cw * 8 + 3]
        m[r + 4, c + 4] ^= True
    return np.where(np.kron(m, np.ones((box, box), bool)), 0, 255).astype(np.uint8)


class TestRSCapacityIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspector = QRInspector()

    def test_missing_grid_is_unavailable_not_clean(self):
        r = rs_capacity_evidence(None)
        self.assertEqual(r["status"], "unavailable")
        self.assertFalse(r["triggered"])

    def test_clean_v3m_reports_within_capacity(self):
        res = self.inspector.inspect_image(image=render(3, qrcode.constants.ERROR_CORRECT_M, "https://example.org/a"))
        rs = res["tampering"]["rs_capacity"]
        self.assertEqual(rs["status"], "within_capacity")
        self.assertEqual((rs["version"], rs["ec_level"], rs["corrected_codewords"]), (3, "M", 0))
        self.assertNotIn("ecc_capacity_exceeded", res["tampering"]["evidence"])

    def test_non_v3_tampering_is_now_detected(self):
        # Version 5-H: previously "unsupported_geometry" with no ECC analysis at all.
        clean = self.inspector.inspect_image(image=render(5, qrcode.constants.ERROR_CORRECT_H, "https://example.org/b"))
        self.assertEqual(clean["tampering"]["status"], "unsupported_geometry")
        self.assertEqual(clean["tampering"]["rs_capacity"]["status"], "within_capacity")

        # 5-H has 4 blocks of 22 ECC codewords (capacity 11); 16 consecutive codewords put 4 errors
        # in every block (4/11 > threshold) while the symbol still decodes.
        img = render(5, qrcode.constants.ERROR_CORRECT_H, "https://example.org/b", flip_codewords=16)
        res = self.inspector.inspect_image(image=img)
        rs = res["tampering"]["rs_capacity"]
        self.assertGreater(rs["max_capacity_used"], RS_CAPACITY_THRESHOLD)
        self.assertIn("ecc_capacity_exceeded", res["tampering"]["evidence"])
        self.assertEqual(res["tampering"]["status"], "tampering_detected")

    def test_url_analysis_unaffected_by_physical_evidence(self):
        img = render(5, qrcode.constants.ERROR_CORRECT_H, "https://example.org/b", flip_codewords=16)
        tampered = self.inspector.inspect_image(image=img)
        clean = self.inspector.inspect_image(image=render(5, qrcode.constants.ERROR_CORRECT_H, "https://example.org/b"))
        self.assertEqual(tampered["url"], clean["url"])


if __name__ == "__main__":
    unittest.main()
