"""Tests for the version-generic Reed-Solomon analysis (ecc/rs_generic.py)."""

import unittest

import numpy as np
import qrcode

from ecc.rs_generic import EC_NAMES, _layout, analyze_matrix


def matrix(version, ec, data="QTDE"):
    q = qrcode.QRCode(version=version, error_correction=ec, border=0)
    q.add_data(data)
    q.make(fit=False)
    return np.array(q.get_matrix(), bool)


class TestRSGeneric(unittest.TestCase):
    def test_clean_symbols_have_no_errors(self):
        for version in (1, 2, 5, 7, 12, 40):
            for ec, name in EC_NAMES.items():
                r = analyze_matrix(matrix(version, ec))
                self.assertEqual(r["status"], "ok")
                self.assertEqual((r["version"], r["ec_level"], r["total_errors"]), (version, name, 0))

    def test_error_count_grows_with_flipped_codewords(self):
        m = matrix(5, qrcode.constants.ERROR_CORRECT_H)
        _, coords, _ = _layout(5)
        for k in (1, 3, 5):
            t = m.copy()
            for cw in range(k):            # one module in each of k distinct codewords
                t[coords[cw * 8 * 3]] ^= True
            self.assertEqual(analyze_matrix(t)["total_errors"], k)

    def test_overwhelmed_block_is_uncorrectable(self):
        m = matrix(3, qrcode.constants.ERROR_CORRECT_L)
        _, coords, _ = _layout(3)
        t = m.copy()
        for r, c in coords[:400]:
            t[r, c] ^= True
        r = analyze_matrix(t)
        self.assertEqual(r["status"], "uncorrectable")
        self.assertEqual(r["max_capacity_used"], float("inf"))

    def test_non_qr_shape_is_unsupported(self):
        self.assertEqual(analyze_matrix(np.zeros((30, 30), bool))["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
