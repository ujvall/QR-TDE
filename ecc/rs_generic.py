"""
Version-generic Reed-Solomon error counting for QR symbols (versions 1-40, ECC L/M/Q/H).

Input is the sampled module matrix (True = dark, no quiet zone), e.g. the `straight_qrcode`
grid OpenCV returns after a successful decode. For every RS block the codewords are
de-interleaved and decoded with reedsolo; the number of corrected codewords is reported.

Why counts instead of the V3-M "any non-zero syndrome" rule: a photographed genuine code
routinely has a few mis-sampled modules, so a binary syndrome test fires on clean photos.
The fraction of each block's correction capacity that was consumed is a graded signal that
can be thresholded on validation data.

Function-pattern geometry and RS block tables come from the `qrcode` package so they match
the encoder exactly.
"""

from functools import lru_cache
from typing import Dict, Any

import numpy as np
import qrcode
from qrcode import base, util
from reedsolo import RSCodec, ReedSolomonError

# Format-info EC bits -> level letter (qrcode uses the on-symbol bit values as constants).
EC_NAMES = {qrcode.constants.ERROR_CORRECT_L: "L", qrcode.constants.ERROR_CORRECT_M: "M",
            qrcode.constants.ERROR_CORRECT_Q: "Q", qrcode.constants.ERROR_CORRECT_H: "H"}
_FORMAT_CODES = {util.BCH_type_info((ec << 3) | mask): (ec, mask)
                 for ec in EC_NAMES for mask in range(8)}


@lru_cache(maxsize=None)
def _layout(version: int):
    """(function-module mask, ordered data coordinates, format-bit coordinates) for a version."""
    qr = qrcode.QRCode(version=version)
    n = qr.modules_count = version * 4 + 17
    qr.modules = [[None] * n for _ in range(n)]
    qr.setup_position_probe_pattern(0, 0)
    qr.setup_position_probe_pattern(n - 7, 0)
    qr.setup_position_probe_pattern(0, n - 7)
    qr.setup_position_adjust_pattern()
    qr.setup_timing_pattern()
    qr.setup_type_info(True, 0)
    if version >= 7:
        qr.setup_type_number(True)
    functional = np.array([[m is not None for m in row] for row in qr.modules])

    coords, row, inc = [], n - 1, -1
    for col in range(n - 1, 0, -2):
        if col <= 6:
            col -= 1  # skip the vertical timing column
        while 0 <= row < n:
            for c in (col, col - 1):
                if not functional[row, c]:
                    coords.append((row, c))
            row += inc
        row -= inc
        inc = -inc

    # Same bit order as qrcode.QRCode.setup_type_info (bit i of the 15-bit word).
    vert = [(i, 8) if i < 6 else (i + 1, 8) if i < 8 else (n - 15 + i, 8) for i in range(15)]
    horz = [(8, n - i - 1) if i < 8 else (8, 15 - i) if i < 9 else (8, 14 - i) for i in range(15)]
    return functional, coords, (vert, horz)


@lru_cache(maxsize=None)
def _codec(nsym: int) -> RSCodec:
    # QR: GF(256) with primitive 0x11D, generator 2, first consecutive root 0.
    return RSCodec(nsym, fcr=0, prim=0x11D, generator=2, c_exp=8)


def _read_format(matrix, coords):
    best = None
    for copy in coords:
        word = sum(int(matrix[r, c]) << i for i, (r, c) in enumerate(copy))
        for code, ecmask in _FORMAT_CODES.items():
            d = bin(code ^ word).count("1")
            if best is None or d < best[0]:
                best = (d, ecmask)
    return best  # (hamming distance, (ec, mask))


def analyze_matrix(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Count RS-corrected codewords in a sampled QR module matrix.

    Returns a dict with: status ("ok" | "uncorrectable" | "unsupported"), version, ec_level,
    blocks [{n, k, errors}], total_errors, max_capacity_used (0..1, >1 means a block failed),
    format_distance (bits corrected in the format word).
    """
    m = np.asarray(matrix).astype(bool)
    n = m.shape[0]
    if m.ndim != 2 or m.shape[0] != m.shape[1] or n < 21 or (n - 17) % 4 or n > 177:
        return {"status": "unsupported", "reason": f"matrix shape {m.shape} is not a QR grid"}
    version = (n - 17) // 4
    functional, coords, fmt_coords = _layout(version)
    fmt_dist, (ec, mask) = _read_format(m, fmt_coords)
    if fmt_dist > 3:  # BCH(15,5) corrects up to 3 bit errors
        return {"status": "uncorrectable", "version": version, "reason": "format information unreadable",
                "format_distance": fmt_dist, "max_capacity_used": float("inf"), "total_errors": None}

    flip = util.mask_func(mask)
    bits = [int(m[r, c] ^ flip(r, c)) for r, c in coords]
    blocks = base.rs_blocks(version, ec)
    total = sum(b.total_count for b in blocks)
    codewords = [int("".join(map(str, bits[i * 8:i * 8 + 8])), 2) for i in range(total)]

    # De-interleave: data codewords column-wise across blocks, then ECC codewords.
    data = [[] for _ in blocks]
    pos = 0
    for i in range(max(b.data_count for b in blocks)):
        for bi, b in enumerate(blocks):
            if i < b.data_count:
                data[bi].append(codewords[pos])
                pos += 1
    ecc = [[] for _ in blocks]
    for i in range(max(b.total_count - b.data_count for b in blocks)):
        for bi, b in enumerate(blocks):
            if i < b.total_count - b.data_count:
                ecc[bi].append(codewords[pos])
                pos += 1

    report, worst, total_err, failed = [], 0.0, 0, False
    for b, d, e in zip(blocks, data, ecc):
        nsym = b.total_count - b.data_count
        try:
            _, _, errata = _codec(nsym).decode(bytearray(d + e))
            errs = len(errata)
        except ReedSolomonError:
            errs, failed = None, True
        report.append({"n": b.total_count, "k": b.data_count, "errors": errs})
        if errs is not None:
            total_err += errs
            worst = max(worst, errs / (nsym // 2))
    return {
        "status": "uncorrectable" if failed else "ok",
        "version": version,
        "ec_level": EC_NAMES[ec],
        "mask": mask,
        "format_distance": fmt_dist,
        "blocks": report,
        "total_errors": total_err,
        "max_capacity_used": float("inf") if failed else worst,
    }


if __name__ == "__main__":
    # Self-check: clean symbols decode with zero errors; flipping data modules is counted.
    rng = np.random.default_rng(0)
    for version in (1, 3, 4, 7, 10, 25):
        for ec in EC_NAMES:
            q = qrcode.QRCode(version=version, error_correction=ec, border=0)
            q.add_data("QTDE")
            q.make(fit=False)
            mat = np.array(q.get_matrix(), bool)
            r = analyze_matrix(mat)
            assert r["status"] == "ok" and r["total_errors"] == 0, (version, ec, r)
            assert r["version"] == version and r["ec_level"] == EC_NAMES[ec]
            _, coords, _ = _layout(version)
            hit = coords[8 * 3][0], coords[8 * 3][1]  # one module inside codeword #3
            mat2 = mat.copy()
            mat2[hit] = ~mat2[hit]
            r2 = analyze_matrix(mat2)
            assert r2["total_errors"] == 1, (version, ec, r2)
    print("rs_generic self-check passed")
