"""
Score every benchmark image with the classical detectors (no training involved).

Writes results/benchmark/scores/<detector>.csv with columns sample_id, score, available:
  baseline_engine   current active pipeline verdict (QRInspector): 1 = tampering_detected
  baseline_ecc_v3m  current V3-M syndrome rule: 1 = residual > 0 (only defined for V3-M codes)
  baseline_peak     current high-frequency peak residual (continuous)
  rs_capacity       version-generic RS analysis: max fraction of a block's correction capacity
                    used (inf when a block is uncorrectable); unavailable if nothing decodes
  decode_failure    1 if the code is localized but cannot be decoded

    python -m experiments.classical
    python -m experiments.classical --engine-only integrated_engine   # re-score the active engine
                                                                      # without touching the frozen baseline

baseline_engine.csv was produced before the RS-capacity component was integrated (commit 040c452c).
"""

import argparse

import math

import cv2

from app.inspector import QRInspector
from ecc.rs_generic import analyze_matrix
from experiments.common import image_path, load_manifest, write_scores

NAMES = ["baseline_engine", "baseline_ecc_v3m", "baseline_peak", "rs_capacity", "decode_failure"]


def score_image(path, inspector, detector):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = inspector.inspect_image(image=img)
    tam = res["tampering"]
    out = {
        "baseline_engine": (int(tam["status"] == "tampering_detected"), 1),
        "baseline_ecc_v3m": (int(tam["ecc"]["residual"] > 0),
                             int(tam["ecc"]["status"] in ("valid_codewords", "integrity_violation_detected"))),
        "baseline_peak": (tam["peak_anomaly"]["score"], int(tam["peak_anomaly"]["status"] in
                                                            ("anomaly_detected", "within_normal_limits"))),
    }
    try:
        data, pts, straight = detector.detectAndDecode(gray)
    except cv2.error:
        data, pts, straight = "", None, None
    if data and straight is not None:
        r = analyze_matrix(straight < 128)
        out["rs_capacity"] = (r.get("max_capacity_used", math.inf), int(r["status"] != "unsupported"))
    else:
        out["rs_capacity"] = (0.0, 0)
    localized = pts is not None or res.get("preprocessing", {}).get("corners") is not None
    out["decode_failure"] = (int(localized and not data), int(localized))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-only", metavar="NAME", help="only score the engine verdict, saved as NAME")
    args = ap.parse_args()
    rows = load_manifest()
    inspector, detector = QRInspector(), cv2.QRCodeDetector()
    per = {k: [] for k in ([args.engine_only] if args.engine_only else NAMES)}
    for i, row in enumerate(rows):
        if args.engine_only:
            res = inspector.inspect_image(image=cv2.imread(image_path(row)))
            per[args.engine_only].append({"sample_id": row["sample_id"], "available": 1,
                                          "score": int(res["tampering"]["status"] == "tampering_detected")})
            continue
        for name, (score, avail) in score_image(image_path(row), inspector, detector).items():
            per[name].append({"sample_id": row["sample_id"], "score": score, "available": avail})
        if i % 250 == 0:
            print(i, "/", len(rows))
    for name, r in per.items():
        write_scores(name, r)
    print("classical scores written")


if __name__ == "__main__":
    main()
