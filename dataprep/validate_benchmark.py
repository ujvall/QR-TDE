"""
Integrity and leakage checks for data/benchmark/manifest.csv. Exits non-zero on any violation
and writes split statistics to results/benchmark/dataset_stats.json.

Checks
  - every image exists and loads; labels are 0/1; categories are known
  - every pair_id has one genuine image and at most one tampered image, in the same split
  - no QR identity (group_id) or payload appears in more than one split
  - no byte-identical images anywhere
  - no near-duplicate genuine images across splits: 64-bit dHash candidates (Hamming <= 4)
    confirmed by Pearson correlation >= 0.9 of 64x64 grayscale thumbnails

    python -m dataprep.validate_benchmark
"""

import collections
import hashlib
import itertools
import os
import sys

import cv2
import numpy as np

from dataprep.build_benchmark import CATEGORIES
from experiments.common import ROOT, image_path, load_manifest, save_json

KNOWN = set(CATEGORIES) | {"genuine", "legacy_synthetic"}


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def validate(rows):
    errors = []
    by_pair = collections.defaultdict(list)
    sha, hashes = {}, []
    for r in rows:
        img = cv2.imread(image_path(r), cv2.IMREAD_GRAYSCALE)
        if img is None:
            errors.append(f"unreadable {r['sample_id']}")
            continue
        if r["label"] not in ("0", "1") or r["category"] not in KNOWN:
            errors.append(f"bad label/category {r['sample_id']}")
        if (r["label"] == "0") != (r["category"] == "genuine"):
            errors.append(f"label/category mismatch {r['sample_id']}")
        by_pair[r["pair_id"]].append(r)
        with open(image_path(r), "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h in sha:
            errors.append(f"identical bytes: {r['sample_id']} == {sha[h]}")
        sha[h] = r["sample_id"]
        if r["label"] == "0":
            hashes.append((r["split"], r["sample_id"], dhash(img), cv2.resize(img, (64, 64)).astype(np.float32).ravel()))

    for pid, members in by_pair.items():
        labels = sorted(m["label"] for m in members)
        if labels not in (["0", "1"], ["0"]) or len({m["split"] for m in members}) != 1:
            errors.append(f"pair {pid}: labels {labels}, splits {sorted({m['split'] for m in members})}")

    for key in ("group_id", "original_payload"):
        seen = collections.defaultdict(set)
        for r in rows:
            if r[key]:
                seen[r[key]].add(r["split"])
        for k, s in seen.items():
            if len(s) > 1:
                errors.append(f"{key} {k!r} in several splits: {sorted(s)}")

    near = {"hash_candidates": 0, "confirmed": 0, "max_candidate_correlation": None, "candidates": []}
    for (s1, a, h1, t1), (s2, b, h2, t2) in itertools.combinations(hashes, 2):
        if s1 != s2 and bin(h1 ^ h2).count("1") <= 4:
            corr = float(np.corrcoef(t1, t2)[0, 1])
            near["hash_candidates"] += 1
            near["candidates"].append({"a": a, "b": b, "correlation": round(corr, 3)})
            near["max_candidate_correlation"] = max(corr, near["max_candidate_correlation"] or -1)
            if corr >= 0.9:
                near["confirmed"] += 1
                errors.append(f"near-duplicate genuine images across splits: {a} ({s1}) ~ {b} ({s2}), r={corr:.3f}")
    return errors, near


def stats(rows):
    out = {}
    for sp in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == sp]
        out[sp] = {
            "images": len(rs),
            "genuine": sum(r["label"] == "0" for r in rs),
            "tampered": sum(r["label"] == "1" for r in rs),
            "qr_identities": len({r["group_id"] for r in rs}),
            "sources": dict(collections.Counter(r["source"] for r in rs)),
            "categories": dict(collections.Counter(r["category"] for r in rs if r["label"] == "1")),
            "decodable_genuine": sum(r["decodes"] == "1" for r in rs if r["label"] == "0"),
            "decodable_tampered": sum(r["decodes"] == "1" for r in rs if r["label"] == "1"),
            "versions": dict(collections.Counter(r["version"] for r in rs if r["label"] == "0")),
            "ecc_levels": dict(collections.Counter(r["ecc_level"] or "unknown" for r in rs if r["label"] == "0")),
        }
    return out


def main():
    rows = load_manifest()
    errors, near = validate(rows)
    st = stats(rows)
    save_json(os.path.join(ROOT, "results", "benchmark", "dataset_stats.json"),
              {"splits": st, "near_duplicates_across_splits": near, "violations": errors})
    for sp, s in st.items():
        print(f"{sp:15s} {s['images']:5d} images  {s['genuine']:4d} genuine  {s['tampered']:4d} tampered  "
              f"{s['qr_identities']:4d} identities  {s['sources']}")
    if errors:
        print("\n".join(errors[:50]))
        sys.exit(f"{len(errors)} violations")
    print("benchmark valid: no leakage or integrity violations found")


if __name__ == "__main__":
    main()
