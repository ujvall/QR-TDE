"""
Compare all scored detectors on identical splits and write the benchmark report.

Threshold rule (fixed before looking at any test split): for every continuous detector the
threshold is the lowest score keeping VALIDATION specificity >= 95 %. Detectors that already
emit a verdict (the current engine, the V3-M syndrome rule, the original peak threshold,
decode failure) are reported as-is. OR-combinations calibrate each component at
1 - 0.05/k validation specificity so the combination stays near the 95 % target.

Unavailable analyses (nothing localized / decoded) never count as an alarm.

    python -m experiments.benchmark

Outputs results/benchmark/report.json and results/benchmark/REPORT.md.
"""

import argparse
import json
import os

import numpy as np

from experiments.common import (ROOT, SCORES, TARGET_SPECIFICITY, binary_metrics, bootstrap_ci,
                                load_manifest, read_scores, save_json, threshold_at_specificity)

OUT = os.path.join(ROOT, "results", "benchmark")
VERDICTS = {"baseline_engine": "Original engine (ECC V3-M + peak), before this work",
            "integrated_engine": "Updated engine (original + RS capacity), as now deployed",
            "baseline_ecc_v3m": "Current ECC rule alone (V3-M, any non-zero syndrome)",
            "decode_failure": "Localized but undecodable"}
CONTINUOUS = {"baseline_peak": "Peak residual, recalibrated",
              "rs_capacity": "Generic RS capacity used (new, all versions)",
              "resnet18": "ResNet-18 fine-tuned",
              "vit_finetune": "ViT-B/16 fine-tuned",
              "vit_probe": "ViT-B/16 frozen + logistic regression",
              "conv_ae": "Conv autoencoder (one-class)",
              "patchcore": "PatchCore-style kNN (one-class)"}
COMBOS = {"rs_capacity+baseline_peak": "Classical: RS capacity OR peak (recalibrated)",
          "rs_capacity+resnet18": "RS capacity OR ResNet-18",
          "rs_capacity+baseline_peak+resnet18": "RS capacity OR peak OR ResNet-18"}
ABLATION = {
    "ablation_engine_plus_rs": (["baseline_ecc_v3m", "baseline_peak_original", "rs_capacity"],
                                "Ablation: current engine + RS capacity"),
    "ablation_rs_plus_peak": (["rs_capacity", "baseline_peak_original"],
                              "Ablation: RS capacity + original peak (V3-M binary rule removed)"),
}
TEST_SPLITS = ["test_synthetic", "test_real", "test_external", "test_physical", "legacy_val", "legacy_test"]


def load_scores(names, ids):
    out = {}
    for n in names:
        if not os.path.exists(os.path.join(SCORES, f"{n}.csv")):
            continue
        s = read_scores(n)
        sc = np.array([float(s[i]["score"]) for i in ids])
        av = np.array([int(s[i]["available"]) for i in ids], bool)
        out[n] = np.where(av, sc, -np.inf), av
    return out


def evaluate(pred, score, y, split, cats, units, mask):
    m = binary_metrics(y[mask], pred[mask], score[mask] if score is not None else None)
    m["recall_ci95"] = bootstrap_ci(y[mask], pred[mask], units[mask], "recall")
    m["specificity_ci95"] = bootstrap_ci(y[mask], pred[mask], units[mask], "specificity")
    m["per_category_recall"] = {c: {"n": int(((cats == c) & mask).sum()),
                                    "recall": float(pred[(cats == c) & mask].mean())}
                                for c in sorted(set(cats[mask & (y == 1)]))}
    return m


def paired_recall_gain(pa, pb, y, units, mask, n=2000, seed=42):
    """Bootstrap CI of recall(a) - recall(b) on the same tampered samples."""
    idx = np.flatnonzero(mask & (y == 1))
    u = units[idx]
    ug = np.unique(u)
    by = {g: idx[u == g] for g in ug}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sel = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        diffs.append(pa[sel].mean() - pb[sel].mean())
    return float(np.mean(pa[idx]) - np.mean(pb[idx])), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    ids = [r["sample_id"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    cats = np.array([r["category"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    pairs = np.array([r["pair_id"] for r in rows])
    val = split == "val"

    sc = load_scores(list(VERDICTS) + list(CONTINUOUS), ids)
    preds, scores, thresholds, labels = {}, {}, {}, {}
    for n, (s, av) in sc.items():
        labels[n] = VERDICTS.get(n) or CONTINUOUS[n]
        if n in VERDICTS:
            preds[n], scores[n], thresholds[n] = (s > 0.5).astype(int), np.where(av, s, 0), "verdict"
        else:
            t = threshold_at_specificity(s[val & (y == 0)])
            preds[n], scores[n], thresholds[n] = (s > t).astype(int), s, t
    if "baseline_peak" in sc:   # also report the peak detector at its original fixed threshold
        from fusion.decision_engine import PEAK_ANOMALY_THRESHOLD
        preds["baseline_peak_original"] = (sc["baseline_peak"][0] > PEAK_ANOMALY_THRESHOLD).astype(int)
        scores["baseline_peak_original"] = sc["baseline_peak"][0]
        thresholds["baseline_peak_original"] = PEAK_ANOMALY_THRESHOLD
        labels["baseline_peak_original"] = "Peak residual, original threshold 0.5255"
    for combo, label in COMBOS.items():
        parts = combo.split("+")
        if not all(p in sc for p in parts):
            continue
        target = 1 - (1 - TARGET_SPECIFICITY) / len(parts)
        ps = []
        for p in parts:
            t = threshold_at_specificity(sc[p][0][val & (y == 0)], target)
            ps.append(sc[p][0] > t)
        preds[combo] = np.any(ps, 0).astype(int)
        scores[combo] = None
        thresholds[combo] = f"each component at {target:.3f} validation specificity"
        labels[combo] = label
    # Ablation of the deployed decision logic: OR of verdicts at their deployed thresholds.
    for name, (parts, label) in ABLATION.items():
        if all(p in preds for p in parts):
            preds[name] = np.any([preds[p] for p in parts], 0).astype(int)
            scores[name], thresholds[name], labels[name] = None, "deployed thresholds", label

    report = {"threshold_rule": f"validation specificity >= {TARGET_SPECIFICITY}", "thresholds": thresholds,
              "labels": labels, "splits": {}, "paired_vs_baseline": {}, "coverage": {}}
    for sp in ["val"] + TEST_SPLITS:
        mask = split == sp
        if not mask.any():
            continue
        units = groups if len(np.unique(groups[mask])) >= 30 else pairs
        report["splits"][sp] = {
            "n_genuine": int((mask & (y == 0)).sum()), "n_tampered": int((mask & (y == 1)).sum()),
            "bootstrap_unit": "group_id" if units is groups else "pair_id (too few QR identities)",
            "detectors": {n: evaluate(preds[n], scores[n], y, split, cats, units, mask) for n in preds},
        }
        report["coverage"][sp] = {n: float(av[mask].mean()) for n, (_, av) in sc.items()}
        if "baseline_engine" in preds:
            report["paired_vs_baseline"][sp] = {
                n: dict(zip(("recall_gain", "ci95"), paired_recall_gain(preds[n], preds["baseline_engine"], y, units, mask)))
                for n in preds if n != "baseline_engine"}

    save_json(os.path.join(OUT, "report.json"), report)
    write_markdown(report)
    print(open(os.path.join(OUT, "REPORT.md"), encoding="utf-8").read())


def pct(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.1f}"


def write_markdown(rep):
    lines = ["# Q-TDE tampering benchmark report", "",
             "Generated by `python -m experiments.benchmark`. Thresholds: " + rep["threshold_rule"] +
             " (continuous detectors); verdict detectors as deployed. CIs: 95 % bootstrap.", ""]
    for sp, d in rep["splits"].items():
        lines += [f"## {sp}  ({d['n_genuine']} genuine / {d['n_tampered']} tampered, bootstrap unit: {d['bootstrap_unit']})", "",
                  "| Detector | Recall [95% CI] | Specificity [95% CI] | Precision | F1 | ROC-AUC | PR-AUC | TP/FN/FP/TN |",
                  "|---|---|---|---|---|---|---|---|"]
        for n, m in d["detectors"].items():
            rc, scf = m["recall_ci95"], m["specificity_ci95"]
            lines.append(f"| {rep['labels'][n]} | {pct(m['recall'])} [{pct(rc[0])}-{pct(rc[1])}] | "
                         f"{pct(m['specificity'])} [{pct(scf[0])}-{pct(scf[1])}] | {pct(m['precision'])} | "
                         f"{pct(m['f1'])} | {m.get('roc_auc', float('nan')):.3f} | {m.get('pr_auc', float('nan')):.3f} | "
                         f"{m['tp']}/{m['fn']}/{m['fp']}/{m['tn']} |")
        if sp in rep["paired_vs_baseline"]:
            lines += ["", "Recall gain over the current engine (paired bootstrap):", ""]
            for n, g in rep["paired_vs_baseline"][sp].items():
                lines.append(f"- {rep['labels'][n]}: {100 * g['recall_gain']:+.1f} pp [{100 * g['ci95'][0]:+.1f}, {100 * g['ci95'][1]:+.1f}]")
        lines.append("")
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
