"""
Shared helpers for the Q-TDE benchmark: manifest loading, QR crop rectification, metrics,
validation-only threshold selection and bootstrap confidence intervals.
"""

import csv
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "data", "benchmark")
SCORES = os.path.join(ROOT, "results", "benchmark", "scores")
SEED = 42

# Pre-registered operating point: every detector's threshold is the lowest score that keeps
# validation specificity >= 95 %. Chosen before any test-set result was looked at.
TARGET_SPECIFICITY = 0.95


def load_manifest(path: Optional[str] = None) -> List[Dict[str, str]]:
    with open(path or os.path.join(BENCH, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def image_path(row) -> str:
    return os.path.join(BENCH, row["image_path"])


def rectified_crop(gray: np.ndarray, size: int = 224, margin: float = 0.12) -> Optional[np.ndarray]:
    """
    Localize the QR code with OpenCV and warp it to a size x size crop (with a margin so the
    code border and any sticker edge stay visible). Returns None if nothing is localized.
    Uses the image's own detection, never ground-truth corners, so it matches deployment.
    """
    pts = None
    for det in (cv2.QRCodeDetector(), getattr(cv2, "QRCodeDetectorAruco", lambda: None)()):
        if det is None:
            continue
        try:
            ok, p = det.detect(gray)
        except cv2.error:
            ok, p = False, None
        if ok and p is not None:
            pts = p.reshape(-1, 2).astype(np.float32)
            break
    if pts is None or len(pts) != 4:
        return None
    m = size * margin
    dst = np.float32([[m, m], [size - m, m], [size - m, size - m], [m, size - m]])
    try:
        H = cv2.getPerspectiveTransform(pts, dst)
    except cv2.error:
        return None
    return cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_AREA, borderValue=255)


def threshold_at_specificity(genuine_scores, target=TARGET_SPECIFICITY) -> float:
    """Smallest threshold t such that the share of genuine validation scores <= t is >= target."""
    s = np.sort(np.asarray(genuine_scores, dtype=float))
    s = s[np.isfinite(s)] if np.isfinite(s).any() else s
    k = int(np.ceil(target * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def binary_metrics(y, pred, score=None) -> Dict[str, float]:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    rec, prec = div(tp, tp + fn), div(tp, tp + fp)
    out = {
        "n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": div(tp + tn, len(y)), "precision": prec, "recall": rec,
        "specificity": div(tn, tn + fp), "f1": div(2 * prec * rec, prec + rec) if tp else 0.0,
        "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
    }
    if score is not None and len(set(y)) == 2:
        sc = np.nan_to_num(np.asarray(score, float), nan=-1e9, posinf=1e9, neginf=-1e9)
        out["roc_auc"] = float(roc_auc_score(y, sc))
        out["pr_auc"] = float(average_precision_score(y, sc))
    return out


def bootstrap_ci(y, pred, groups, metric="recall", n=2000, seed=SEED):
    """95 % percentile CI, resampling whole groups (original QR identity) to respect pairing."""
    y, pred, groups = map(np.asarray, (y, pred, groups))
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        v = binary_metrics(y[sel], pred[sel])[metric]
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def write_scores(name: str, rows: List[Dict]):
    """rows: [{sample_id, score, (optional) available}]; higher score = more likely tampered."""
    os.makedirs(SCORES, exist_ok=True)
    with open(os.path.join(SCORES, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def read_scores(name: str) -> Dict[str, Dict]:
    with open(os.path.join(SCORES, f"{name}.csv"), newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
