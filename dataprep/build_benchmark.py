"""
Build the Q-TDE tampering benchmark in data/benchmark/ (git-ignored) with manifest.csv.

Splits
  train / val / test_synthetic  Simulated captures: many QR identities (versions 1-10, ECC
                                L/M/Q/H) printed on procedural backgrounds; tampering is applied
                                to the printed symbol *before* a camera model (perspective,
                                lighting, blur, noise, JPEG). Split by QR identity (group_id).
  test_real                     The 100 real photographs of Mendeley doi:10.17632/m6mfwc52vk.1
                                (CC BY 4.0). Genuine = untouched photo. Tampered = the same photo
                                with tampering applied *after* capture, blur/noise matched.
  test_external                 The Szeged Huawei photos (Bodnar et al. 2018), built the same way
                                as test_real. Different camera, lab and QR codes.

Genuine and tampered images of a source go through an identical resize + JPEG step, so
compression cannot separate the classes. Real photos are never used for training or for
threshold selection.

    python -m dataprep.build_benchmark            # ~3.5k images, a few minutes
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import string

import cv2
import numpy as np
import qrcode

from ecc.rs_generic import _layout
from experiments.common import BENCH, SEED

DATA = os.path.dirname(BENCH)
EXT = os.path.join(DATA, "external")
LEVELS = {"L": qrcode.constants.ERROR_CORRECT_L, "M": qrcode.constants.ERROR_CORRECT_M,
          "Q": qrcode.constants.ERROR_CORRECT_Q, "H": qrcode.constants.ERROR_CORRECT_H}
CATEGORIES = ["sticker_replacement", "label_overlay", "marker_edit", "white_out",
              "module_patch", "scratch", "finder_alteration"]
R = 10        # layer raster resolution, px per module
MARGIN = 4    # layer canvas margin around the symbol, in modules


# ------------------------------------------------------------------ QR identities

def random_payload(rng):
    tok = "".join(rng.choice(list(string.ascii_lowercase + string.digits), rng.integers(4, 14)))
    kind = rng.integers(0, 4)
    host = rng.choice(["example.org", "example.com", "example.net"])  # reserved, inert domains
    if kind == 0:
        return f"https://{host}/{tok}"
    if kind == 1:
        return f"http://{host}/p/{tok}?id={rng.integers(1, 99999)}"
    if kind == 2:
        return f"TICKET-{tok.upper()}"
    return f"https://{host}/{tok}/{''.join(rng.choice(list(string.ascii_lowercase), 6))}"


def qr_matrix(payload, rng, version=None, level=None):
    level = level or rng.choice(list(LEVELS))
    q = qrcode.QRCode(error_correction=LEVELS[level], border=0)
    q.add_data(payload)
    q.make(fit=True)
    if version is None:
        version = int(min(10, q.version + rng.integers(0, 4)))
    q = qrcode.QRCode(version=max(version, q.version), error_correction=LEVELS[level], border=0)
    q.add_data(payload)
    q.make(fit=False)
    return np.array(q.get_matrix(), bool), q.version, level


# ------------------------------------------------------------------ layer compositing
# A layer is drawn in module space on a canvas covering [-MARGIN, n+MARGIN] modules at R px/module:
#   A    alpha (0..1)       T  tone (0 = local ink level, 1 = local paper level)
#   tint per-channel gain   S  shadow mask (darkening)

def blank_layer(n):
    size = (n + 2 * MARGIN) * R
    return {"A": np.zeros((size, size), np.float32), "T": np.ones((size, size), np.float32),
            "S": np.zeros((size, size), np.float32), "tint": np.ones(3, np.float32)}


def px(v):
    """module coordinate -> canvas pixel"""
    return int(round((v + MARGIN) * R))


def fill_modules(layer, cells, tone, grow=0.0, rng=None):
    for r, c in cells:
        g = grow + (rng.uniform(-0.08, 0.12) if rng is not None else 0)
        cv2.rectangle(layer["A"], (px(c - g), px(r - g)), (px(c + 1 + g) - 1, px(r + 1 + g) - 1), 1.0, -1)
        cv2.rectangle(layer["T"], (px(c - g), px(r - g)), (px(c + 1 + g) - 1, px(r + 1 + g) - 1), float(tone), -1)


def paper_patch(layer, r0, c0, r1, c1, rng, matrix=None, shadow=True):
    """Opaque paper rectangle (module coords), optionally printed with a module matrix."""
    layer["A"][px(r0):px(r1), px(c0):px(c1)] = 1.0
    layer["T"][px(r0):px(r1), px(c0):px(c1)] = 1.0
    if matrix is not None:
        for (r, c), dark in np.ndenumerate(matrix):
            if dark:
                layer["T"][px(r0 + r):px(r0 + r + 1), px(c0 + c):px(c0 + c + 1)] = 0.0
    if shadow:
        d = rng.uniform(0.15, 0.5)
        S = np.zeros_like(layer["S"])
        S[px(r0 + d):px(r1 + d), px(c0 + d):px(c1 + d)] = rng.uniform(0.15, 0.35)
        layer["S"] = np.maximum(layer["S"], cv2.GaussianBlur(S, (0, 0), R * 0.4) * (1 - layer["A"]))


def data_cells(n):
    functional = _layout((n - 17) // 4)[0]
    return [(r, c) for r in range(n) for c in range(n) if not functional[r, c]]


def cluster(cells, k, rng):
    """k cells near a random seed cell (tampering tends to be local)."""
    cells = np.array(cells)
    seed = cells[rng.integers(len(cells))]
    d = np.abs(cells - seed).sum(1) + rng.uniform(0, 3, len(cells))
    return [tuple(x) for x in cells[np.argsort(d)[:k]]]


def tamper_layer(category, matrix, version, level, rng):
    """Return (layer, info) describing one tampering operation in module space."""
    n = len(matrix)
    L = blank_layer(n)
    info = {}
    if category == "sticker_replacement":
        payload = random_payload(rng)
        new, _, new_level = qr_matrix(payload, rng, version=version)
        q = rng.uniform(0.3, 2.0)
        sub = blank_layer(n)
        paper_patch(sub, -q, -q, n + q, n + q, rng)
        for (r, c), dark in np.ndenumerate(new):
            if dark:
                sub["T"][px(r):px(r + 1), px(c):px(c + 1)] = 0.0
        # misregistration of a hand-applied sticker
        ang, s = rng.uniform(-3, 3), rng.uniform(0.96, 1.05)
        M = cv2.getRotationMatrix2D((px(n / 2), px(n / 2)), ang, s)
        M[:, 2] += rng.uniform(-0.8, 0.8, 2) * R
        size = L["A"].shape[::-1]
        for k in ("A", "S"):
            L[k] = cv2.warpAffine(sub[k], M, size, flags=cv2.INTER_LINEAR)
        L["T"] = cv2.warpAffine(sub["T"], M, size, flags=cv2.INTER_LINEAR, borderValue=1.0)
        L["tint"] = np.float32(rng.uniform(0.99, 1.07, 3))
        info = {"replacement_payload": payload, "replacement_ecc": new_level}
    elif category == "label_overlay":
        area = rng.uniform(0.05, 0.15) * n * n
        h = np.sqrt(area) * rng.uniform(0.6, 1.6)
        w = area / h
        r0, c0 = rng.uniform(7, max(8, n - h - 1)), rng.uniform(7, max(8, n - w - 1))
        paper_patch(L, r0, c0, r0 + h, c0 + w, rng)
        for i in range(int(rng.integers(0, 4))):  # printed text lines on the label
            y = r0 + h * (i + 1) / 5
            L["T"][px(y):px(y + 0.35), px(c0 + 0.5):px(c0 + w * rng.uniform(0.4, 0.9))] = 0.35
        L["tint"] = np.float32(rng.choice([[1.0, 1.0, 1.0], [0.85, 1.0, 1.05], [0.95, 0.95, 1.08],
                                           [1.05, 0.95, 0.9]]) * rng.uniform(0.97, 1.05))
        info = {"area_fraction": round(float(area / n / n), 3)}
    elif category in ("marker_edit", "white_out"):
        want_dark = category == "white_out"   # white-out covers dark modules, marker fills light ones
        cand = [rc for rc in data_cells(n) if matrix[rc] == want_dark]
        k = int(rng.integers(3, 21))
        cells = cluster(cand, k, rng) if rng.random() < 0.7 else \
            [cand[i] for i in rng.choice(len(cand), k, replace=False)]
        tone = rng.uniform(-0.1, 0.12) if category == "marker_edit" else rng.uniform(0.95, 1.12)
        fill_modules(L, cells, tone, grow=0.05, rng=rng)
        L["tint"] = np.float32([1, 1, 1] if category == "white_out" else rng.choice([[1, 1, 1], [1.15, 0.95, 0.85]]))
        info = {"modules_changed": k}
    elif category == "module_patch":
        h, w = rng.integers(3, 8, 2)
        cells = data_cells(n)
        r0, c0 = cells[rng.integers(len(cells))]
        r0, c0 = min(r0, n - h), min(c0, n - w)
        paper_patch(L, r0 - 0.1, c0 - 0.1, r0 + h + 0.1, c0 + w + 0.1, rng,
                    matrix=None, shadow=rng.random() < 0.6)
        bits = rng.random((h, w)) < 0.5
        for (r, c), dark in np.ndenumerate(bits):
            if dark:
                L["T"][px(r0 + r):px(r0 + r + 1), px(c0 + c):px(c0 + c + 1)] = 0.0
        info = {"patch_modules": int(h * w)}
    elif category == "scratch":
        for _ in range(int(rng.integers(1, 4))):
            pts = np.cumsum(rng.uniform(-n / 3, n / 3, (4, 2)), 0) + rng.uniform(0, n, 2)
            pts = np.clip(pts, -1, n + 1)
            poly = np.int32([[px(x), px(y)] for x, y in pts])
            th = max(1, int(R * rng.uniform(0.15, 0.5)))
            cv2.polylines(L["A"], [poly], False, 1.0, th, cv2.LINE_AA)
        L["T"][:] = rng.uniform(0.75, 1.05)
        info = {}
    elif category == "finder_alteration":
        fr, fc = [(0, 0), (0, n - 7), (n - 7, 0)][rng.integers(3)]
        if rng.random() < 0.5:   # paper over part of the finder
            h, w = rng.uniform(2.5, 6, 2)
            r0, c0 = fr + rng.uniform(0, 7 - h), fc + rng.uniform(0, 7 - w)
            paper_patch(L, r0, c0, r0 + h, c0 + w, rng)
        else:                    # ink over the finder's light ring
            ring = [(fr + i, fc + j) for i in range(1, 6) for j in range(1, 6) if not matrix[fr + i, fc + j]]
            fill_modules(L, ring[:int(rng.integers(3, len(ring) + 1))], rng.uniform(-0.05, 0.1), 0.05, rng)
        info = {}
    return L, info


def module_homography(corners, n):
    """canvas px -> image px, given the 4 symbol corners (TL, TR, BR, BL) in the image."""
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    H = cv2.getPerspectiveTransform(src, np.float32(corners))
    to_mod = np.array([[1 / R, 0, -MARGIN], [0, 1 / R, -MARGIN], [0, 0, 1]], np.float64)
    return H @ to_mod


def tone_fields(img, mod_px):
    """Local paper (light) and ink (dark) levels, per channel."""
    k = max(3, int(mod_px * 1.6) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sig = max(1.0, mod_px * 2)
    light = cv2.GaussianBlur(cv2.dilate(img, ker), (0, 0), sig).astype(np.float32)
    dark = cv2.GaussianBlur(cv2.erode(img, ker), (0, 0), sig).astype(np.float32)
    return light, dark


def composite(img, layer, H, light, dark, edge_sigma, noise_sigma, rng):
    h, w = img.shape[:2]
    A = cv2.warpPerspective(layer["A"], H, (w, h), flags=cv2.INTER_LINEAR)
    T = cv2.warpPerspective(layer["T"], H, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    S = cv2.warpPerspective(layer["S"], H, (w, h), flags=cv2.INTER_LINEAR)
    if edge_sigma > 0.3:
        A, T = cv2.GaussianBlur(A, (0, 0), edge_sigma), cv2.GaussianBlur(T, (0, 0), edge_sigma)
    color = (dark + T[..., None] * (light - dark)) * layer["tint"][None, None, :]
    color += rng.normal(0, noise_sigma, color.shape).astype(np.float32)
    out = img.astype(np.float32) * (1 - S[..., None])
    out = out * (1 - A[..., None]) + color * A[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ simulated captures

def background(rng, h, w):
    bg = np.zeros((h, w, 3), np.float32)
    c0, c1 = rng.uniform(30, 230, 3), rng.uniform(30, 230, 3)
    t = np.linspace(0, 1, w)[None, :, None] if rng.random() < 0.5 else np.linspace(0, 1, h)[:, None, None]
    bg[:] = c0 + (c1 - c0) * t
    for _ in range(int(rng.integers(3, 15))):  # clutter: boxes and lines
        col = tuple(float(x) for x in rng.uniform(0, 255, 3))
        p1 = tuple(int(x) for x in (rng.integers(0, w), rng.integers(0, h)))
        p2 = tuple(int(x) for x in (rng.integers(0, w), rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.rectangle(bg, p1, p2, col, -1)
        else:
            cv2.line(bg, p1, p2, col, int(rng.integers(1, 6)))
    bg += cv2.GaussianBlur(rng.normal(0, rng.uniform(2, 12), (h, w, 3)).astype(np.float32), (0, 0), 1.5)
    return np.clip(bg, 0, 255)


def capture_params(rng):
    h, w = (600, 800) if rng.random() < 0.6 else (800, 600)
    return {
        "h": h, "w": w, "scale": float(rng.uniform(0.3, 0.65)), "persp": float(rng.uniform(0, 0.18)),
        "rot": float(rng.uniform(0, 360)), "quiet": float(rng.uniform(1.5, 4)),
        "paper": float(rng.uniform(200, 250)), "ink": float(rng.uniform(10, 70)),
        "light_grad": float(rng.uniform(0, 0.45)), "blur": float(rng.uniform(0, 1.6)),
        "noise": float(rng.uniform(1, 7)), "jpeg": int(rng.integers(60, 96)),
        "bg_seed": int(rng.integers(1 << 30)), "light_seed": int(rng.integers(1 << 30)),
    }


def simulate(matrix, p, tamper, rng):
    """Render one capture. tamper: layer or None. Same p => same pose, lighting and camera."""
    n = len(matrix)
    img = background(np.random.default_rng(p["bg_seed"]), p["h"], p["w"])
    side = p["scale"] * min(p["h"], p["w"])
    ctr = np.array([p["w"] / 2, p["h"] / 2]) + np.random.default_rng(p["bg_seed"] + 1).uniform(-0.15, 0.15, 2) * side
    sq = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    a = np.deg2rad(p["rot"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    jit = np.random.default_rng(p["bg_seed"] + 2).uniform(-p["persp"], p["persp"], (4, 2)) * side
    corners = (sq @ rot.T) + jit + ctr
    H = module_homography(corners, n)
    paper, ink = np.full((p["h"], p["w"], 3), p["paper"], np.float32), np.full((p["h"], p["w"], 3), p["ink"], np.float32)

    sheet = blank_layer(n)
    paper_patch(sheet, -p["quiet"], -p["quiet"], n + p["quiet"], n + p["quiet"], rng, shadow=False)
    for (r, c), dark in np.ndenumerate(matrix):
        if dark:
            sheet["T"][px(r):px(r + 1), px(c):px(c + 1)] = 0.0
    img = composite(img, sheet, H, paper, ink, 0.6, 0, rng)
    if tamper is not None:
        img = composite(img, tamper, H, paper, ink, 0.6, 0, rng)

    lr = np.random.default_rng(p["light_seed"])   # lighting gradient + soft shadow blob
    yy, xx = np.mgrid[0:p["h"], 0:p["w"]] / max(p["h"], p["w"])
    d = lr.uniform(-1, 1, 2)
    gain = 1 - p["light_grad"] * ((xx * d[0] + yy * d[1]) - (xx * d[0] + yy * d[1]).min())
    if lr.random() < 0.4:
        cx, cy, s = lr.uniform(0, 1), lr.uniform(0, 1), lr.uniform(0.1, 0.4)
        gain *= 1 - 0.35 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    img = img.astype(np.float32) * gain[..., None]
    if p["blur"] > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), p["blur"])
    img += lr.normal(0, p["noise"], img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), corners


# ------------------------------------------------------------------ real photographs

def normalize_photo(img, max_side=1600):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def jpeg_roundtrip(img, q=92):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def photo_geometry(img):
    """(corners, matrix, payload) from OpenCV on the untouched photo, or None if not decodable."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data, pts, straight = cv2.QRCodeDetector().detectAndDecode(gray)
    if not data or pts is None or straight is None or straight.shape[0] < 21:
        return None
    return pts.reshape(4, 2), straight < 128, data


def decode(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    except cv2.error:
        data = ""
    return data or ""


# ------------------------------------------------------------------ build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", type=int, default=1000)
    ap.add_argument("--physical", default=None, help="captures.csv from dataprep/COLLECTION_PROTOCOL.md")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(BENCH, "images"), exist_ok=True)
    rows = []

    def save(img, sid, **meta):
        rel = f"images/{sid}.jpg"
        cv2.imwrite(os.path.join(BENCH, rel), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        observed = decode(img)
        rows.append({"sample_id": sid, "image_path": rel, "observed_payload": observed,
                     "decodes": int(bool(observed)), **meta})

    # Simulated identities, split by identity before anything is generated.
    ids = np.arange(args.identities)
    rng.shuffle(ids)
    n_tr, n_va = int(0.7 * len(ids)), int(0.15 * len(ids))
    split_of = {int(i): ("train" if k < n_tr else "val" if k < n_tr + n_va else "test_synthetic")
                for k, i in enumerate(ids)}
    cat_cycle = 0
    for ident in range(args.identities):
        payload = random_payload(rng)
        matrix, version, level = qr_matrix(payload, rng)
        group = f"syn{ident:04d}"
        for cap in range(2 if rng.random() < 0.5 else 1):
            p = capture_params(rng)
            cat = CATEGORIES[cat_cycle % len(CATEGORIES)]
            cat_cycle += 1
            layer, info = tamper_layer(cat, matrix, version, level, rng)
            common = dict(split=split_of[ident], source="simulated", subset="simulated", group_id=group,
                          pair_id=f"{group}_c{cap}", original_payload=payload, version=version,
                          ecc_level=level, capture=json.dumps(p), annotation_confidence="high (constructed)",
                          license="generated (this repository)")
            g_img, _ = simulate(matrix, p, None, np.random.default_rng(p["bg_seed"] + 7))
            t_img, _ = simulate(matrix, p, layer, np.random.default_rng(p["bg_seed"] + 7))
            save(g_img, f"{group}_c{cap}_g", label=0, category="genuine", tamper_info="{}", **common)
            save(t_img, f"{group}_c{cap}_t", label=1, category=cat, tamper_info=json.dumps(info), **common)
        if ident % 100 == 0:
            print("simulated", ident)

    # Real photographs.
    real_sources = [
        ("test_real", "mendeley_surfaces",
         sorted(glob.glob(os.path.join(EXT, "mendeley_surfaces", "flat", "**", "*.jpg"), recursive=True)
                + glob.glob(os.path.join(EXT, "mendeley_surfaces", "random", "**", "*.jpg"), recursive=True)),
         "CC BY 4.0, doi:10.17632/m6mfwc52vk.1"),
        ("test_external", "szeged_huawei",
         sorted(glob.glob(os.path.join(EXT, "szeged_huawei", "huawei", "*.jpg"))),
         "no explicit licence; citation requested (doi:10.1007/s10044-017-0619-6); local use only"),
    ]
    skipped = {}
    for split, source, files, lic in real_sources:
        files = [f for f in files if os.sep + "bitmaps" + os.sep not in f]
        skipped[source] = 0
        for i, path in enumerate(files):
            img = normalize_photo(cv2.imread(path))
            geo = photo_geometry(img)
            stem = os.path.splitext(os.path.relpath(path, os.path.join(EXT, source)))[0]
            sid = f"{source}_{i:04d}"
            if geo is None:
                # Still a genuine capture: kept for specificity, but without module geometry no
                # tampered counterpart can be constructed.
                skipped[source] += 1
                save(jpeg_roundtrip(img), f"{sid}_g", label=0, category="genuine", tamper_info="{}",
                     split=split, source=source, subset=stem.split(os.sep)[0],
                     group_id=f"{source}:undecodable", pair_id=f"{source}:{stem}", original_payload="",
                     version="", ecc_level="", capture=json.dumps({"file": stem.replace(os.sep, "/")}),
                     annotation_confidence="genuine: as published by dataset authors", license=lic)
                continue
            corners, matrix, payload = geo
            n = len(matrix)
            version = (n - 17) // 4
            mod_px = float(np.linalg.norm(corners[1] - corners[0]) / n)
            light, dark = tone_fields(img, mod_px)
            cat = CATEGORIES[cat_cycle % len(CATEGORIES)]
            cat_cycle += 1
            layer, info = tamper_layer(cat, matrix, version, None, rng)
            noise = float(np.std(img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 2).astype(np.float32)))
            tampered = composite(img, layer, module_homography(corners, n), light, dark,
                                 edge_sigma=float(np.clip(0.12 * mod_px, 0.5, 3)),
                                 noise_sigma=min(noise, 6), rng=rng)
            group = f"{source}:{hashlib.sha1(payload.encode()).hexdigest()[:10]}"
            common = dict(split=split, source=source, subset=stem.split(os.sep)[0], group_id=group,
                          pair_id=f"{source}:{stem}", original_payload=payload, version=version,
                          ecc_level="", capture=json.dumps({"file": stem.replace(os.sep, "/")}),
                          annotation_confidence="genuine: as published by dataset authors; "
                                                "tampered: high (constructed)", license=lic)
            save(jpeg_roundtrip(img), f"{sid}_g", label=0, category="genuine", tamper_info="{}", **common)
            save(jpeg_roundtrip(tampered), f"{sid}_t", label=1, category=cat, tamper_info=json.dumps(info), **common)
        print(source, "photos:", len(files), "genuine-only (not decodable, no tampered pair):", skipped[source])

    # Legacy synthetic splits (original data/val, data/test), scored in place for continuity.
    for sp in ("val", "test"):
        for lab, cls in ((0, "genuine"), (1, "tampered")):
            for fn in sorted(os.listdir(os.path.join(DATA, sp, cls))):
                num = fn.split("_")[1].split(".")[0]
                rows.append({"sample_id": f"legacy_{sp}_{cls}_{num}", "image_path": f"../{sp}/{cls}/{fn}",
                             "observed_payload": "", "decodes": "", "label": lab,
                             "category": "genuine" if lab == 0 else "legacy_synthetic",
                             "tamper_info": "{}", "split": f"legacy_{sp}", "source": "legacy_synthetic",
                             "subset": sp, "group_id": f"legacy:{num}", "pair_id": f"legacy:{num}",
                             "original_payload": f"https://example.com/page/{int(num) - 1}", "version": 3,
                             "ecc_level": "M", "capture": "{}", "annotation_confidence": "high (constructed)",
                             "license": "generated (this repository)"})

    if args.physical:   # dataprep/COLLECTION_PROTOCOL.md
        with open(args.physical, newline="") as f:
            for cap in csv.DictReader(f):
                rel = os.path.relpath(os.path.join(os.path.dirname(args.physical), "images", cap["image_id"]), BENCH)
                rows.append({"sample_id": f"physical_{os.path.splitext(cap['image_id'])[0]}", "image_path": rel,
                             "observed_payload": "", "decodes": "", "label": int(cap["label"] == "tampered"),
                             "category": cap["tampering_category"] or "genuine", "tamper_info": "{}",
                             "split": "test_physical", "source": "physical", "subset": cap["session_id"],
                             "group_id": f"physical:{cap['original_id']}",
                             "pair_id": f"physical:{cap['original_id']}:{cap['session_id']}",
                             "original_payload": "", "version": "", "ecc_level": "",
                             "capture": json.dumps(cap), "annotation_confidence": "high (physical, recorded)",
                             "license": "collected by the repository owner"})

    with open(os.path.join(BENCH, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(BENCH, "build_info.json"), "w") as f:
        json.dump({"seed": SEED, "identities": args.identities, "genuine_only_real": skipped,
                   "n_rows": len(rows)}, f, indent=2)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
