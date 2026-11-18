"""
CPU inference latency and model size for every detector (deployment target: CPU Streamlit).

    python -m experiments.latency      # run in an environment with torch + transformers

Writes results/benchmark/latency.json. Latency = median over 50 test images, batch size 1.
"""

import os
import statistics
import time

import cv2
import numpy as np
import torch

from app.inspector import QRInspector
from ecc.rs_generic import analyze_matrix
from experiments.common import ROOT, image_path, load_manifest, rectified_crop, save_json

CKPT = os.path.join(ROOT, "models", "benchmark")


def median_ms(fn, items, warmup=3):
    for it in items[:warmup]:
        fn(it)
    ts = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        ts.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ts), 1)


def main():
    torch.set_num_threads(os.cpu_count())
    rows = [r for r in load_manifest() if r["split"] in ("test_real", "test_external")][:50]
    imgs = [cv2.imread(image_path(r)) for r in rows]
    grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    crops = [c for c in (rectified_crop(g) for g in grays) if c is not None]
    out = {"images": len(imgs), "device": "cpu", "threads": os.cpu_count()}

    insp, det = QRInspector(), cv2.QRCodeDetector()
    out["baseline_engine_full_inspection_ms"] = median_ms(lambda im: insp.inspect_image(image=im), imgs)

    def rs(g):
        data, _, st = det.detectAndDecode(g)
        return analyze_matrix(st < 128) if data and st is not None else None
    out["rs_capacity_ms (incl. OpenCV decode)"] = median_ms(rs, grays)
    out["rectified_crop_ms"] = median_ms(rectified_crop, grays)

    from experiments import learned
    x = torch.from_numpy(np.stack(crops[:1]).astype(np.float32) / 255)[:, None].repeat(1, 3, 1, 1)
    models = {}
    for name in ("resnet18", "vit_finetune"):
        m, _, fwd, _ = learned.build_classifier(name)
        m.load_state_dict(torch.load(os.path.join(CKPT, f"{name}.pt"), map_location="cpu"))
        models[name] = (m.eval(), fwd)
    ae = learned.ConvAE()
    ae.load_state_dict(torch.load(os.path.join(CKPT, "conv_ae.pt"), map_location="cpu"))
    with torch.no_grad():
        for name, (m, fwd) in models.items():
            out[f"{name}_forward_ms"] = median_ms(lambda c: fwd(m, x), crops[:30])
        ae.eval()
        out["conv_ae_forward_ms"] = median_ms(lambda c: ae(x[:, :1, ::2, ::2].contiguous()[:, :, :128, :128]), crops[:30])
    sizes = {}
    for name in ("resnet18", "vit_finetune", "conv_ae"):
        p = os.path.join(CKPT, f"{name}.pt")
        params = sum(v.numel() for v in torch.load(p, map_location="cpu").values())
        sizes[name] = {"parameters": int(params), "checkpoint_mb": round(os.path.getsize(p) / 2 ** 20, 1)}
    sizes["rs_capacity"] = {"parameters": 0, "checkpoint_mb": 0}
    sizes["patchcore"] = {"parameters": "ResNet-18 backbone (11.7 M) + memory bank of genuine patch features",
                          "checkpoint_mb": "bank not persisted"}
    out["model_size"] = sizes
    save_json(os.path.join(ROOT, "results", "benchmark", "latency.json"), out)
    print(out)


if __name__ == "__main__":
    main()
