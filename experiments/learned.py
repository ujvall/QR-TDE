"""
Train and score the learned tampering detectors on the benchmark.

All models see the same input: the QR crop rectified from the image's *own* OpenCV detection
(experiments.common.rectified_crop), 224x224 grayscale. Images where nothing is localized get
no score (available = 0, never flagged). Model selection and early stopping use the validation
split only; test splits are scored once with the selected checkpoint.

  resnet18        ImageNet ResNet-18, fully fine-tuned (supervised)
  vit_finetune    google/vit-base-patch16-224, fully fine-tuned with its own normalization
  vit_probe       same ViT frozen; logistic regression on the CLS embedding
  conv_ae         one-class convolutional autoencoder trained on genuine crops only;
                  score = max of the smoothed local reconstruction error
  patchcore       PatchCore-style: ImageNet ResNet-18 layer2+3 patch features of genuine train
                  crops form a memory bank; score = max nearest-neighbour distance over patches

    python -m experiments.learned --models resnet18 vit_finetune vit_probe conv_ae patchcore
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

from experiments.common import BENCH, ROOT, SEED, image_path, load_manifest, rectified_crop, write_scores

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(ROOT, "results", "benchmark")
CKPT = os.path.join(ROOT, "models", "benchmark")  # git-ignored
TRAIN, VAL = "train", "val"


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_crops(rows):
    """Cache rectified crops once: data/benchmark/crops.npz (uint8 Nx224x224, available mask)."""
    cache = os.path.join(BENCH, "crops.npz")
    ids = [r["sample_id"] for r in rows]
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == ids:
            return z["x"], z["ok"]
    x = np.full((len(rows), 224, 224), 255, np.uint8)
    ok = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        crop = rectified_crop(cv2.imread(image_path(r), cv2.IMREAD_GRAYSCALE))
        if crop is not None:
            x[i], ok[i] = crop, True
        if i % 500 == 0:
            print("crops", i, "/", len(rows))
    np.savez_compressed(cache, x=x, ok=ok, ids=np.array(ids))
    return x, ok


def augment(batch):
    """Label-preserving augmentation on uint8 crops (B,H,W): rot90, small shift, contrast, blur, noise."""
    out = []
    for im in batch:
        im = np.rot90(im, random.randint(0, 3)).astype(np.float32)
        M = np.float32([[1, 0, random.uniform(-6, 6)], [0, 1, random.uniform(-6, 6)]])
        im = cv2.warpAffine(im, M, im.shape[::-1], borderValue=255)
        im = im * random.uniform(0.7, 1.15) + random.uniform(-25, 25)
        if random.random() < 0.4:
            im = cv2.GaussianBlur(im, (0, 0), random.uniform(0.3, 1.4))
        im += np.random.normal(0, random.uniform(0, 6), im.shape)
        out.append(np.clip(im, 0, 255))
    return np.stack(out)


def to_tensor(batch, mean, std):
    t = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32) / 255.0)[:, None]
    return ((t - mean) / std).repeat(1, 3, 1, 1)


# ------------------------------------------------------------------ supervised classifiers

def build_classifier(name):
    if name == "resnet18":
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m, (0.449, 0.226), lambda m, x: m(x), {"lr": 3e-4, "wd": 1e-4, "epochs": 15, "bs": 32}
    from transformers import ViTForImageClassification
    m = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2,
                                                  ignore_mismatched_sizes=True)
    # ViT-B/16 expects mean=std=0.5 normalization (preprocessor_config of this checkpoint).
    return m, (0.5, 0.5), lambda m, x: m(pixel_values=x).logits, {"lr": 2e-5, "wd": 0.05, "epochs": 8, "bs": 16}


def train_classifier(name, x, ok, y, split):
    seed_all()
    model, (mu, sd), fwd, hp = build_classifier(name)
    model.to(DEV)
    tr = np.flatnonzero((split == TRAIN) & ok)
    va = np.flatnonzero((split == VAL) & ok)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    steps = hp["epochs"] * int(np.ceil(len(tr) / hp["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps, pct_start=0.1)
    best, best_state, history, patience = -1, None, [], 0
    for ep in range(hp["epochs"]):
        model.train()
        np.random.shuffle(tr)
        tl, t0 = 0.0, time.time()
        for i in range(0, len(tr), hp["bs"]):
            b = tr[i:i + hp["bs"]]
            xb = to_tensor(augment(x[b]), mu, sd).to(DEV)
            with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                loss = F.cross_entropy(fwd(model, xb).float(), torch.from_numpy(y[b]).long().to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(b)
        va_scores = predict_classifier(model, fwd, x[va], mu, sd)
        va_loss = float(F.binary_cross_entropy(torch.tensor(va_scores).clamp(1e-6, 1 - 1e-6),
                                               torch.tensor(y[va], dtype=torch.float32)))
        auc = roc_auc_score(y[va], va_scores)
        history.append({"epoch": ep + 1, "train_loss": tl / len(tr), "val_loss": va_loss, "val_auc": auc,
                        "seconds": round(time.time() - t0, 1)})
        print(name, history[-1])
        if auc > best:
            best, patience = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:   # early stopping on validation AUC
                break
    model.load_state_dict(best_state)
    os.makedirs(CKPT, exist_ok=True)
    torch.save(best_state, os.path.join(CKPT, f"{name}.pt"))
    scores = np.full(len(y), np.nan)
    scores[ok] = predict_classifier(model, fwd, x[ok], mu, sd)
    return scores, {"hyperparameters": hp, "history": history, "best_val_auc": best}


@torch.no_grad()
def predict_classifier(model, fwd, xs, mu, sd, bs=64):
    model.eval()
    out = []
    for i in range(0, len(xs), bs):
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            logits = fwd(model, to_tensor(xs[i:i + bs], mu, sd).to(DEV)).float()
        out.append(torch.softmax(logits, 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def vit_probe(x, ok, y, split):
    from transformers import ViTModel
    seed_all()
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEV).eval()
    feats = np.zeros((len(y), 768), np.float32)
    idx = np.flatnonzero(ok)
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            feats[b] = vit(pixel_values=to_tensor(x[b], 0.5, 0.5).to(DEV)).last_hidden_state[:, 0].cpu().numpy()
    tr, va = (split == TRAIN) & ok, (split == VAL) & ok
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):   # C chosen on validation AUC
        clf = LogisticRegression(C=C, max_iter=5000).fit(feats[tr], y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(feats[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    scores = np.full(len(y), np.nan)
    scores[ok] = best[2].predict_proba(feats[ok])[:, 1]
    return scores, {"C": best[1], "best_val_auc": best[0]}


# ------------------------------------------------------------------ one-class models

class ConvAE(nn.Module):
    """128x128 -> 8x8x32 bottleneck; narrow enough that it cannot learn the identity map."""

    def __init__(self):
        super().__init__()
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc = nn.Sequential(down(1, 32), down(32, 64), down(64, 128), down(128, 128), nn.Conv2d(128, 32, 1))
        self.dec = nn.Sequential(nn.Conv2d(32, 128, 1), up(128, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def ssim_map(a, b, w=7):
    mu_a, mu_b = F.avg_pool2d(a, w, 1, w // 2), F.avg_pool2d(b, w, 1, w // 2)
    va = F.avg_pool2d(a * a, w, 1, w // 2) - mu_a ** 2
    vb = F.avg_pool2d(b * b, w, 1, w // 2) - mu_b ** 2
    cov = F.avg_pool2d(a * b, w, 1, w // 2) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def ae_batch(xs, aug):
    xs = augment(xs) if aug else xs.astype(np.float32)
    xs = np.stack([cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA) for im in xs])
    return torch.from_numpy(xs / 255.0).float()[:, None].to(DEV)


def conv_ae(x, ok, y, split):
    seed_all()
    tr = np.flatnonzero((split == TRAIN) & ok & (y == 0))   # genuine only
    va_g = np.flatnonzero((split == VAL) & ok & (y == 0))
    model = ConvAE().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-5)
    history, best, best_state = [], 1e9, None
    for ep in range(40):
        model.train()
        np.random.shuffle(tr)
        tl = 0.0
        for i in range(0, len(tr), 32):
            xb = ae_batch(x[tr[i:i + 32]], True)
            rec = model(xb)
            loss = F.l1_loss(rec, xb) + (1 - ssim_map(rec, xb)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            xb = ae_batch(x[va_g], False)
            rec = model(xb)
            vl = float(F.l1_loss(rec, xb) + (1 - ssim_map(rec, xb)).mean())
            # collapse check: a model that outputs the mean image has variance ~0
            rec_std = float(rec.std(dim=(2, 3)).mean())
        history.append({"epoch": ep + 1, "train_loss": tl / len(tr), "val_genuine_loss": vl,
                        "recon_pixel_std": rec_std, "input_pixel_std": float(xb.std(dim=(2, 3)).mean())})
        if ep % 5 == 0:
            print("conv_ae", history[-1])
        if vl < best:   # selection on genuine validation reconstruction only (no labels used)
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    os.makedirs(CKPT, exist_ok=True)
    torch.save(best_state, os.path.join(CKPT, "conv_ae.pt"))
    scores = np.full(len(y), np.nan)
    idx = np.flatnonzero(ok)
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            xb = ae_batch(x[b], False)
            rec = model(xb)
            err = (0.5 * (xb - rec).abs() + 0.5 * (1 - ssim_map(rec, xb)).clamp(0, 2) / 2)
            err = F.avg_pool2d(err, 9, 1, 4)   # localized error, smoothed
            scores[b] = err.flatten(1).max(1).values.cpu().numpy()
    save_ae_examples(model, x, y, split, ok)
    return scores, {"epochs": 40, "best_val_genuine_loss": best, "history": history}


def save_ae_examples(model, x, y, split, ok):
    """Figure: input / reconstruction / error for 4 genuine + 4 tampered validation crops."""
    rng = np.random.default_rng(SEED)
    pick = [rng.choice(np.flatnonzero((split == VAL) & ok & (y == lab)), 4, replace=False) for lab in (0, 1)]
    tiles = []
    with torch.no_grad():
        for b in pick:
            xb = ae_batch(x[b], False)
            rec = model(xb)
            err = (xb - rec).abs()
            for i in range(len(b)):
                row = [t[i, 0].cpu().numpy() for t in (xb, rec, err / max(float(err.max()), 1e-6))]
                tiles.append(np.hstack(row))
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    cv2.imwrite(os.path.join(OUT, "figures", "conv_ae_reconstructions.png"), (np.vstack(tiles) * 255).astype(np.uint8))


def patchcore(x, ok, y, split):
    import torchvision
    seed_all()
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(DEV).eval()

    def feats(xs):
        t = to_tensor(xs, 0.449, 0.226).to(DEV)
        h = net.maxpool(net.relu(net.bn1(net.conv1(t))))
        l2 = net.layer2(net.layer1(h))
        l3 = net.layer3(l2)
        f = torch.cat([l2, F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)], 1)
        f = F.avg_pool2d(f, 3, 1, 1)                       # local neighbourhood aggregation
        return f.permute(0, 2, 3, 1).reshape(len(xs), -1, f.shape[1])  # (B, 28*28, 384)

    tr = np.flatnonzero((split == TRAIN) & ok & (y == 0))
    bank = []
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for i in range(0, len(tr), 32):
            f = feats(x[tr[i:i + 32]]).reshape(-1, 384)
            bank.append(f[torch.randperm(len(f), generator=g)[: len(f) // 10]])  # ponytail: random 10 % subsample, coreset selection if the bank must shrink further
    bank = torch.cat(bank)
    scores = np.full(len(y), np.nan)
    idx = np.flatnonzero(ok)
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            b = idx[i:i + 16]
            f = feats(x[b])
            d = torch.stack([torch.cdist(fi, bank).min(1).values for fi in f])
            scores[b] = d.max(1).values.cpu().numpy()
    return scores, {"memory_bank_patches": int(len(bank)), "feature_dim": 384, "subsample": 0.1}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "vit_probe", "conv_ae", "patchcore", "vit_finetune"])
    args = ap.parse_args()
    rows = load_manifest()
    y = np.array([int(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    x, ok = load_crops(rows)
    print("device", DEV, "| crops localized:", int(ok.sum()), "/", len(ok))
    meta_path = os.path.join(OUT, "training_metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name in args.models:
        t0 = time.time()
        fn = {"resnet18": lambda: train_classifier("resnet18", x, ok, y, split),
              "vit_finetune": lambda: train_classifier("vit_finetune", x, ok, y, split),
              "vit_probe": lambda: vit_probe(x, ok, y, split),
              "conv_ae": lambda: conv_ae(x, ok, y, split),
              "patchcore": lambda: patchcore(x, ok, y, split)}[name]
        scores, info = fn()
        # CPU inference latency and model size are measured separately by experiments.latency.
        info.update({"train_and_score_seconds": round(time.time() - t0, 1), "device": str(DEV), "seed": SEED})
        meta[name] = info
        write_scores(name, [{"sample_id": r["sample_id"], "score": float(s) if np.isfinite(s) else 0.0,
                             "available": int(np.isfinite(s))} for r, s in zip(rows, scores)])
        os.makedirs(OUT, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(name, "done in", info["train_and_score_seconds"], "s")


if __name__ == "__main__":
    main()
