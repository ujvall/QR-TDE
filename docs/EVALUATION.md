# Evaluation: detectors, real-world transfer and the promotion decision

This document records how Q-TDE's physical-tampering detectors were compared, what was promoted
to the active pipeline and why. Every number below is reproduced by the commands at the end; the
full machine-generated tables are in [`results/benchmark/REPORT.md`](../results/benchmark/REPORT.md)
and [`results/benchmark/report.json`](../results/benchmark/report.json).

## 1. Original baseline, re-verified

Running `python -m fusion.evaluate_decision_engine --split all` on the original synthetic splits
reproduces the previously reported figures exactly:

| Split (15 genuine / 15 tampered) | Sensitivity | Specificity | Accuracy |
|---|---|---|---|
| Validation | 13.33 % (2/15) | 100 % | 56.67 % |
| Test | 13.33 % (2/15) | 100 % | 56.67 % |

The original ViT checkpoint scores 50 % accuracy on the old test split (confusion matrix
`[[10 5] [10 5]]`), and the original autoencoder's mean reconstruction error is 0.117 for both
classes.

The audit turned up four problems with that evaluation:

- **Only digital Version 3-M images.** Every image is a perfect render of
  `https://example.com/page/N`. Nothing tests photographs, other versions or other ECC levels.
- **Near-identical symbols across splits.** Sequential payloads give visually near-identical codes
  in validation and test (thumbnail correlation 0.84 for `/page/77` vs `/page/75`).
- **Most "tampering" leaves modules intact.** Blur, ±25 % brightness, mild noise and shadow
  gradients don't change any module, so neither ECC nor anything else can detect them. The low
  sensitivity measures that, as much as it measures the detectors.
- **The peak threshold τ = 0.525490 was fixed on the same synthetic renders,** with no separate
  calibration split.

## 2. What was evaluated

**Data.** The [benchmark](../dataprep/DATASET_CARD.md) has a simulated capture set (1 000 QR
identities, versions 1–8, ECC L/M/Q/H, seven tampering categories, split by identity), plus two
real-photo test sets from independent public sources:

- **Mendeley:** 100 photos of one printed V7-H code on flat and curved surfaces.
- **Szeged:** 98 smartphone photos of V4-L codes.

Tampering on the real photos is applied digitally in module space, with blur and noise matched to
the photo.

**Detectors.** All detectors see identical images and splits.

| Detector | Type | Input |
|---|---|---|
| Original engine | V3-M syndrome rule OR peak residual (τ = 0.5255), as deployed before this work | full image via `QRInspector` |
| Peak residual | the engine's high-frequency residual, threshold recalibrated | full image |
| **RS correction load** (new) | version-generic Reed-Solomon decoding of OpenCV's sampled module grid; score = largest share of any block's correction capacity used | full image |
| ResNet-18 | ImageNet-pretrained, fully fine-tuned, 15 epochs max, early stopping on validation AUC | 224 px rectified crop |
| ViT-B/16 fine-tune | `google/vit-base-patch16-224`, fully fine-tuned with its own normalization (mean = std = 0.5), AdamW 2e-5, wd 0.05, one-cycle schedule, bf16 | 224 px rectified crop |
| ViT-B/16 probe | same backbone frozen, logistic regression on the CLS token (C chosen on validation) | 224 px rectified crop |
| Conv autoencoder | one-class, genuine-only training, 8×8×32 bottleneck, L1 + SSIM loss; score = max of smoothed local error | 128 px rectified crop |
| PatchCore-style | one-class; ImageNet ResNet-18 layer2+3 patch features of genuine crops, 10 % random memory bank; score = max nearest-neighbour distance | 224 px rectified crop |

The rectified crop comes from OpenCV's own detection of each image (never ground-truth corners).
Unlocalized images get no score and count as "no alarm", so every number is a system-level result.

**Threshold rule (fixed before any test split was scored).** For each continuous detector, the
threshold is the lowest score that keeps **validation** specificity ≥ 95 %. Verdict detectors are
evaluated as deployed. No threshold was tuned on a test split.

**Statistics.** 95 % bootstrap CIs resample whole QR identities, or source photos where a split
has fewer than 30 identities (`test_real`). Recall gains over the original engine are paired
bootstrap differences on the same tampered samples.

## 3. Results

Recall and specificity in %, 95 % CI in brackets.

| Detector | Synthetic test (225/225) | Real, Mendeley (100/34) | External, Szeged (98/76) |
|---|---|---|---|
| Original engine | 15.1 / 85.3 | 0.0 / 100 | 0.0 / 100 |
| **Updated engine (original + RS load)** | **45.8** [39.0–52.4] / 81.8 | **23.5** [10.0–38.7] / **99.0** | **44.7** [33.8–55.4] / **89.8** [80.5–95.9] |
| RS correction load alone | 39.6 / 96.0 | 23.5 / 99.0 | 44.7 / 89.8 |
| Peak residual (recalibrated) | 8.0 / 93.8 | 20.6 / 78.0 | 0.0 / 100 |
| ResNet-18 | 56.0 / 92.4 | 44.1 / **70.0** | 85.5 / **20.4** |
| ViT-B/16 fine-tune | 59.6 / 93.3 | 79.4 / **39.0** | 86.8 / **25.5** |
| ViT-B/16 probe | 40.4 / 92.9 | 47.1 / 79.0 | 52.6 / 66.3 |
| Conv autoencoder | 4.4 / 94.7 | 14.7 / 79.0 | 1.3 / 99.0 |
| PatchCore-style | 6.7 / 96.9 | 20.6 / 58.0 | 42.1 / 57.1 |

Threshold-free ranking quality (ROC-AUC):

| Detector | Synthetic | Real | External |
|---|---|---|---|
| RS correction load | 0.625 | 0.864 | 0.605 |
| ResNet-18 | 0.737 | 0.596 | 0.753 |
| ViT-B/16 fine-tune | 0.757 | 0.713 | 0.818 |
| ViT-B/16 probe | 0.662 | 0.652 | 0.610 |
| Conv autoencoder | 0.446 | 0.475 | 0.508 |
| PatchCore-style | 0.522 | 0.426 | 0.486 |

The updated engine's recall gain over the original engine, paired bootstrap:

| Split | Gain | 95 % CI |
|---|---|---|
| Synthetic test | +30.7 pp | +24.1 to +37.4 |
| Real (Mendeley) | +23.5 pp | +11.8 to +38.2 |
| External (Szeged) | +44.7 pp | +34.2 to +55.3 |

### Per tampering category (updated engine vs. ViT fine-tune, recall)

| Category | Synthetic | Real | External |
|---|---|---|---|
| `label_overlay` | 78 % vs 100 % | 5/5 vs 5/5 | 45 % vs 100 % |
| `marker_edit` | 68 % vs 59 % | 1/5 vs 5/5 | 45 % vs 100 % |
| `white_out` | 53 % vs 53 % | 2/5 vs 4/5 | 82 % vs 100 % |
| `module_patch` | 66 % vs 26 % | 0/5 vs 5/5 | 73 % vs 100 % |
| `scratch` | 29 % vs 71 % | 0/5 vs 4/5 | 55 % vs 100 % |
| `sticker_replacement` | 14 % vs 93 % | 0/4 vs 4/4 | 0 % vs 73 % |
| `finder_alteration` | 0 % vs 21 % | 0/5 vs 0/5 | 10 % vs 30 % |

Real-split categories have 4–5 samples each; treat them as indications only.

### Ablation of the deployed decision logic

| Variant | Synthetic rec / spec | Real rec / spec | External rec / spec | Legacy test rec / spec |
|---|---|---|---|---|
| Original engine (V3-M rule + peak) | 15.1 / 85.3 | 0.0 / 100 | 0.0 / 100 | 13.3 / 100 |
| + RS correction load (**deployed**) | 45.8 / 81.8 | 23.5 / 99.0 | 44.7 / 89.8 | 13.3 / 100 |
| RS load + peak, V3-M binary rule removed | 39.6 / 96.0 | 23.5 / 99.0 | 44.7 / 89.8 | 13.3 / 100 |

Re-scoring the integrated engine end to end (`experiments.classical --engine-only
integrated_engine`) reproduces the ablation row exactly on every test split.

### Cost

CPU, 8 threads, median of 50 real test images, batch size 1 (`results/benchmark/latency.json`):

| Component | Latency | Size |
|---|---|---|
| Full inspection, original engine | 108 ms | – |
| RS correction load (includes the OpenCV decode) | 91 ms | no parameters |
| Rectified crop for learned models | 62 ms | – |
| ResNet-18 forward | 39 ms | 11.2 M parameters, 43 MB |
| ViT-B/16 forward | 240 ms | 85.8 M parameters, 327 MB |
| Conv autoencoder forward | 7 ms | 0.86 M parameters, 3.3 MB |

## 4. Why the learned models were not promoted

**The supervised classifiers learn transferable cues but lose calibration on real photos.** The
ViT fine-tune ranks the external photos well (AUC 0.82). It catches sticker replacements, which
no ECC method can see, because a replacement code is itself valid. At the threshold calibrated on
validation, however, it flags **75 % of genuine Szeged photos and 61 % of genuine Mendeley
photos**. ResNet-18 behaves the same way. Their score distributions shift upwards on real
captures (sensor texture, print grain, surface curvature), and no labelled real-photo validation
set exists to recalibrate on without touching the test data. A detector that alarms on most
genuine codes is unusable, so neither model was promoted.

**The ViT result does not repeat the original failure.** The original run trained the top two
blocks on 140 digital images whose "tampering" barely changed anything, and its checkpoint
scored at chance. With 2 128 training images, correct normalization, full fine-tuning,
augmentation and early stopping, the same architecture reaches validation AUC 0.885 on localized
crops (0.71 system-level, counting unlocalized images as misses) and synthetic test AUC 0.76.
The pretrained backbone does help: a frozen-feature linear probe already reaches 0.78 on
validation crops and 0.66 on the synthetic test. The bottleneck is domain transfer, not the architecture.

**Reconstruction-based one-class models cannot work here, and the old autoencoder failed for the
same reason.** A QR data region is deliberately pseudo-random: masked Reed-Solomon codewords.
"Normal" therefore has no local regularity for a model to learn, and a tampered module pattern
looks exactly as plausible as a genuine one. The redesigned autoencoder does not collapse.
Reconstruction pixel standard deviation matches the input's (0.238 vs 0.231 at the final epoch) and finders and quiet zone are
reproduced (`results/benchmark/figures/conv_ae_reconstructions.png`). Even so, its error map is
uniform over the data region for both classes (AUC 0.45–0.51). PatchCore-style patch statistics
fail for the same reason (AUC 0.43–0.52).

## 5. What was promoted, and why

**Promotion criterion.** This was written down after the interim ResNet-18 and RS results were
visible, and it follows the project brief. A component is promoted only if both conditions hold:

1. Its paired recall gain over the original engine has a 95 % CI above zero on the synthetic
   test, real and external splits.
2. Its specificity on each real split is not significantly below the 95 % target: the CI upper
   bound is ≥ 95 %.

**The RS correction-load check meets both conditions**, so it was integrated as an
**additional, independent evidence component** (`ecc_capacity_exceeded`):

- It runs for every QR version and ECC level. Previously, every non-V3-M code, which includes
  every real photo in the benchmark, received no ECC analysis at all.
- It is skipped (status `unavailable`, never "clean") when the payload cannot be decoded from the
  original image.
- It cannot influence URL analysis. That is tested in `fusion/test_rs_capacity_integration.py`.

The learned models fail criterion 2 by a wide margin and remain experimental.

**What was deliberately not changed.** The ablation shows that the original V3-M "any non-zero
syndrome" rule costs 10–14 pp of specificity on the simulated validation and test splits. Its
false alarms come from photographed V3-M codes, where one mis-sampled module trips it. Its only measured benefit is catching sub-module digital edits on the legacy
images. Those edits change single centre pixels without changing any module's majority colour, so
the area-based RS check does not count them. Removing the rule is a behavioural change to the
existing product, so it is left in place and reported here for a decision rather than changed
silently.

## 6. Limitations of this evaluation

- Real-photo tampering is **digital**, so the real splits measure transfer to real capture
  conditions, not detection of physical attacks. `dataprep/COLLECTION_PROTOCOL.md` defines the
  physical test set that would close this gap.
- `test_real` contains one physical code, and each category on the real splits has 4–11 samples.
- On the external set the RS check's specificity (89.8 %) sits below the 95 % calibration target.
  Real photos of low-ECC (L) codes use more correction capacity than simulated captures.
- The threshold was calibrated on simulated captures only. A labelled set of genuine real photos
  for calibration would most likely improve all detectors, the learned ones most of all.

## Reproduce

```powershell
python -m dataprep.acquire
python -m dataprep.build_benchmark
python -m dataprep.validate_benchmark
python -m experiments.classical
python -m experiments.classical --engine-only integrated_engine
python -m experiments.learned          # GPU recommended; see requirements-train.txt
python -m experiments.benchmark
python -m experiments.latency
```

Seeds are fixed (`SEED = 42`). Training metadata, per-epoch curves and hyperparameters are written
to `results/benchmark/training_metadata.json`.
