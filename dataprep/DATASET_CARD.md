# Dataset card: Q-TDE tampering benchmark v1

## Summary

The benchmark measures one question: given a photograph of a QR code, can a detector tell whether
the printed symbol was physically altered? It combines a simulated capture set (for training and
threshold calibration) with real photographs from two independent public sources (for testing
only). Physical tampering on the real photographs is applied digitally, because no public dataset
of photographed, physically tampered QR codes with a retained original could be found (see
*Research* below). `COLLECTION_PROTOCOL.md` describes how to close that gap with physical captures.

Build it with:

```powershell
python -m dataprep.acquire
python -m dataprep.build_benchmark
python -m dataprep.validate_benchmark
```

Images, crops and the manifest live in `data/benchmark/` and `data/external/` and are not committed.
Split statistics are committed in `results/benchmark/dataset_stats.json`.

## Research: candidate public datasets (checked November 2026)

| Dataset | Source | Licence | Content | Used? |
|---|---|---|---|---|
| QR codes on different surfaces | Mendeley Data, [doi:10.17632/m6mfwc52vk.1](https://doi.org/10.17632/m6mfwc52vk.1) | CC BY 4.0 (stated on the record) | 100 real photos (42 flat + 8 unannotated flat, 50 cylindrical/free-form) of **one** printed V7-H code at 3000×4000 to 4608×3456; 819 synthetic renders | **Yes**: real photos as `test_real` (synthetic part not used) |
| Szeged visual-code database, Huawei subset | Univ. of Szeged, [page](http://www.inf.u-szeged.hu/~bodnaar/barcode_database/), Bodnár et al. 2018, [doi:10.1007/s10044-017-0619-6](https://doi.org/10.1007/s10044-017-0619-6) | No licence stated; citation requested | 98 smartphone photos (1600×1200) of V4-L codes; 10 000 synthetic images | **Yes**: real photos as `test_external`, local use only, never redistributed |
| Dubská QR datasets I/II, Szentandrási QR tiles | Brno Univ. of Technology, Dubská et al. 2016 [doi:10.1007/s11554-013-0325-6](https://doi.org/10.1007/s11554-013-0325-6) | Listed as CC BY 4.0 by a third-party catalogue | 410 + 400 real photos with original bitmaps | **No**: download links return 404 and are not in the Wayback Machine |
| QR-DN1.0 | Mendeley Data, [doi:10.17632/t2bdr663ms.2](https://doi.org/10.17632/t2bdr663ms.2) | CC BY 4.0 | 6 750 QR images extracted from watermarked photos of screens | No: distortion comes from watermark extraction, not printing or tampering (1 GB) |
| LL-TMDN anti-counterfeiting | IEEE DataPort | Login required | 64 genuine / 576 counterfeit QR images (copy detection) | No: requires an account; targets copy detection rather than tampering |
| BanglaQR-Quish | Mendeley Data, [doi:10.17632/pz95pr8pyd.2](https://doi.org/10.17632/pz95pr8pyd.2) | CC BY 4.0 | 50 000 synthetic payment QR codes, payload-level attacks | No: labels describe payloads, not physical alteration |
| CIC-Trap4Phish (QR subset) | UNB CIC | Research use | ~1 M QR codes labelled by URL maliciousness | No: URL labels; Q-TDE's URL analysis is static and not trained |
| Roboflow Universe QR sets | universe.roboflow.com | Mostly CC BY 4.0 | Detection boxes on photos | No: account/API key needed; no tampering labels |

The key distinction: these sources provide either **real photographs without tampering labels**
or **tampering/maliciousness labels on synthetic or payload-level data**. None provides real
photographs of physically altered codes paired with their originals.

## Composition

| Split | Source | Role |
|---|---|---|
| `train`, `val`, `test_synthetic` | Simulated captures (this repository) | train / threshold calibration / in-distribution test |
| `test_real` | Mendeley real photographs | real-world test (never trained or calibrated on) |
| `test_external` | Szeged Huawei photographs | external-source test (different device, lab and codes) |
| `legacy_val`, `legacy_test` | Original `data/val`, `data/test` (30 + 30) | continuity with the original baseline |
| `test_physical` | `COLLECTION_PROTOCOL.md` captures, when collected | physical tampering test |

| Split | Images | Genuine | Tampered | QR identities | Genuine decodable by OpenCV |
|---|---|---|---|---|---|
| `train` | 2 128 | 1 064 | 1 064 | 700 | 962 |
| `val` | 420 | 210 | 210 | 150 | 186 |
| `test_synthetic` | 450 | 225 | 225 | 150 | 199 |
| `test_real` | 134 | 100 | 34 | 1 physical code (+ 66 undecodable photos) | 34 |
| `test_external` | 174 | 98 | 76 | 76 distinct codes (+ 22 undecodable photos) | 74 |
| `legacy_val` / `legacy_test` | 30 / 30 | 15 / 15 | 15 / 15 | 15 / 15 | – |

Real-photo symbols: Mendeley V7-H, Szeged V4-L. Tampered real samples per category: 4–5
(`test_real`) and 10–11 (`test_external`). Full counts per split, category, version and ECC level
are in `results/benchmark/dataset_stats.json`.

### Simulated captures

1 000 QR identities with inert payloads on reserved `example.*` domains or plain tokens, versions
1–8 in practice (the generator picks the minimum version plus 0–3, capped at 10) and ECC
L/M/Q/H in roughly equal shares. Each identity is rendered on a procedural background with a random
perspective, rotation, quiet zone, paper/ink tone, lighting gradient, optional shadow, blur,
sensor noise and JPEG quality. Each capture exists as a genuine/tampered **pair rendered with the
same capture parameters**; about half of the identities have a second pair with new parameters
and a different tampering category. Tampering is applied to the printed symbol before the camera
model.

### Real photographs

Each photo is downscaled to at most 1 600 px. If OpenCV decodes it, its sampled module grid and
corners define a module-to-image homography and one tampering operation is composited onto the
photo in module space. Local paper and ink levels come from morphological filtering of the photo,
and edge blur and noise are matched to the photo. Photos that OpenCV cannot decode, mostly the
codes on curved surfaces, are kept as **genuine-only** samples, because without module geometry
no faithful tampered counterpart can be made. Genuine and tampered images go through an identical
JPEG step.

### Tampering categories

| Category | Operation |
|---|---|
| `sticker_replacement` | A different valid QR (new payload) printed on a sticker covering the original, with misregistration and shadow |
| `label_overlay` | Paper label covering 5–15 % of the symbol, optionally with printed lines |
| `marker_edit` | 3–20 light data modules filled with ink (mostly clustered) |
| `white_out` | 3–20 dark data modules covered with white |
| `module_patch` | 3–7 × 3–7 module region replaced by a printed patch of random modules |
| `scratch` | 1–3 scratch strokes exposing paper |
| `finder_alteration` | Part of a finder pattern covered or its light ring inked |

## Labelling rules

- `label` comes from construction: a sample is tampered if and only if an operation was applied.
  It is never inferred from decodability. `decodes` and `observed_payload` are recorded separately,
  so decodable tampered images are explicit.
- Real genuine photos are labelled genuine on the basis that the dataset authors published them as
  untampered captures of their own printed codes. `annotation_confidence` records this.

## Leakage prevention

- Simulated splits are assigned **by QR identity before generation** (70/15/15); both images of a
  pair and both captures of an identity share a split.
- Real photographs appear only in test splits. Their codes, devices and labs are disjoint from the
  simulated data and from each other.
- `validate_benchmark.py` checks: no `group_id` or payload in more than one split, every pair in a
  single split, no byte-identical files, and no near-duplicate genuine images across splits (dHash
  Hamming ≤ 4 candidates, confirmed when thumbnail correlation ≥ 0.9; candidates and their
  correlations are listed in `dataset_stats.json`).
- Augmentation is applied on the fly during training only. No image is duplicated to balance classes.
- Thresholds are calibrated on `val` only (95 % specificity target).

## Finding about the legacy synthetic set

The original 100 codes encode sequential URLs (`https://example.com/page/0` … `/page/99`), all as
V3-M with the same mask selection logic, so codes in the legacy validation and test splits are
near-identical images even though no identity is shared (e.g. `/page/77` vs `/page/75`:
r = 0.84 on 64×64 thumbnails, flagged by the dHash check). The legacy splits therefore behave much
like an in-sample evaluation. They are kept only for continuity with the original baseline numbers.

## Known limitations

- **Tampering on real photos is digital, not physical.** It is physically motivated and
  geometry-aware, but it does not reproduce sticker thickness, gloss, specular highlights or
  real marker ink. Results on `test_real` and `test_external` measure transfer to real capture
  conditions, not detection of real-world attacks.
- `test_real` contains **one** physical QR code; results there mostly measure robustness to surface
  shape, perspective and lighting.
- Tampered real samples exist only where OpenCV could decode the original, which biases them
  towards easier captures. The genuine-only photos partly offset this for specificity.
- Simulated backgrounds are procedural, not natural scenes.
- Sample sizes on the real test splits are small; confidence intervals are reported and wide.

## Citation

Please cite the source datasets when using this benchmark:

- *A dataset of QR Codes on top of different surfaces (flat and challenging surfaces)*, Mendeley Data, V1, doi:10.17632/m6mfwc52vk.1 (CC BY 4.0).
- P. Bodnár, T. Grósz, L. Tóth, L. G. Nyúl, *Efficient visual code localization with neural networks*, Pattern Analysis and Applications 21, 249–260 (2018), doi:10.1007/s10044-017-0619-6.
