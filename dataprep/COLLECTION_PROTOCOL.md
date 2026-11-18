# Physical tampering capture protocol

No public dataset found during this work provides **photographs of physically tampered QR codes with
a retained original** (see `DATASET_CARD.md`). The benchmark in this repository therefore applies
tampering digitally to real photographs. This protocol describes how to collect the missing physical
data so the semi-synthetic results can be checked against real tampering.

## 1. Print the originals

```powershell
python -m dataprep.physical_kit --out data/physical/kit --count 24
```

This writes 24 printable PNGs and `originals.csv` (`original_id, payload, version, ecc_level`).
Payloads are inert `HTTPS://EXAMPLE.ORG/Q/<ID>` URLs, so no real destination is ever encoded.
The kit mixes versions 3–7 and ECC levels L/M/Q/H. Print each at two sizes (about 3 cm and 6 cm)
on plain paper and, if available, on glossy or sticker paper.

## 2. Photograph genuine captures first

For every printed original, take at least 4 photos before any modification, varying one factor
per photo: distance, angle (up to about 45°), lighting (daylight / indoor / phone flash) and device.
Do not crop. Keep the full-resolution files.

## 3. Apply one tampering operation per printed copy

| Category | How |
|---|---|
| `sticker_replacement` | Print a different kit QR on sticker paper and stick it over the original |
| `label_overlay` | Stick a small paper label (5–15% of the code area) over the data region |
| `marker_edit` | Fill 3–20 light modules with a black marker |
| `white_out` | Cover 3–20 dark modules with correction tape or white paint |
| `scratch` | Scratch or tear through part of the symbol |
| `finder_alteration` | Cover or draw over part of one finder pattern |

Photograph the tampered copy under the same conditions as its genuine captures (same session,
device and lighting), at least 4 photos.

## 4. Record metadata

Fill `data/physical/captures.csv` (one row per photo):

```
image_id,original_id,label,tampering_category,device,session_id,lighting,angle_deg,distance_cm,paper,notes
```

- `label` is `genuine` or `tampered`; it is set by what was physically done, never by whether the
  code decodes.
- `session_id` groups photos taken together; splitting is done by `original_id` and `session_id`.
- Do not photograph people, screens showing personal data or private locations.

## 5. Evaluate

Place photos in `data/physical/images/` and rebuild, rescore and report:

```powershell
python -m dataprep.build_benchmark --physical data/physical/captures.csv
python -m experiments.classical
python -m experiments.learned
python -m experiments.benchmark
```

The photos appear as the `test_physical` split.

Physical captures are only ever used as a test set. Thresholds stay the ones calibrated on the
benchmark validation split.
