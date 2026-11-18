"""
Generate printable originals for the physical tampering protocol (dataprep/COLLECTION_PROTOCOL.md).

    python -m dataprep.physical_kit --out data/physical/kit --count 24
"""

import argparse
import csv
import itertools
import os

import qrcode

LEVELS = {"L": qrcode.constants.ERROR_CORRECT_L, "M": qrcode.constants.ERROR_CORRECT_M,
          "Q": qrcode.constants.ERROR_CORRECT_Q, "H": qrcode.constants.ERROR_CORRECT_H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/physical/kit")
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    combos = itertools.cycle(itertools.product([3, 4, 5, 6, 7], "LMQH"))
    with open(os.path.join(args.out, "originals.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_id", "payload", "version", "ecc_level", "file"])
        for i in range(args.count):
            version, level = next(combos)
            oid = f"phys_{i:03d}"
            payload = f"HTTPS://EXAMPLE.ORG/Q/{oid.upper()}"  # inert; uppercase fits alphanumeric mode
            q = qrcode.QRCode(version=version, error_correction=LEVELS[level], box_size=20, border=4)
            q.add_data(payload)
            q.make(fit=False)
            name = f"{oid}_v{version}{level}.png"
            q.make_image().save(os.path.join(args.out, name))
            w.writerow([oid, payload, version, level, name])
    print(f"wrote {args.count} originals to {args.out}")


if __name__ == "__main__":
    main()
