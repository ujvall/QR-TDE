"""
Download the external real-world QR datasets used by Q-TDE into data/external/.

Sources (verified November 2026, see dataprep/DATASET_CARD.md):
  mendeley_surfaces  "A dataset of QR Codes on top of different surfaces", Mendeley Data,
                     doi:10.17632/m6mfwc52vk.1, CC BY 4.0. Full zip (~384 MB).
  szeged_huawei      University of Szeged visual-code database (Bodnar et al., 2018,
                     doi:10.1007/s10044-017-0619-6). Only the 100 real smartphone photos in
                     huawei/ are pulled from the 2 GB archive with HTTP range reads; the
                     synthetic part is skipped. No explicit licence is stated on the page
                     (citation requested), so the images are used locally and never
                     redistributed.

Nothing here is committed: data/external/ is git-ignored.

    python -m dataprep.acquire            # both sources
    python -m dataprep.acquire --only szeged_huawei
"""

import argparse
import hashlib
import io
import os
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "external"))
UA = {"User-Agent": "qtde-dataset-fetch"}

MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/m6mfwc52vk/files/"
                "67b47b85-0e10-4095-8c61-331de5039791/file_downloaded")
SZEGED_URL = "http://www.inf.u-szeged.hu/~bodnaar/barcode_database/syn10k_plus_huawei.zip"


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, so zipfile can read single members."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA))
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _safe_extract(zf, member, dest_root):
    # Archives are untrusted: refuse absolute paths and parent traversal.
    target = os.path.abspath(os.path.join(dest_root, member))
    if not target.startswith(os.path.abspath(dest_root) + os.sep):
        raise ValueError(f"unsafe path in archive: {member}")
    if member.endswith("/"):
        os.makedirs(target, exist_ok=True)
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with zf.open(member) as src, open(target, "wb") as dst:
        dst.write(src.read())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_mendeley():
    dest = os.path.join(ROOT, "mendeley_surfaces")
    archive = os.path.join(ROOT, "_downloads", "qr_codes_in_surfaces.zip")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    if not os.path.exists(archive):
        print("downloading", MENDELEY_URL)
        with urllib.request.urlopen(urllib.request.Request(MENDELEY_URL, headers=UA)) as r, \
                open(archive, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    print("sha256", sha256(archive))
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        for m in members:
            _safe_extract(zf, m, dest)
    print(f"mendeley_surfaces: {len(members)} entries -> {dest}")


def fetch_szeged_huawei():
    dest = os.path.join(ROOT, "szeged_huawei")
    zf = zipfile.ZipFile(io.BufferedReader(_HttpRangeFile(SZEGED_URL), buffer_size=1 << 16))
    members = [m for m in zf.namelist() if m.startswith("huawei/")]
    for m in members:
        _safe_extract(zf, m, dest)
    print(f"szeged_huawei: {len(members)} entries -> {dest}")


SOURCES = {"mendeley_surfaces": fetch_mendeley, "szeged_huawei": fetch_szeged_huawei}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SOURCES))
    args = ap.parse_args()
    for name, fn in SOURCES.items():
        if args.only in (None, name):
            fn()
