"""Fetch the original Berkeley images and create fixed image-level splits."""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
SHA256 = "7f855a00491a3f1ca2609b3c96651b88b265fe59b70a7b007cc7a2850a22b386"
ARCHIVE = ROOT / "data/raw/BSDS300-images.tgz"
MANIFEST = ROOT / "data/manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare() -> dict:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(URL, ARCHIVE)
    actual = sha256(ARCHIVE)
    if actual != SHA256:
        raise RuntimeError(f"Archive checksum mismatch: {actual}")

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile() and m.name.endswith(".jpg")}
        train_ids = sorted(Path(n).stem for n in members if "/train/" in n)
        test_ids = sorted(Path(n).stem for n in members if "/test/" in n)
        splits = {"train": train_ids[:60], "val": train_ids[60:72], "pilot": test_ids[:20], "test": test_ids[20:40]}
        for split, ids in splits.items():
            out_dir = ROOT / "data/clean" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            original_split = "test" if split in ("pilot", "test") else "train"
            for image_id in ids:
                target = out_dir / f"{image_id}.png"
                if target.exists():
                    continue
                member = members[f"BSDS300/images/{original_split}/{image_id}.jpg"]
                pixels = cv2.imdecode(np.frombuffer(archive.extractfile(member).read(), dtype="uint8"), cv2.IMREAD_GRAYSCALE)
                if pixels is None:
                    raise RuntimeError(f"Could not decode {member.name}")
                height, width = pixels.shape
                scale = min(1.0, 256 / max(height, width))
                if scale < 1:
                    pixels = cv2.resize(pixels, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(target), pixels):
                    raise RuntimeError(f"Could not write {target}")

    manifest = {
        "source": URL,
        "source_sha256": SHA256,
        "source_dataset": "Berkeley Segmentation Dataset BSDS300, images archive",
        "intended_use": "non-commercial education and research",
        "reference": "decoded original JPEG, converted to grayscale and resized to max side 256 px",
        "splits": splits,
        "selection": "first sorted train IDs; pilot=first 20 sorted original test IDs (used during development); final test=next 20 original test IDs, not used for design",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare()["splits"], indent=2))
