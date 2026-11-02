"""Check integrity of the delivered data, model, and measured artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import ARCHIVE, MANIFEST, ROOT, SHA256, sha256
from .learn import ResidualMLP


def csv_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    assert {k: len(v) for k, v in splits.items()} == {"train": 60, "val": 12, "pilot": 20, "test": 20}
    for left, left_ids in splits.items():
        for right, right_ids in splits.items():
            if left < right:
                assert not set(left_ids) & set(right_ids), f"Overlapping splits: {left}, {right}"
        for image_id in left_ids:
            assert (ROOT / "data/clean" / left / f"{image_id}.png").is_file(), f"Missing {left}/{image_id}"
    assert sha256(ARCHIVE) == SHA256
    model, h_map, texture = ResidualMLP.load(ROOT / "models/denoiser.npz")
    assert set(h_map) == {15, 25, 50}
    assert len(model.mean) == (10 if texture else 7)
    weights = json.loads((ROOT / "models/fusion.json").read_text(encoding="utf-8"))
    for sigma in (15, 25, 50):
        row = weights[str(sigma)]
        assert len(row) == 3 and all(0 <= w <= 1 for w in row) and abs(sum(row) - 1) < 1e-10
    rows = csv_rows("per_image.csv")
    methods = {"noisy", "gaussian", "bilateral", "nlm", "fusion", "mlp"}
    expected = {(str(sigma), image_id, method) for sigma in (15, 25, 50)
                for image_id in splits["test"] for method in methods}
    actual = {(r["sigma"], r["image_id"], r["method"]) for r in rows}
    assert len(rows) == len(actual) == len(expected) == 360 and actual == expected
    for item in csv_rows("summary.csv"):
        subset = [r for r in rows if r["sigma"] == item["sigma"] and r["method"] == item["method"]]
        assert len(subset) == int(item["n_images"]) == 20
        assert np.isclose(np.mean([float(r["psnr_db"]) for r in subset]), float(item["mean_psnr_db"]))
        assert np.isclose(np.mean([float(r["ssim"]) for r in subset]), float(item["mean_ssim"]))
    assert (ROOT / "report/results.tex").is_file()
    assert (ROOT / "report/build/main.pdf").is_file()
    print("OK: archive SHA-256; disjoint 60/12/20/20 splits; 360 unique test scores; summary means; model and fusion weights; LaTeX/PDF artifacts")


if __name__ == "__main__":
    main()
