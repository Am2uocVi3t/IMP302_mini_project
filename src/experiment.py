"""Run a reproducible noise-conditional denoising study on fixed BSDS300 subsets."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import features, filters, noisy_image, psnr, ssim
from .data import ROOT, prepare
from .learn import ResidualMLP, restore

RESULTS = ROOT / "results"
MODEL = ROOT / "models/denoiser.npz"
LEVELS = (15, 25, 50)
H_GRID = {15: (12, 16, 20, 24), 25: (20, 24, 28, 32, 36), 50: (32, 36, 40, 48, 56)}


def image(split: str, image_id: str) -> np.ndarray:
    result = cv2.imread(str(ROOT / "data/clean" / split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise FileNotFoundError(f"Missing image {split}/{image_id}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tune_nlm(ids: list[str]) -> tuple[dict[int, float], list[dict]]:
    records = []
    selected = {}
    for sigma in LEVELS:
        for h in H_GRID[sigma]:
            values = []
            for image_id in ids:
                clean = image("val", image_id)
                noisy = noisy_image(clean, sigma, image_id)
                denoised = cv2.fastNlMeansDenoising(noisy, None, float(h), 7, 21)
                values.append(psnr(clean, denoised))
            records.append({"sigma": sigma, "h": h, "val_psnr_db": float(np.mean(values))})
        best = max((r for r in records if r["sigma"] == sigma), key=lambda r: r["val_psnr_db"])
        selected[sigma] = float(best["h"])
    return selected, records


def samples(ids: list[str], h_map: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    rng = np.random.default_rng(302)
    for sigma in LEVELS:
        for image_id in ids:
            clean = image("train", image_id)
            outputs = filters(noisy_image(clean, sigma, image_id), h_map[sigma], sigma)
            x = features(outputs, sigma, True)
            y = (clean.reshape(-1).astype(np.float32) - outputs["nlm"].reshape(-1)) / 255
            indices = rng.choice(len(x), size=min(700, len(x)), replace=False)
            xs.append(x[indices])
            ys.append(y[indices])
    return np.concatenate(xs), np.concatenate(ys)


def fit_fusion(x: np.ndarray, residual: np.ndarray) -> dict[int, list[float]]:
    """Train non-negative, sum-to-one Gaussian/Bilateral/NLM weights per noise level."""
    weights = {}
    for sigma in LEVELS:
        mask = np.isclose(x[:, 6], sigma / 50)
        predictions = x[mask, 1:4].astype(np.float64)  # Gaussian, bilateral, NLM; scaled 0..1
        clean = predictions[:, 2] + residual[mask].astype(np.float64)
        gram = predictions.T @ predictions / len(predictions)
        cross = predictions.T @ clean / len(predictions)
        candidates = []
        for gauss in range(21):
            for bilateral in range(21 - gauss):
                nlm = 20 - gauss - bilateral
                w = np.array([gauss, bilateral, nlm], dtype=float) / 20
                loss = w @ gram @ w - 2 * w @ cross
                candidates.append((loss, w))
        weights[sigma] = min(candidates, key=lambda item: item[0])[1].tolist()
    return weights


def fuse(outputs: dict[str, np.ndarray], weights: list[float]) -> np.ndarray:
    result = sum(w * outputs[name].astype(np.float32) for w, name in zip(weights, ("gaussian", "bilateral", "nlm")))
    return np.clip(result, 0, 255).astype(np.uint8)


def assess(model: ResidualMLP, texture: bool, split: str, ids: list[str], h_map: dict[int, float]) -> float:
    values = []
    for sigma in LEVELS:
        for image_id in ids:
            clean = image(split, image_id)
            outputs = filters(noisy_image(clean, sigma, image_id), h_map[sigma], sigma)
            values.append(psnr(clean, restore(outputs, model, sigma, texture)))
    return float(np.mean(values))


def evaluate(model: ResidualMLP, texture: bool, ids: list[str], h_map: dict[int, float], fusion: dict[int, list[float]]) -> list[dict]:
    rows = []
    for sigma in LEVELS:
        for image_id in ids:
            clean = image("test", image_id)
            outputs = filters(noisy_image(clean, sigma, image_id), h_map[sigma], sigma)
            outputs["fusion"] = fuse(outputs, fusion[sigma])
            outputs["mlp"] = restore(outputs, model, sigma, texture)
            for method, estimate in outputs.items():
                rows.append({"sigma": sigma, "image_id": image_id, "method": method,
                             "psnr_db": psnr(clean, estimate), "ssim": ssim(clean, estimate)})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for sigma in LEVELS:
        for method in ("noisy", "gaussian", "bilateral", "nlm", "fusion", "mlp"):
            values = [r for r in rows if r["sigma"] == sigma and r["method"] == method]
            summary.append({"sigma": sigma, "method": method, "n_images": len(values),
                            "mean_psnr_db": float(np.mean([r["psnr_db"] for r in values])),
                            "mean_ssim": float(np.mean([r["ssim"] for r in values]))})
    return summary


def paired(rows: list[dict], a: str = "fusion", b: str = "nlm") -> list[dict]:
    rng = np.random.default_rng(302)
    results = []
    for sigma in LEVELS:
        left = {(r["image_id"]): r for r in rows if r["sigma"] == sigma and r["method"] == a}
        right = {(r["image_id"]): r for r in rows if r["sigma"] == sigma and r["method"] == b}
        diffs = np.array([left[k]["psnr_db"] - right[k]["psnr_db"] for k in sorted(left)], dtype=float)
        boots = rng.choice(diffs, size=(5000, len(diffs)), replace=True).mean(axis=1)
        results.append({"sigma": sigma, "comparison": f"{a} minus {b}", "n_images": len(diffs),
                        "mean_delta_db": float(diffs.mean()), "ci95_low_db": float(np.quantile(boots, .025)),
                        "ci95_high_db": float(np.quantile(boots, .975)), "win_count": int((diffs > 0).sum())})
    return results


def figures(rows: list[dict], summary: list[dict], h_map: dict[int, float], fusion: dict[int, list[float]], model: ResidualMLP, texture: bool, test_id: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for method, marker in (("noisy", "x"), ("gaussian", "o"), ("bilateral", "s"), ("nlm", "^"), ("fusion", "D"), ("mlp", "v")):
        data = [r for r in summary if r["method"] == method]
        ax.plot([r["sigma"] for r in data], [r["mean_psnr_db"] for r in data], marker=marker, label=method)
    ax.set(xlabel="Noise standard deviation σ (pixel values)", ylabel="Mean PSNR (dB)", xticks=LEVELS)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / "psnr_by_noise.png", dpi=180)
    plt.close(fig)

    clean = image("test", test_id)
    noisy = noisy_image(clean, 25, test_id)
    outputs = filters(noisy, h_map[25], 25)
    outputs["fusion"] = fuse(outputs, fusion[25])
    items = [("Reference", clean), ("Noisy", noisy), ("Gaussian", outputs["gaussian"]),
             ("Bilateral", outputs["bilateral"]), ("NLM", outputs["nlm"]), ("Fusion", outputs["fusion"])]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, (title, arr) in zip(axes.ravel(), items):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title + (f" | {psnr(clean, arr):.2f} dB" if title != "Reference" else ""))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    manifest = prepare()
    splits = manifest["splits"]
    h_map, tuning = tune_nlm(splits["val"])
    write_csv(RESULTS / "nlm_tuning.csv", tuning)
    x, y = samples(splits["train"], h_map)
    fusion = fit_fusion(x, y)
    (ROOT / "models/fusion.json").write_text(json.dumps({str(k): v for k, v in fusion.items()}, indent=2), encoding="utf-8")
    candidates = []
    best = None
    for hidden, texture in ((16, False), (16, True), (32, True)):
        selected_x = x if texture else x[:, :7]
        model = ResidualMLP(selected_x.shape[1], hidden)
        model.fit(selected_x, y)
        score = assess(model, texture, "val", splits["val"], h_map)
        candidates.append({"hidden": hidden, "texture": texture, "val_mean_psnr_db": score})
        if best is None or score > best[0]:
            best = (score, model, texture)
    write_csv(RESULTS / "model_selection.csv", candidates)
    _, model, texture = best
    model.save(MODEL, h_map, texture)
    rows = evaluate(model, texture, splits["test"], h_map, fusion)
    write_csv(RESULTS / "per_image.csv", rows)
    summary = summarize(rows)
    write_csv(RESULTS / "summary.csv", summary)
    intervals = paired(rows) + paired(rows, "fusion", "bilateral")
    write_csv(RESULTS / "paired_bootstrap.csv", intervals)
    figures(rows, summary, h_map, fusion, model, texture, splits["test"][0])
    metadata = {"dataset": manifest, "sigma_levels": LEVELS, "seed": 302,
                "selected_nlm_h": h_map, "fusion_weights_gaussian_bilateral_nlm": fusion, "model_selection": candidates,
                "selected_model": {"hidden": len(model.weights["b1"]), "texture": texture},
                "training_samples": len(x), "versions": {"python": platform.python_version(),
                "numpy": np.__version__, "opencv": cv2.__version__}}
    (RESULTS / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"selected_h": h_map, "model": metadata["selected_model"], "summary": summary, "paired": intervals}, indent=2))


if __name__ == "__main__":
    main()
