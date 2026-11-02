"""Fill the two skill-scaffolded EDA notebooks with reproducible cells.

Run after scaffolding with agents/skills/jupyter-notebook/scripts/new_notebook.py.
Executing notebooks is handled by run_notebooks.py, not by this authoring script.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def save(name: str, cells: list) -> None:
    path = HERE / name
    notebook = nbf.read(path, as_version=4)
    notebook.cells = cells
    notebook.metadata.kernelspec = {
        "display_name": "IMP302 (.venv)", "language": "python", "name": "imp302"
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)


COMMON = r'''
from pathlib import Path
import csv, hashlib, json, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "data/manifest.json").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Run the notebook from mini-project/ or a child folder")
sys.path.insert(0, str(ROOT))
from src.core import noisy_image, filters, features, psnr

manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
splits = manifest["splits"]
FIG = ROOT / "notebooks/figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 302
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
def image(split, image_id):
    value = cv2.imread(str(ROOT / "data/clean" / split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(f"Missing image: {split}/{image_id}")
    return value
print("Project:", ROOT.name, "| seed:", SEED, "| source:", manifest["source_dataset"])
'''


save("01_dataset_eda.ipynb", [
    md("""# 01 · Dataset audit and train-image EDA

**Question.** What images are available, are the split IDs valid, and what properties should the denoising model expect?  
**Decision boundary.** Audit integrity across all splits, but examine image content and simulated noise only on **train**. The final test set stays outside feature and parameter choices.  
**Expected outputs.** Saved charts in `notebooks/figures/` and a machine-readable `results/eda_summary.json`. This notebook does not fit or select a model."""),
    md("## Setup, source, and fixed protocol\n\nThe Berkeley archive URL, SHA-256, split IDs, grayscale resize rule, and seed come from `data/manifest.json` and project source. The full license/use note is in the manifest and project README."),
    code(COMMON),
    code(r'''
archive = ROOT / "data/raw/BSDS300-images.tgz"
actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual_sha == manifest["source_sha256"]
all_ids = [i for ids in splits.values() for i in ids]
assert len(all_ids) == len(set(all_ids)), "An image ID crosses splits"
for split, ids in splits.items():
    assert all((ROOT / "data/clean" / split / f"{i}.png").is_file() for i in ids)
print("Archive SHA-256 verified:", actual_sha)
print("Disjoint, complete split counts:", {s: len(v) for s, v in splits.items()})
fig, ax = plt.subplots(figsize=(6.5, 3.5))
labels = list(splits)
bars = ax.bar(labels, [len(splits[k]) for k in labels], color=["#136f63", "#659d8e", "#dba55e", "#394b59"])
ax.bar_label(bars)
ax.set(ylabel="Images", title="Fixed image-level splits")
fig.tight_layout(); fig.savefig(FIG / "eda_split_counts.png", dpi=160); plt.show()
'''),
    md("## Data quality and leakage audit\n\nOnly dimensions, encoding, exact hashes, and small perceptual hashes are inspected across splits. This is a one-time integrity check. Any feature or method choice later in this notebook uses train pixels only."),
    code(r'''
records = []
hashes = {}
perceptual = {}
for split, ids in splits.items():
    for image_id in ids:
        path = ROOT / "data/clean" / split / f"{image_id}.png"
        pixels = image(split, image_id)
        assert pixels.ndim == 2 and pixels.dtype == np.uint8
        height, width = pixels.shape
        assert max(height, width) <= 256
        records.append((split, image_id, width, height))
        hashes[(split, image_id)] = hashlib.sha256(path.read_bytes()).hexdigest()
        tiny = cv2.resize(pixels, (9, 8), interpolation=cv2.INTER_AREA)
        perceptual[(split, image_id)] = (tiny[:, 1:] > tiny[:, :-1]).flatten()
exact_cross_split = [(a, b) for a in hashes for b in hashes
                     if a < b and a[0] != b[0] and hashes[a] == hashes[b]]
near_cross_split = [(a, b, int(np.count_nonzero(perceptual[a] != perceptual[b])))
                    for a in perceptual for b in perceptual
                    if a < b and a[0] != b[0]
                    and np.count_nonzero(perceptual[a] != perceptual[b]) <= 4]
print("Images decoded:", len(records), "| corrupt/missing: 0")
print("Exact cross-split duplicates:", len(exact_cross_split))
print("Cross-split dHash candidates (Hamming <= 4; inspect manually):", len(near_cross_split))
print("First candidate pairs:", near_cross_split[:5])
'''),
    md("## What do train images look like?\n\nA seeded sample grid makes content and preprocessing visible. The images below come from train only; the grid is descriptive, not a test result."),
    code(r'''
rng = np.random.default_rng(SEED)
chosen = rng.choice(splits["train"], size=12, replace=False)
fig, axes = plt.subplots(3, 4, figsize=(10, 7.3))
for ax, image_id in zip(axes.flat, chosen):
    pixels = image("train", image_id)
    ax.imshow(pixels, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"ID {image_id} · {pixels.shape[1]}×{pixels.shape[0]}", fontsize=8)
    ax.axis("off")
fig.suptitle("Seeded sample of 12 training images")
fig.tight_layout(); fig.savefig(FIG / "eda_train_gallery.png", dpi=160); plt.show()
'''),
    md("## Resolution and pixel statistics\n\nThe model sees gray pixel values from 0 to 255. Brightness and contrast can vary across natural scenes. The histograms and scatter below use all 60 train images, never pilot or final-test pixels."),
    code(r'''
train_arrays = [image("train", i) for i in splits["train"]]
train_width = np.array([a.shape[1] for a in train_arrays])
train_height = np.array([a.shape[0] for a in train_arrays])
train_mean = np.array([a.mean() for a in train_arrays])
train_std = np.array([a.std() for a in train_arrays])
hist = np.sum([np.histogram(a, bins=np.arange(0, 257, 16))[0]
               for a in train_arrays], axis=0)
hist = hist / hist.sum()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].scatter(train_width, train_height, s=24, alpha=.8, c="#136f63")
axes[0].set(xlabel="Width (px)", ylabel="Height (px)", title="Train image dimensions")
axes[1].scatter(train_mean, train_std, s=24, alpha=.8, c="#c16d3d")
axes[1].set(xlabel="Mean gray level", ylabel="Pixel standard deviation", title="Brightness and contrast")
axes[2].bar(np.arange(16)*16+8, hist, width=14, color="#395c78")
axes[2].set(xlabel="Gray level (0–255)", ylabel="Fraction of train pixels", title="Pixel intensity distribution")
fig.tight_layout(); fig.savefig(FIG / "eda_train_properties.png", dpi=160); plt.show()
print("Train dimensions (width × height), unique:", sorted(set(zip(train_width, train_height))))
print("Train mean brightness range:", round(float(train_mean.min()), 2), "to", round(float(train_mean.max()), 2))
print("Train pixel std range:", round(float(train_std.min()), 2), "to", round(float(train_std.max()), 2))
'''),
    md("## Simulated Gaussian noise on a train image\n\nThe same clean train image is corrupted independently at three documented σ levels. Clipping to [0,255] means the measured pixel difference is not exactly a perfect Gaussian near black or white."),
    code(r'''
example_id = splits["train"][0]
clean = image("train", example_id)
fig, axes = plt.subplots(1, 4, figsize=(11, 3.3))
axes[0].imshow(clean, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Reference (train)")
noise_checks = {}
for ax, sigma in zip(axes[1:], (15, 25, 50)):
    noisy = noisy_image(clean, sigma, example_id, SEED)
    difference = noisy.astype(np.float32) - clean.astype(np.float32)
    noise_checks[str(sigma)] = {"measured_difference_std": float(difference.std()),
                                "clipped_fraction": float(((noisy == 0) | (noisy == 255)).mean())}
    ax.imshow(noisy, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"σ={sigma}; diff SD={difference.std():.1f}")
for ax in axes: ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "eda_noise_levels.png", dpi=160); plt.show()
print(json.dumps(noise_checks, indent=2))
'''),
    md("## Recorded findings and limits\n\nThe JSON below records audit counts and train-only summary values for the report. dHash is a screening tool, not proof that two images depict the same scene. For method selection, use train/validation as specified in `src/experiment.py`; the final test set must remain locked."),
    code(r'''
eda = {
  "source": manifest["source"], "source_sha256": actual_sha,
  "seed": SEED, "split_counts": {s: len(v) for s, v in splits.items()},
  "image_count_audited": len(records),
  "exact_cross_split_duplicates": len(exact_cross_split),
  "dhash_candidates_cross_split_hamming_le_4": len(near_cross_split),
  "train_width_min_max": [int(train_width.min()), int(train_width.max())],
  "train_height_min_max": [int(train_height.min()), int(train_height.max())],
  "train_brightness_min_max": [float(train_mean.min()), float(train_mean.max())],
  "train_contrast_std_min_max": [float(train_std.min()), float(train_std.max())],
  "noise_example_train_id": example_id, "noise_checks": noise_checks,
  "software": {"numpy": np.__version__, "opencv": cv2.__version__},
}
target = ROOT / "results/eda_summary.json"
target.write_text(json.dumps(eda, indent=2), encoding="utf-8")
print("Saved", target)
'''),
])


save("02_features_and_model.ipynb", [
    md("""# 02 · Denoising features and model analysis

**Question.** What information reaches the fusion and residual MLP, and how did validation choose a configuration?  
**Data boundary.** Feature pictures and residual distributions use **train** only. Validation CSVs document choices already made. Final test CSVs are read only as locked outcomes; they never drive a change here.  
**Outputs.** Feature visualizations and model-selection plots in `notebooks/figures/`."""),
    md("## Setup and saved configuration\n\nLoad the same preprocessing and feature functions as the experiment, plus the frozen NLM and fusion parameters. Nothing is retrained in this notebook."),
    code(COMMON + r'''
from src.learn import ResidualMLP
run = json.loads((ROOT / "results/run.json").read_text(encoding="utf-8"))
h_map = {int(k): float(v) for k, v in run["selected_nlm_h"].items()}
fusion = json.loads((ROOT / "models/fusion.json").read_text(encoding="utf-8"))
model, stored_h, texture = ResidualMLP.load(ROOT / "models/denoiser.npz")
assert h_map == stored_h
print("NLM h by σ:", h_map, "| fusion weights:", fusion)
print("MLP input dimensions:", len(model.mean), "| hidden units:", len(model.weights["b1"]), "| texture:", texture)
'''),
    md("## One train image through the filter bank\n\nGaussian averages nearby pixels, bilateral also checks intensity similarity, and NLM looks for similar patches. All receive exactly the same synthetic noisy input."),
    code(r'''
sigma = 25
image_id = splits["train"][0]
clean = image("train", image_id)
noisy = noisy_image(clean, sigma, image_id, SEED)
outputs = filters(noisy, h_map[sigma], sigma)
fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
for ax, (label, pixels) in zip(axes, [("Reference", clean), *outputs.items()]):
    ax.imshow(pixels, cmap="gray", vmin=0, vmax=255)
    ax.set_title(label if label == "Reference" else f"{label}\n{psnr(clean,pixels):.2f} dB")
    ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "eda_filter_bank.png", dpi=160); plt.show()
'''),
    md("## The ten feature channels\n\nFeatures 1–4 are the noisy image and three filter outputs, each divided by 255. Channels 5–6 are disagreements with NLM divided by 50. Channel 7 gives σ/50. The last three represent local mean, local standard deviation, and Sobel edge magnitude. These are per-pixel inputs, not extra photographs."),
    code(r'''
feature_names = ["noisy / 255", "Gaussian / 255", "bilateral / 255", "NLM / 255",
                 "(noisy−NLM) / 50", "(bilateral−NLM) / 50", "σ / 50",
                 "local mean / 255", "local std / 50", "Sobel edge / 400"]
x = features(outputs, sigma, include_texture=True)
assert x.shape == (clean.size, 10)
fig, axes = plt.subplots(2, 5, figsize=(12, 5.7))
for k, ax in enumerate(axes.flat):
    layer = x[:, k].reshape(clean.shape)
    im = ax.imshow(layer, cmap="viridis")
    ax.set_title(feature_names[k], fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
fig.tight_layout(); fig.savefig(FIG / "eda_feature_maps.png", dpi=160); plt.show()
'''),
    md("## Residual target and actual MLP correction\n\nThe MLP learns `(reference − NLM)/255` from train pixels. The network is `10 inputs → 32 ReLU units → 1 correction`, with normalization learned on train, Adam, and 12 epochs. This is a small model trained from scratch; there is no pretrained checkpoint to fine-tune."),
    code(r'''
target = (clean.astype(np.float32) - outputs["nlm"].astype(np.float32)).ravel() / 255
pred = model.predict(x)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(target, bins=60, alpha=.55, label="Needed correction", color="#136f63")
axes[0].hist(pred, bins=60, alpha=.55, label="MLP prediction", color="#cc7b43")
axes[0].set(xlabel="Correction / 255", ylabel="Train pixels", title="Residual distributions")
axes[0].legend(fontsize=8)
idx = np.random.default_rng(SEED).choice(len(target), min(2500, len(target)), replace=False)
axes[1].scatter(target[idx], pred[idx], s=3, alpha=.3, c="#385f82")
axes[1].plot([-.4,.4],[-.4,.4], color="black", lw=1)
axes[1].set(xlim=(-.4,.4), ylim=(-.4,.4), xlabel="Needed correction", ylabel="MLP prediction", title="Train-pixel signal")
fig.tight_layout(); fig.savefig(FIG / "eda_residual_signal.png", dpi=160); plt.show()
print("Residual target SD:", round(float(target.std()), 4), "| prediction SD:", round(float(pred.std()), 4))
'''),
    md("## Recorded model selection\n\nThe CSV below is the original validation search, and the weights were fitted on train. Do not pick a new setting from the final-test chart later in this notebook."),
    code(r'''
with (ROOT / "results/model_selection.csv").open(newline="", encoding="utf-8") as f:
    selection = list(csv.DictReader(f))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
names = [f'{r["hidden"]} units; texture={r["texture"]}' for r in selection]
values = [float(r["val_mean_psnr_db"]) for r in selection]
bars = axes[0].barh(names, values, color=["#8ba9a4", "#8ba9a4", "#136f63"])
axes[0].bar_label(bars, fmt="%.2f", padding=3)
axes[0].set(xlim=(0, max(values)+4), xlabel="Validation mean PSNR (dB)", title="MLP configuration selection")
levels = [15,25,50]
colors = ["#6e88a4", "#dba55e", "#136f63"]
bottom = np.zeros(3)
for col, label, color in zip(range(3), ["Gaussian", "Bilateral", "NLM"], colors):
    vals = np.array([fusion[str(s)][col] for s in levels])
    axes[1].bar([str(s) for s in levels], vals, bottom=bottom, label=label, color=color)
    bottom += vals
axes[1].set(xlabel="Noise σ", ylabel="Fusion weight", ylim=(0,1), title="Train-fitted weights")
axes[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "eda_model_selection.png", dpi=160); plt.show()
print("Selected validation configuration:", max(zip(values, names)))
'''),
    md("## Locked final-test results for interpretation only\n\nThis reads the existing saved `summary.csv`; it neither changes the model nor recomputes test predictions. The final test is 20 image IDs per σ. A larger PSNR means lower average squared pixel error; SSIM is a complementary structural score."),
    code(r'''
with (ROOT / "results/summary.csv").open(newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))
methods = ["gaussian", "bilateral", "nlm", "fusion", "mlp"]
levels = [15,25,50]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
palette = ["#8ba9a4", "#dba55e", "#476883", "#136f63", "#b95d48"]
for method, color in zip(methods, palette):
    rows = [next(r for r in summary if r["method"] == method and int(r["sigma"]) == s) for s in levels]
    axes[0].plot(levels, [float(r["mean_psnr_db"]) for r in rows], "o-", label=method, color=color)
    axes[1].plot(levels, [float(r["mean_ssim"]) for r in rows], "o-", label=method, color=color)
axes[0].set(xlabel="Noise σ", ylabel="Mean PSNR (dB)", title="Frozen final-test PSNR", xticks=levels)
axes[1].set(xlabel="Noise σ", ylabel="Mean SSIM", title="Frozen final-test SSIM", xticks=levels)
axes[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "eda_locked_test_results.png", dpi=160); plt.show()
'''),
    md("## Findings and next step\n\nThe feature maps show what the model can see. They do not prove each channel helped. The validation ablation is the evidence for the texture-channel choice; the locked test scores show whether that choice generalized. The MLP remains a negative result against the fusion in this protocol. See `report/main.tex` for the scientific discussion and `docs/teach-me-like-grade-5.md` for a simpler explanation."),
])
