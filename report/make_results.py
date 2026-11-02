"""Generate LaTeX tables from the saved CSV results, never typed values."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/results.tex"
NL = " " + chr(92) * 2


def read(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def table(lines: list[str], caption: str, label: str, cols: str, header: str, rows: list[str]) -> None:
    lines.extend([r"\begin{table}[ht]", r"\centering\small", r"\caption{" + caption + "}",
                  r"\label{" + label + "}", r"\begin{tabular}{" + cols + "}", r"\toprule",
                  header + NL, r"\midrule"])
    lines.extend(row + NL for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main() -> None:
    summary, pairs, selection = read("summary.csv"), read("paired_bootstrap.csv"), read("model_selection.csv")
    fusion = json.loads((ROOT / "models/fusion.json").read_text(encoding="utf-8"))
    names = {"noisy": "Noisy input", "gaussian": "Gaussian", "bilateral": "Bilateral", "nlm": "NLM",
             "fusion": "Learned fusion", "mlp": "Residual MLP"}
    lines = ["% Generated from results/*.csv; measured values are not edited by hand."]
    rows = []
    for method, name in names.items():
        values = [next(r for r in summary if r["method"] == method and int(r["sigma"]) == sigma)
                  for sigma in (15, 25, 50)]
        cells = [f'{float(r["mean_psnr_db"]):.2f} / {float(r["mean_ssim"]):.3f}' for r in values]
        rows.append(name + " & " + " & ".join(cells))
    table(lines, "Mean PSNR (dB) / SSIM on 20 held-out test images per noise level.", "tab:main", "lccc",
          r"Method & $\sigma=15$ & $\sigma=25$ & $\sigma=50$", rows)

    rows = []
    for row in pairs:
        delta = float(row["mean_delta_db"])
        percent = 100 * (1 - 10 ** (-delta / 10))
        baseline = "NLM" if row["comparison"] == "fusion minus nlm" else "Bilateral"
        rows.append(f'{row["sigma"]} & {baseline} & {delta:+.2f} & '
                    f'[{float(row["ci95_low_db"]):+.2f}, {float(row["ci95_high_db"]):+.2f}] & '
                    f'{row["win_count"]}/20 & {percent:.1f}\\%')
    table(lines, r"Paired fusion gains; 95\% image bootstrap intervals. MSE reduction is the geometric-mean equivalent of mean $\Delta$PSNR.",
          "tab:paired", "c l c c c c",
          r"$\sigma$ & Baseline & $\Delta$ dB & 95\% CI & Wins & MSE reduction", rows)

    rows = []
    for sigma in (15, 25, 50):
        weights = fusion[str(sigma)]
        rows.append(f'{sigma} & {weights[0]:.2f} & {weights[1]:.2f} & {weights[2]:.2f}')
    table(lines, "Fusion weights fitted on training images; each row sums to one.", "tab:weights", "cccc",
          r"$\sigma$ & Gaussian & Bilateral & NLM", rows)

    rows = [f'{r["hidden"]} & {"Yes" if r["texture"] == "True" else "No"} & '
            f'{float(r["val_mean_psnr_db"]):.2f}' for r in selection]
    table(lines, "MLP ablation and configuration selection on validation images.", "tab:ablation", "ccc",
          "Hidden units & Texture features & Validation PSNR (dB)", rows)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
