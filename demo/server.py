"""Local, dependency-light demonstration server for the trained denoiser."""
from __future__ import annotations

import base64
import binascii
import csv
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import filters, noisy_image, psnr, ssim  # noqa: E402
from src.learn import ResidualMLP, restore  # noqa: E402
from src.experiment import fuse  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
MODEL = ROOT / "models/denoiser.npz"


def png_data(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def reply(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/bootstrap":
            ids = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))["splits"]["test"]
            with (ROOT / "results/summary.csv").open(newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            with (ROOT / "results/paired_bootstrap.csv").open(newline="", encoding="utf-8") as stream:
                paired = list(csv.DictReader(stream))
            self.reply(200, {"examples": ids, "summary": summary, "paired": paired})
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/process":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 6_000_000:
                raise ValueError("Input too large (maximum 6 MB)")
            payload = json.loads(self.rfile.read(length))
            sigma = int(payload.get("sigma", 25))
            if sigma not in (15, 25, 50):
                raise ValueError("Choose σ = 15, 25, or 50")
            image_id = str(payload.get("example_id", ""))
            if payload.get("image"):
                encoded = str(payload["image"]).split(",")[-1]
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > 4_000_000:
                    raise ValueError("Image too large (maximum 4 MB)")
                clean = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                if clean is None:
                    raise ValueError("Unsupported image")
                image_id = "uploaded-image"
            else:
                safe_ids = set(json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))["splits"]["test"])
                if image_id not in safe_ids:
                    raise ValueError("Choose a listed example")
                clean = cv2.imread(str(ROOT / "data/clean/test" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
            if min(clean.shape) < 16:
                raise ValueError("Image must be at least 16 × 16 pixels")
            height, width = clean.shape
            scale = min(1.0, 256 / max(height, width))
            if scale < 1:
                clean = cv2.resize(clean, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            model, h_map, texture = ResidualMLP.load(MODEL)
            fusion = json.loads((ROOT / "models/fusion.json").read_text(encoding="utf-8"))
            noisy = noisy_image(clean, sigma, image_id)
            outputs = filters(noisy, h_map[sigma], sigma)
            outputs["fusion"] = fuse(outputs, fusion[str(sigma)])
            outputs["mlp"] = restore(outputs, model, sigma, texture)
            all_images = {"reference": clean, **outputs}
            scores = {key: {"psnr_db": round(psnr(clean, value), 3), "ssim": round(ssim(clean, value), 4)}
                      for key, value in outputs.items()}
            self.reply(200, {"id": image_id, "sigma": sigma, "size": f"{clean.shape[1]} × {clean.shape[0]}",
                             "images": {key: png_data(value) for key, value in all_images.items()}, "scores": scores})
        except (ValueError, KeyError, json.JSONDecodeError, FileNotFoundError, binascii.Error, cv2.error) as error:
            self.reply(400, {"error": str(error)})


if __name__ == "__main__":
    if not MODEL.exists():
        raise SystemExit("Run `python -m src.experiment` before starting the demo.")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Demo: http://127.0.0.1:8765")
    server.serve_forever()
