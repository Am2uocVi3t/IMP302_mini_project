"""Small NumPy neural residual model with Adam optimization."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ResidualMLP:
    def __init__(self, n_features: int, hidden: int, seed: int = 302):
        rng = np.random.default_rng(seed)
        self.weights = {
            "w1": (rng.normal(0, np.sqrt(2 / n_features), (n_features, hidden))).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "w2": (rng.normal(0, np.sqrt(1 / hidden), (hidden, 1))).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 12, batch_size: int = 2048) -> None:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-5
        x = ((x - self.mean) / self.std).astype(np.float32)
        y = y.reshape(-1, 1).astype(np.float32)
        rng = np.random.default_rng(302)
        first = {k: np.zeros_like(v) for k, v in self.weights.items()}
        second = {k: np.zeros_like(v) for k, v in self.weights.items()}
        step = 0
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
                xb, target = x[ids], y[ids]
                z = xb @ self.weights["w1"] + self.weights["b1"]
                a = np.maximum(z, 0)
                pred = a @ self.weights["w2"] + self.weights["b2"]
                dp = 2 * (pred - target) / len(ids)
                grads = {
                    "w2": a.T @ dp,
                    "b2": dp.sum(axis=0),
                    "w1": xb.T @ ((dp @ self.weights["w2"].T) * (z > 0)),
                    "b1": ((dp @ self.weights["w2"].T) * (z > 0)).sum(axis=0),
                }
                step += 1
                for key, grad in grads.items():
                    first[key] = 0.9 * first[key] + 0.1 * grad
                    second[key] = 0.999 * second[key] + 0.001 * (grad * grad)
                    self.weights[key] -= 0.001 * (first[key] / (1 - 0.9**step)) / (np.sqrt(second[key] / (1 - 0.999**step)) + 1e-8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = ((x - self.mean) / self.std).astype(np.float32)
        out = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), 32768):
            part = x[start:start + 32768]
            out[start:start + len(part)] = (np.maximum(part @ self.weights["w1"] + self.weights["b1"], 0) @ self.weights["w2"] + self.weights["b2"]).ravel()
        return out

    def save(self, path: Path, h_map: dict[int, float], texture: bool) -> None:
        np.savez_compressed(path, **self.weights, mean=self.mean, std=self.std,
                            h_levels=np.array(sorted(h_map)), h_values=np.array([h_map[k] for k in sorted(h_map)]),
                            texture=int(texture))

    @classmethod
    def load(cls, path: Path) -> tuple["ResidualMLP", dict[int, float], bool]:
        data = np.load(path)
        model = cls(len(data["mean"]), len(data["b1"]))
        model.weights = {k: data[k] for k in ("w1", "b1", "w2", "b2")}
        model.mean, model.std = data["mean"], data["std"]
        return model, dict(zip(data["h_levels"].astype(int).tolist(), data["h_values"].astype(float).tolist())), bool(data["texture"])


def restore(outputs: dict, model: ResidualMLP, sigma: float, texture: bool) -> np.ndarray:
    from .core import features
    correction = model.predict(features(outputs, sigma, texture)).reshape(outputs["nlm"].shape)
    return np.clip(outputs["nlm"].astype(np.float32) + 255 * correction, 0, 255).astype(np.uint8)
