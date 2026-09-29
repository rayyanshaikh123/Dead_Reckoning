"""
Export the IDNN v5 checkpoint for the Flutter app's pure-Dart inference engine.

Writes (default: app/assets/models/):
  idnn_v5.bin   little-endian float32 blob, tensors concatenated in manifest order
  idnn_v5.json  config, input/output normalisation, tensor manifest (name, shape, offset)

BatchNorm1d layers are folded into the preceding Linear layer (eval mode), so the
Dart side only needs Linear + GELU:
    W' = W * gamma / sqrt(var + eps)        (row-wise)
    b' = (b - mean) * gamma / sqrt(var + eps) + beta

Usage (from repo root):
    tools/app_export/.venv/bin/python tools/app_export/export_model.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from src.idnn_model_v5 import IDNNv5  # noqa: E402

FEATURE_ORDER = [
    "lin_ax", "lin_ay", "lin_az",
    "grav_x", "grav_y", "grav_z",
    "gyro_x", "gyro_y", "gyro_z",
    "jerk_ax", "jerk_ay", "jerk_az",
    "jerk_gx", "jerk_gy", "jerk_gz",
    "a_horiz_norm",
    "pitch",
    "e_vib_norm",
]


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = IDNNv5(
        n_features=cfg["n_features"],
        delay_taps=cfg["delay_taps"],
        n_outputs=cfg["n_outputs"],
        hidden_sizes=tuple(cfg["hidden_sizes"]),
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, cfg


def fold(linear, bn):
    w = linear.weight.detach().double().numpy()
    b = linear.bias.detach().double().numpy()
    if bn is None:
        return w, b
    gamma = bn.weight.detach().double().numpy()
    beta = bn.bias.detach().double().numpy()
    mean = bn.running_mean.detach().double().numpy()
    var = bn.running_var.detach().double().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    return w * scale[:, None], (b - mean) * scale + beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "results", "crossdrive_v5_model.pth"))
    ap.add_argument("--out", default=os.path.join(REPO, "app", "assets", "models"))
    args = ap.parse_args()

    model, ckpt, cfg = load_model(args.checkpoint)
    layers = [
        ("in_proj", fold(model.in_proj[0], model.in_proj[1])),
        ("block1", fold(model.block1[0], model.block1[1])),
        ("block2", fold(model.block2[0], model.block2[1])),
        ("head", fold(model.head, None)),
    ]

    x_mean = np.asarray(ckpt["X_mean"], dtype=np.float64).ravel()
    x_std = np.asarray(ckpt["X_std"], dtype=np.float64).ravel()
    tensors = [("x_mean", x_mean), ("x_std", x_std)]
    for name, (w, b) in layers:
        tensors.append((f"{name}.weight", w))
        tensors.append((f"{name}.bias", b))

    manifest, offset, blobs = [], 0, []
    for name, arr in tensors:
        a = np.ascontiguousarray(arr, dtype="<f4")
        manifest.append({"name": name, "shape": list(a.shape), "offset": offset, "count": int(a.size)})
        blobs.append(a.tobytes())
        offset += a.size

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "idnn_v5.bin"), "wb") as f:
        for blob in blobs:
            f.write(blob)

    meta = {
        "model": "IDNN v5",
        "source": os.path.relpath(args.checkpoint, REPO),
        "n_features": cfg["n_features"],
        "delay_taps": cfg["delay_taps"],
        "hidden_sizes": list(cfg["hidden_sizes"]),
        "activation": "gelu_erf",
        "sample_rate_hz": 10.0,
        "window_order": "newest_first",
        "feature_order": FEATURE_ORDER,
        "y_mean": float(ckpt["y_mean"]),
        "y_std": float(ckpt["y_std"]),
        "tensors": manifest,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    with open(os.path.join(args.out, "idnn_v5.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Sanity check: folded float32 weights reproduce the torch model.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, x_mean.size))
    with torch.no_grad():
        ref = model(torch.from_numpy(x).float()).numpy().ravel()
    h = x
    for i, (name, (w, b)) in enumerate(layers):
        h = h @ w.astype(np.float32).T + b.astype(np.float32)
        if name != "head":
            h = 0.5 * h * (1.0 + np.vectorize(__import__("math").erf)(h / np.sqrt(2.0)))
    err = float(np.max(np.abs(h.ravel() - ref)))
    print(f"exported {offset:,} floats ({meta['parameter_count']:,} params) -> {args.out}")
    print(f"folded-vs-torch max abs diff (normalised output): {err:.2e}")
    if err > 1e-4:
        raise SystemExit("fold check failed")


if __name__ == "__main__":
    main()
