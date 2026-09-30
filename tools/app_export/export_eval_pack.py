"""
Pack a whole IO-VNBD drive for the app's full-drive evaluator
(app/tool/evaluate.dart), plus OpenStreetMap roads along its route.

The pack uses the project's own preprocessing (process_file_v5, the same
column mapping as training) so the evaluator sees exactly what the model saw.

Output (git-ignored, under data/eval/):
  <name>.idreval     one JSON header line, then little-endian column blocks
  roads/<cell>.json  OSM roads per 0.02° cell (same cells/format as the app)

Usage (from repo root):
    tools/app_export/.venv/bin/python tools/app_export/export_eval_pack.py            # Drive M
    tools/app_export/.venv/bin/python tools/app_export/export_eval_pack.py --drive S-Y.csv --name drive_y

Note: only Drive M is guaranteed unseen in training (see README); other drives
may have been training data.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

from src.baseline_graphs import detect_smartphone_columns, find_paired_files, numeric_series  # noqa: E402
from src.crossdrive_pipeline_v5 import (  # noqa: E402
    DELAY_TAPS,
    apply_cruise_lock,
    apply_enhanced_zupt,
    ekf_with_acceleration_feedforward,
    kinematic_plausibility_gate,
    process_file_v5,
)
from export_model import load_model  # noqa: E402
from export_osm import compact, fetch, query  # noqa: E402

OUT = os.path.join(REPO, "data", "eval")
CELL_DEG = 0.02  # must match app/lib/data/roads/road_repository.dart
MARGIN_DEG = 0.01


def find_drive(name):
    for s, v in find_paired_files():
        if os.path.basename(s) == name:
            return s, v
    raise SystemExit(f"{name} not found under data/raw/IO-VNBD")


def clean(series):
    return pd.Series(series).interpolate().ffill().bfill().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="S-M.csv", help="smartphone CSV file name (S-*.csv)")
    ap.add_argument("--name", default="drive_m")
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "results", "crossdrive_v5_model.pth"))
    ap.add_argument("--no-roads", action="store_true")
    args = ap.parse_args()

    s_path, v_path = find_drive(args.drive)
    print(f"drive: {os.path.relpath(s_path, REPO)}")
    m = process_file_v5(s_path, v_path, for_training=False, stride=1)
    n = len(m["time_axis"])

    # Python reference speeds over the whole drive (for a full-scale parity check).
    import torch
    model, ckpt, _ = load_model(args.checkpoint)
    xn = ((m["X"] - np.asarray(ckpt["X_mean"])) / np.asarray(ckpt["X_std"])).astype(np.float32)
    with torch.no_grad():
        pred = np.concatenate([model(torch.from_numpy(xn[i:i + 8192])).numpy().ravel() for i in range(0, len(xn), 8192)])
    raw = np.zeros(n)
    raw[DELAY_TAPS:] = np.maximum(pred * float(ckpt["y_std"]) + float(ckpt["y_mean"]), 0.0)
    enhanced = apply_enhanced_zupt(
        apply_cruise_lock(
            ekf_with_acceleration_feedforward(kinematic_plausibility_gate(raw, m["lin_ax"]), m["lin_ax"]),
            m["lin_ax"], m["gz"],
        ),
        m["ts_feats"],
    )

    s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1").iloc[:n]
    cols = detect_smartphone_columns(s_df)
    sensors = {k: clean(numeric_series(s_df, cols[k])) for k in ["ax", "ay", "az", "grav_x", "grav_y", "grav_z", "gx", "gz"]}
    pitch_col = next(c for c in s_df.columns if "GYROSCOPE" in c.upper() and "PITCH" in c.upper())
    gyro_up = clean(numeric_series(s_df, pitch_col))

    v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1").iloc[:n]
    v_df.columns = [c.strip() for c in v_df.columns]
    lat = clean(pd.to_numeric(v_df["Latitude (degrees)"], errors="coerce"))
    lon = clean(pd.to_numeric(v_df["Longitude (degrees)"], errors="coerce"))
    t = m["time_axis"] - m["time_axis"][0]

    columns = [
        ("t", "f8", t),
        ("ax", "f4", sensors["ax"]), ("ay", "f4", sensors["ay"]), ("az", "f4", sensors["az"]),
        ("grav_x", "f4", sensors["grav_x"]), ("grav_y", "f4", sensors["grav_y"]), ("grav_z", "f4", sensors["grav_z"]),
        # Physical gyro as in the app's replays: logger "Yaw", "Roll", "Pitch" (= up).
        ("gyro_x", "f4", sensors["gx"]), ("gyro_y", "f4", sensors["gz"]), ("gyro_up", "f4", gyro_up),
        ("gt_speed", "f4", m["gt"]["velocity_ms"]), ("gt_heading", "f4", m["gt"]["heading"]),
        ("lat", "f8", lat), ("lon", "f8", lon),
        ("py_enhanced", "f4", enhanced),
    ]
    os.makedirs(OUT, exist_ok=True)
    header = {
        "format": "idreval/1",
        "drive": os.path.basename(s_path).replace("S-", "").replace(".csv", ""),
        "n": n,
        "idle_baseline": float(m["idle_base"]),
        "columns": [{"name": c, "dtype": d} for c, d, _ in columns],
    }
    path = os.path.join(OUT, f"{args.name}.idreval")
    with open(path, "wb") as f:
        f.write((json.dumps(header) + "\n").encode())
        for _, dtype, arr in columns:
            f.write(np.ascontiguousarray(arr, dtype="<" + dtype).tobytes())
    km = float(np.sum(m["gt"]["velocity_ms"]) * 0.1 / 1000)
    print(f"wrote {os.path.relpath(path, REPO)}: {n:,} samples, {t[-1] / 3600:.2f} h, {km:.1f} km, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")

    if args.no_roads:
        return
    cells = sorted({(math.floor(a / CELL_DEG), math.floor(b / CELL_DEG)) for a, b in zip(lat[::50], lon[::50])})
    roads_dir = os.path.join(OUT, "roads")
    os.makedirs(roads_dir, exist_ok=True)
    todo = [c for c in cells if not os.path.exists(os.path.join(roads_dir, f"{c[0]}_{c[1]}.json"))]
    print(f"route touches {len(cells)} map cells; downloading {len(todo)} (cached: {len(cells) - len(todo)})")
    for i, (cy, cx) in enumerate(todo, 1):
        s0, w0 = cy * CELL_DEG, cx * CELL_DEG
        roads = compact(fetch(query(s0 - MARGIN_DEG, w0 - MARGIN_DEG, s0 + CELL_DEG + MARGIN_DEG, w0 + CELL_DEG + MARGIN_DEG)))
        with open(os.path.join(roads_dir, f"{cy}_{cx}.json"), "w") as f:
            json.dump(roads, f, separators=(",", ":"))
        print(f"  [{i}/{len(todo)}] cell {cy}_{cx}: {len(roads['ways'])} ways")
        time.sleep(1.5)  # be polite to the public Overpass servers


if __name__ == "__main__":
    main()
