"""
Cut the five README blackout scenarios out of Drive M as replay files for the app,
and record the Python IDNN v5 benchmark result for each so the Dart engine can be
checked against it.

For every scenario (start index and duration from src/simulate_blackout.py):
  * replay window = 5 min of lead-in (GNSS healthy: filters, calibrator,
                    gyro scale/bias learn)
                    + the outage + 30 s after it
  * sensors are exported exactly as the model saw them in training, i.e. using
    the column mapping from detect_smartphone_columns (gyro = [Yaw, Yaw, Roll];
    see CANONICAL_FRAME.md)
  * ground truth = vehicle RTK speed, heading, lat/lon

Writes:
  app/assets/replays/<id>.json        replay data
  app/assets/replays/index.json       scenario list + Python reference metrics

Usage (from repo root, needs data/raw/IO-VNBD Drive M):
    tools/app_export/.venv/bin/python tools/app_export/export_replay.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)  # the pipeline resolves data/ relative to the repo root

from src.baseline_graphs import (  # noqa: E402
    detect_smartphone_columns,
    find_paired_files,
    numeric_series,
)
from src.crossdrive_pipeline_v5 import (  # noqa: E402
    DELAY_TAPS,
    OnlineGNSSCalibrator,
    apply_cruise_lock,
    apply_enhanced_zupt,
    ekf_with_acceleration_feedforward,
    kinematic_plausibility_gate,
    process_file_v5,
)
from src.simulate_blackout import map_match_to_road  # noqa: E402
from export_model import load_model  # noqa: E402

# (id, name, start index, outage seconds) — src/simulate_blackout.py
SCENARIOS = [
    ("highway_tunnel", "Straight highway tunnel", 92100, 30.0),
    ("short_underpass", "Short underpass", 15000, 30.0),
    ("medium_tunnel", "Medium tunnel (stop & go)", 35000, 60.0),
    ("mountain_tunnel", "Long mountain tunnel", 50000, 120.0),
    ("city_canyon", "Complex city canyon", 75000, 90.0),
]
LEAD_IN = 3000  # samples (5 min): filters, calibrator and gyro scale/bias warm up
TAIL = 300  # samples (30 s)
ROAD_EXTRA = 250  # simulate_blackout extends the road 250 samples past the exit


def drive_m():
    for s, v in find_paired_files():
        if os.path.basename(s) == "S-M.csv":
            return s, v
    raise SystemExit("Drive M (S-M.csv / V-M.csv) not found under data/raw/IO-VNBD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "results", "crossdrive_v5_model.pth"))
    ap.add_argument("--out", default=os.path.join(REPO, "app", "assets", "replays"))
    args = ap.parse_args()

    s_path, v_path = drive_m()
    print(f"Drive M: {os.path.relpath(s_path, REPO)}")

    model, ckpt, _ = load_model(args.checkpoint)
    x_mean, x_std = np.asarray(ckpt["X_mean"]), np.asarray(ckpt["X_std"])
    y_mean, y_std = float(ckpt["y_mean"]), float(ckpt["y_std"])

    m = process_file_v5(s_path, v_path, for_training=False, stride=1)
    n = len(m["time_axis"])
    with torch.no_grad():
        xn = ((m["X"] - x_mean) / x_std).astype(np.float32)
        pred = np.concatenate([
            model(torch.from_numpy(xn[i:i + 8192])).numpy().ravel() for i in range(0, len(xn), 8192)
        ])
    raw = np.zeros(n)
    raw[DELAY_TAPS:] = np.maximum(pred * y_std + y_mean, 0.0)
    enhanced = apply_enhanced_zupt(
        apply_cruise_lock(
            ekf_with_acceleration_feedforward(kinematic_plausibility_gate(raw, m["lin_ax"]), m["lin_ax"]),
            m["lin_ax"], m["gz"],
        ),
        m["ts_feats"],
    )
    feats = m["ts_feats"]
    lin_norm = np.sqrt(m["lin_ax"] ** 2 + feats[:, 1] ** 2 + feats[:, 2] ** 2)
    ac_var = pd.Series(lin_norm).rolling(10, min_periods=1).var().fillna(0.0).to_numpy()

    gt = m["gt"]
    time_axis = m["time_axis"]
    dt_arr = np.diff(time_axis, prepend=time_axis[0])
    dt_arr[0] = 0.1

    # Raw sensor columns, mapped exactly as in training.
    s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1").iloc[:n]
    cols = detect_smartphone_columns(s_df)
    series = {k: pd.Series(numeric_series(s_df, cols[k])).interpolate().ffill().bfill().to_numpy()
              for k in ["ax", "ay", "az", "grav_x", "grav_y", "grav_z", "gx", "gy", "gz"]}
    # The logger's "Pitch" gyro column is the real yaw rate (rotation about the
    # vertical; see CANONICAL_FRAME.md). The model never saw it, but heading
    # propagation during an outage needs it.
    pitch_col = next(c for c in s_df.columns if "GYROSCOPE" in c.upper() and "PITCH" in c.upper())
    gyro_up = pd.Series(numeric_series(s_df, pitch_col)).interpolate().ffill().bfill().to_numpy()
    yaw_col = next(c for c in s_df.columns if "ORIENTATION" in c.upper() and "YAW" in c.upper())
    phone_yaw = pd.Series(numeric_series(s_df, yaw_col)).interpolate().ffill().bfill().to_numpy()

    v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1").iloc[:n]
    v_df.columns = [c.strip() for c in v_df.columns]
    lat = pd.to_numeric(v_df["Latitude (degrees)"], errors="coerce").interpolate().ffill().bfill().to_numpy()
    lon = pd.to_numeric(v_df["Longitude (degrees)"], errors="coerce").interpolate().ffill().bfill().to_numpy()

    os.makedirs(args.out, exist_ok=True)
    index = {
        "drive": "IO-VNBD Drive M (Driver B)",
        "idle_baseline": float(m["idle_base"]),
        "sample_rate_hz": 10.0,
        "scenarios": [],
    }

    for sid, name, start, dur in SCENARIOS:
        n_out = int(dur / 0.1)
        end = min(start + n_out, n - 1)
        sl = slice(start, end + 1)

        # --- Python reference, same maths as run_blackout_scenario (v5 branch) ---
        cal = OnlineGNSSCalibrator(history_len=50)
        bias = cal.calibrate_at_blackout_entry(enhanced, gt["velocity_ms"], start)
        v_cal = cal.apply_calibration(enhanced[sl].copy(), bias, ac_vibration=ac_var[sl])
        true_dist = float(np.sum(gt["velocity_ms"][sl] * dt_arr[sl]))
        pred_dist = float(np.sum(v_cal * dt_arr[sl]))
        road_end = min(end + ROAD_EXTRA, n - 1)
        mx, my = map_match_to_road(np.cumsum(v_cal * dt_arr[sl]), gt["gt_x"][start:road_end], gt["gt_y"][start:road_end])
        exit_err = float(np.hypot(mx[-1] - gt["gt_x"][end], my[-1] - gt["gt_y"][end]))
        ref = {
            "bias": float(bias),
            "true_distance_m": true_dist,
            "pred_distance_m": pred_dist,
            "along_track_drift_pct": abs(pred_dist - true_dist) / true_dist * 100,
            "exit_error_m": exit_err,
            "exit_drift_pct": exit_err / true_dist * 100,
        }

        # --- replay window ---
        w0 = max(start - LEAD_IN, 0)
        w1 = min(max(end + TAIL, road_end) + 1, n)
        w = slice(w0, w1)
        r6 = lambda a: [round(float(x), 6) for x in a]  # noqa: E731
        replay = {
            "id": sid,
            "name": name,
            "source_index": w0,
            "outage_start": start - w0,
            "outage_end": end - w0,  # inclusive
            "road_end": road_end - w0,
            "idle_baseline": float(m["idle_base"]),
            "t": r6(time_axis[w] - time_axis[w0]),
            "sensors": {k: r6(v[w]) for k, v in series.items()},
            "gyro_up": r6(gyro_up[w]),
            "phone_yaw_deg": r6(phone_yaw[w]),
            "gt": {
                "speed_ms": r6(gt["velocity_ms"][w]),
                "heading_deg": r6(gt["heading"][w]),
                "lat": [round(float(x), 8) for x in lat[w]],
                "lon": [round(float(x), 8) for x in lon[w]],
            },
            "python": {
                "raw": r6(raw[w]),
                "enhanced": r6(enhanced[w]),
            },
        }
        with open(os.path.join(args.out, f"{sid}.json"), "w") as f:
            json.dump(replay, f, separators=(",", ":"))

        index["scenarios"].append({
            "id": sid, "name": name, "outage_s": dur,
            "samples": w1 - w0, "reference": ref,
        })
        print(f"{name:28s} dist {true_dist:7.1f} m  drift {ref['along_track_drift_pct']:5.1f}%  "
              f"exit {exit_err:6.1f} m ({ref['exit_drift_pct']:4.1f}%)  bias {bias:+.3f}")

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"idle baseline {index['idle_baseline']:.4f}; wrote {len(SCENARIOS)} replays -> {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
