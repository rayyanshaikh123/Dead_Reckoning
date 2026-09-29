"""
Generate golden vectors for the Dart engine's parity tests.

Runs a synthetic 90 s drive (idle, accelerate, cruise with a turn, brake, stop)
through the project's own v5 pipeline functions, so the Dart port can be
checked stage by stage without the IO-VNBD dataset:

  sensors -> extract_features_v5 -> delay windows -> IDNN v5 -> gate -> EKF
          -> cruise lock -> enhanced ZUPT -> GNSS calibrator (at two entries)

plus dead_reckon_2d / map_match_to_road on a synthetic heading/road.

Usage (from repo root):
    tools/app_export/.venv/bin/python tools/app_export/make_goldens.py
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

from src.crossdrive_pipeline_v5 import (  # noqa: E402
    DELAY_TAPS,
    OnlineGNSSCalibrator,
    apply_cruise_lock,
    apply_enhanced_zupt,
    create_delay_windows_v5,
    ekf_with_acceleration_feedforward,
    extract_features_v5,
    kinematic_plausibility_gate,
)
from src.simulate_blackout import dead_reckon_2d, map_match_to_road  # noqa: E402
from export_model import load_model  # noqa: E402

DT = 0.1


def synthetic_drive(seed=7):
    """Phone x-axis forward, z up (Android convention: face-up gravity = +9.81 z)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 90, DT)
    n = len(t)

    a_long = np.zeros(n)
    a_long[(t >= 10) & (t < 25)] = 1.0
    a_long[(t >= 45) & (t < 55)] = -1.0
    a_long[(t >= 70) & (t < 75)] = -1.0
    speed = np.clip(np.cumsum(a_long) * DT, 0.0, None)

    yaw_rate = np.zeros(n)
    yaw_rate[(t >= 30) & (t < 36)] = 0.12

    tilt = np.deg2rad(3.0)
    grav = np.stack([
        np.full(n, 9.80665 * np.sin(tilt)),
        np.full(n, 0.15),
        np.full(n, 9.80665 * np.cos(tilt)),
    ], axis=1)
    grav += rng.normal(0, 0.002, size=grav.shape)

    idle_noise = 0.08
    road_noise = 0.02 * speed
    vib = rng.normal(0, 1, size=(n, 3)) * (idle_noise + road_noise)[:, None]
    lin = np.stack([a_long, speed * yaw_rate, np.zeros(n)], axis=1) + vib

    acc = grav + lin
    gyro = np.stack([
        rng.normal(0, 0.004, n),
        rng.normal(0, 0.004, n),
        yaw_rate + rng.normal(0, 0.004, n),
    ], axis=1)
    heading = (40.0 + np.rad2deg(np.cumsum(yaw_rate) * DT)) % 360.0
    return t, acc, grav, gyro, speed, heading


def predict(model, X, x_mean, x_std, y_mean, y_std):
    xn = ((X - x_mean) / x_std).astype(np.float32)
    with torch.no_grad():
        p = model(torch.from_numpy(xn)).numpy().ravel()
    return np.maximum(p * y_std + y_mean, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "results", "crossdrive_v5_model.pth"))
    ap.add_argument("--out", default=os.path.join(REPO, "app", "test", "golden", "v5_pipeline.json"))
    args = ap.parse_args()

    model, ckpt, _ = load_model(args.checkpoint)
    x_mean, x_std = np.asarray(ckpt["X_mean"]), np.asarray(ckpt["X_std"])
    y_mean, y_std = float(ckpt["y_mean"]), float(ckpt["y_std"])

    t, acc, grav, gyro, gnss_speed, heading = synthetic_drive()
    n = len(t)
    cols = ["ax", "ay", "az", "grav_x", "grav_y", "grav_z", "gx", "gy", "gz"]
    df = pd.DataFrame(np.hstack([acc, grav, gyro]), columns=cols)
    s_cols = {c: c for c in cols}

    feats, raw_meta = extract_features_v5(df, s_cols, n, target_speed=gnss_speed)
    lin_ax, gz, idle_base = raw_meta[6], raw_meta[7], raw_meta[8]

    X, _ = create_delay_windows_v5(feats, gnss_speed, delay_taps=DELAY_TAPS, stride=1)
    raw = np.zeros(n)
    raw[DELAY_TAPS:] = predict(model, X, x_mean, x_std, y_mean, y_std)

    gated = kinematic_plausibility_gate(raw, lin_ax)
    ekf = ekf_with_acceleration_feedforward(gated, lin_ax)
    cruise = apply_cruise_lock(ekf, lin_ax, gz)
    enhanced = apply_enhanced_zupt(cruise, feats)

    lin_norm = np.sqrt(feats[:, 0].astype(np.float64) ** 2 + feats[:, 1] ** 2 + feats[:, 2] ** 2)
    ac_var = pd.Series(lin_norm).rolling(10, min_periods=1).var().fillna(0.0).to_numpy()

    cal = OnlineGNSSCalibrator(history_len=50)
    calib = []
    for entry, length in [(380, 150), (500, 200)]:  # cruising entry, braking entry
        bias = cal.calibrate_at_blackout_entry(enhanced, gnss_speed, entry)
        sl = slice(entry, entry + length)
        out = cal.apply_calibration(enhanced[sl].copy(), bias, ac_vibration=ac_var[sl])
        calib.append({"entry": entry, "length": length, "bias": bias, "speed": out.tolist()})

    # 2D: dead reckoning with a heading, and map-matching onto a curved road.
    dt_arr = np.full(n, DT)
    dr_x, dr_y = dead_reckon_2d(gnss_speed, heading, dt_arr, 0.0, 0.0)
    s_road = np.linspace(0, 1200, 400)
    road_x = 300 * np.sin(s_road / 400.0)
    road_y = s_road
    cum = np.cumsum(gnss_speed * dt_arr)
    mm_x, mm_y = map_match_to_road(cum, road_x, road_y)

    golden = {
        "dt": DT,
        "delay_taps": DELAY_TAPS,
        "idle_baseline": float(idle_base),
        "inputs": {
            "acc": acc.tolist(), "grav": grav.tolist(), "gyro": gyro.tolist(),
            "gnss_speed": gnss_speed.tolist(), "heading": heading.tolist(),
        },
        "features": feats.astype(np.float64).tolist(),
        "model_probe": {
            "x": X[[0, 150, 300, 450, 600]].astype(np.float64).tolist(),
            "y": raw[[DELAY_TAPS + i for i in (0, 150, 300, 450, 600)]].tolist(),
        },
        "raw": raw.tolist(),
        "gated": gated.tolist(),
        "ekf": ekf.tolist(),
        "cruise": cruise.tolist(),
        "enhanced": enhanced.tolist(),
        "ac_var": ac_var.tolist(),
        "calibration": calib,
        "dead_reckon": {"x": dr_x.tolist(), "y": dr_y.tolist()},
        "map_match": {
            "road_x": road_x.tolist(), "road_y": road_y.tolist(),
            "cum": cum.tolist(), "x": mm_x.tolist(), "y": mm_y.tolist(),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(golden, f)
    print(f"wrote {args.out}  ({n} samples, idle baseline {idle_base:.4f})")
    print(f"raw speed range {raw.min():.2f}..{raw.max():.2f} m/s, "
          f"enhanced {enhanced.min():.2f}..{enhanced.max():.2f} m/s")
    print("calibration biases:", [round(c["bias"], 4) for c in calib])


if __name__ == "__main__":
    main()
