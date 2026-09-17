"""
Generate full diagnostic plots for IDNN v4 in results/plots/crossdrive_v4/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.idnn_model_v4 import IDNNv4
from src.crossdrive_pipeline_v4 import (
    process_file_v4,
    predict_v4,
    kinematic_plausibility_gate,
    ekf_with_acceleration_feedforward,
    apply_cruise_lock,
    apply_enhanced_zupt,
    MODEL_PATH,
    DELAY_TAPS,
    DEVICE,
)
from src.baseline_graphs import find_paired_files

OUTPUT_DIR = "results/plots/crossdrive_v4"


def generate_v4_plots():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs = find_paired_files()
    primary = pairs[0]

    print("Loading IDNN v4 model...")
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    cfg = ckpt["config"]
    model = IDNNv4(
        n_features=cfg["n_features"],
        delay_taps=cfg["delay_taps"],
        n_outputs=cfg["n_outputs"],
        hidden_sizes=cfg["hidden_sizes"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.X_mean = ckpt["X_mean"]
    model.X_std = ckpt["X_std"]
    model.y_mean = ckpt["y_mean"]
    model.y_std = ckpt["y_std"]

    print("Processing Drive M for v4 plots...")
    m_data = process_file_v4(primary[0], primary[1], for_training=False, stride=1)
    time_axis = m_data["time_axis"]
    gt_speed = m_data["gt"]["velocity_ms"]
    n_full = len(time_axis)

    v4_pred = predict_v4(model, m_data["X"])
    v4_raw = np.zeros(n_full)
    v4_raw[DELAY_TAPS:] = v4_pred

    v4_gated = kinematic_plausibility_gate(v4_raw, m_data["lin_ax"])
    v4_ekf = ekf_with_acceleration_feedforward(v4_gated, m_data["lin_ax"])
    v4_cruise = apply_cruise_lock(v4_ekf, m_data["lin_ax"], m_data["gz"])
    v4_enh = apply_enhanced_zupt(v4_cruise, m_data["ts_feats"])

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    # 1. Speed Tracking (first 500s)
    fig, ax = plt.subplots(figsize=(14, 5))
    t_sub = time_axis[:5000]
    ax.plot(t_sub, gt_speed[:5000] * 3.6, label="Ground Truth (GNSS)", color="black", lw=1.5)
    ax.plot(t_sub, v4_raw[:5000] * 3.6, label="v4 Raw Speed", color="tab:orange", alpha=0.6, lw=1.2)
    ax.plot(t_sub, v4_enh[:5000] * 3.6, label="v4 Enhanced Speed", color="tab:blue", lw=1.5)
    ax.set(title="IDNN v4 Speed Tracking on 100% Blind Drive M", xlabel="Time (s)", ylabel="Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_v4_speed_tracking.png", dpi=300)
    plt.close(fig)

    # 2. Cumulative Distance
    dt = np.diff(time_axis, prepend=time_axis[0])
    dt[0] = 0.1
    d_gt = np.cumsum(gt_speed * dt) / 1000.0
    d_raw = np.cumsum(v4_raw * dt) / 1000.0
    d_enh = np.cumsum(v4_enh * dt) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis / 60.0, d_gt, label="Ground Truth (105.1 km)", color="black", lw=2.0)
    ax.plot(time_axis / 60.0, d_raw, label=f"v4 Raw ({d_raw[-1]:.1f} km)", color="tab:orange", lw=1.5, ls="--")
    ax.plot(time_axis / 60.0, d_enh, label=f"v4 Enhanced ({d_enh[-1]:.1f} km)", color="tab:blue", lw=1.8)
    ax.set(title="Cumulative Distance: IDNN v4 vs Ground Truth", xlabel="Drive Duration (min)", ylabel="Distance (km)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_v4_cumulative_distance.png", dpi=300)
    plt.close(fig)

    # 3. Error Histogram
    err_raw = (v4_raw - gt_speed) * 3.6
    err_enh = (v4_enh - gt_speed) * 3.6

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(err_raw, bins=80, range=(-25, 25), alpha=0.5, label=f"v4 Raw (RMSE: {np.sqrt(np.mean(err_raw**2)):.1f} km/h)", color="tab:orange")
    ax.hist(err_enh, bins=80, range=(-25, 25), alpha=0.6, label=f"v4 Enhanced (RMSE: {np.sqrt(np.mean(err_enh**2)):.1f} km/h)", color="tab:blue")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(title="IDNN v4 Speed Error Distribution (km/h)", xlabel="Error (km/h)", ylabel="Sample Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_v4_error_histogram.png", dpi=300)
    plt.close(fig)

    print(f"IDNN v4 plots saved successfully to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_v4_plots()

