"""
Clean, High-Fidelity GNSS Blackout Simulation & 2D Dead Reckoning Pipeline.

Evaluates navigation performance during GNSS outages across 5 distinct, non-overlapping scenarios:
  1. Straight Highway Tunnel (30s, ~650m)
  2. Short Underpass         (30s, ~220m)
  3. Medium Tunnel           (60s, ~450m)
  4. Long Mountain Tunnel    (120s, ~1.7km)
  5. Complex City Canyon     (90s, ~600m)

Models compared in 2D space:
  - Baseline INS (Double Integration: ∬ a dt²)
  - IDNN v1 Dead Reckoning
  - IDNN v2 Dead Reckoning
  - IDNN v3 (KF + ZUPT) + Map Matching
  - IDNN v4 (Physics Gated + EKF Feedforward) + Map Matching
  - IDNN v5 (Idle Normalized + Online Pre-Blackout GNSS Calibrated) + Map Matching
  - Vehicle Ground Truth (GNSS reference)

Outputs generated in results/plots/blackout_simulation/:
  - 01_straight_highway_tunnel_30s_map.png
  - 02_short_underpass_30s_map.png
  - 03_medium_tunnel_60s_map.png
  - 04_long_mountain_tunnel_120s_map.png
  - 05_complex_city_canyon_90s_map.png
  - 06_error_vs_time.png
  - 07_along_track_vs_mapmatch_comparison.png
  - 08_comprehensive_blackout_table.png
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.baseline_graphs import (
    find_paired_files,
    detect_smartphone_columns,
    detect_vehicle_columns,
    numeric_series,
    compute_vehicle_ground_truth,
    compute_baseline_inertial,
    G,
)
from src.idnn_model_v3 import IDNNv3
from src.idnn_pipeline_v3 import (
    process_file as process_file_v3,
    predict_v3,
    kalman_filter_speed as kf_v3,
    apply_zupt as zupt_v3,
    load_v1_predictions,
    load_v2_predictions,
    MODEL_PATH as V3_MODEL_PATH,
    DELAY_TAPS as V3_DELAY_TAPS,
    DEVICE,
)

OUTPUT_DIR = "results/plots/blackout_simulation"


# ============================================================
# 2D PROPAGATION & MAP MATCHING
# ============================================================

def dead_reckon_2d(speed, heading_deg, dt_arr, start_x, start_y):
    n = len(speed)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = start_x
    y[0] = start_y

    heading_rad = np.radians(heading_deg)
    for i in range(1, n):
        dt = dt_arr[i]
        dx = speed[i] * np.sin(heading_rad[i]) * dt
        dy = speed[i] * np.cos(heading_rad[i]) * dt
        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy

    return x, y


def map_match_to_road(cumulative_distance, road_x, road_y):
    dx = np.diff(road_x)
    dy = np.diff(road_y)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    road_cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0.0)

    matched_x = np.interp(cumulative_distance, road_cum_dist, road_x)
    matched_y = np.interp(cumulative_distance, road_cum_dist, road_y)
    return matched_x, matched_y


# ============================================================
# SCENARIO SIMULATOR
# ============================================================

def run_blackout_scenario(
    prefix,
    name,
    start_idx,
    duration_s,
    time_axis,
    gt,
    inertial,
    v1_speed,
    v2_speed,
    v3_speed,
    v4_speed,
    v5_speed,
    phone_yaw_deg,
    ac_vibration=None,
):
    dt_arr = np.diff(time_axis, prepend=time_axis[0])
    dt_arr[0] = 0.1

    n_samples = int(duration_s / 0.1)
    end_idx = min(start_idx + n_samples, len(time_axis) - 1)
    actual_duration = time_axis[end_idx] - time_axis[start_idx]

    idx_slice = slice(start_idx, end_idx + 1)
    t_slice = time_axis[idx_slice] - time_axis[start_idx]
    dt_slice = dt_arr[idx_slice]

    gt_x = gt["gt_x"][idx_slice]
    gt_y = gt["gt_y"][idx_slice]
    gt_speed = gt["velocity_ms"][idx_slice]
    gt_heading = gt["heading"][idx_slice]
    true_distance = np.sum(gt_speed * dt_slice)

    start_x = gt_x[0]
    start_y = gt_y[0]

    # Phone Compass Heading (zeroed at entry)
    yaw_offset = gt_heading[0] - phone_yaw_deg[start_idx]
    calibrated_heading = (phone_yaw_deg[idx_slice] + yaw_offset) % 360.0

    # Extended road for map matching (up to 250 samples / 25s beyond tunnel exit)
    ext_end = min(end_idx + 250, len(gt["gt_x"]) - 1)
    road_x_ext = gt["gt_x"][start_idx:ext_end]
    road_y_ext = gt["gt_y"][start_idx:ext_end]
    true_exit_x = gt_x[-1]
    true_exit_y = gt_y[-1]

    models_trajectories = {}

    # 1. Baseline INS
    bl_s = inertial["speed"][idx_slice]
    bl_x, bl_y = dead_reckon_2d(bl_s, calibrated_heading, dt_slice, start_x, start_y)
    models_trajectories["Baseline INS"] = (bl_x, bl_y, bl_s, False)

    # 2. IDNN v1
    if v1_speed is not None:
        v1_s = v1_speed[idx_slice]
        v1_x, v1_y = dead_reckon_2d(v1_s, calibrated_heading, dt_slice, start_x, start_y)
        models_trajectories["IDNN v1"] = (v1_x, v1_y, v1_s, False)

    # 3. IDNN v2
    if v2_speed is not None:
        v2_s = v2_speed[idx_slice]
        v2_x, v2_y = dead_reckon_2d(v2_s, calibrated_heading, dt_slice, start_x, start_y)
        models_trajectories["IDNN v2"] = (v2_x, v2_y, v2_s, False)

    # 4. IDNN v3 + Map Matching
    if v3_speed is not None:
        v3_s = v3_speed[idx_slice]
        v3_d = np.cumsum(v3_s * dt_slice)
        v3_mx, v3_my = map_match_to_road(v3_d, road_x_ext, road_y_ext)
        models_trajectories["IDNN v3 + MapMatch"] = (v3_mx, v3_my, v3_s, True)

    # 5. IDNN v4 + Map Matching
    if v4_speed is not None:
        v4_s = v4_speed[idx_slice]
        v4_d = np.cumsum(v4_s * dt_slice)
        v4_mx, v4_my = map_match_to_road(v4_d, road_x_ext, road_y_ext)
        models_trajectories["IDNN v4 + MapMatch"] = (v4_mx, v4_my, v4_s, True)

    # 6. IDNN v5 + Map Matching (Pre-Blackout Calibrated with Braking Guard)
    if v5_speed is not None:
        from src.crossdrive_pipeline_v5 import OnlineGNSSCalibrator
        v5_s = v5_speed[idx_slice]
        calibrator = OnlineGNSSCalibrator(history_len=50)
        bias = calibrator.calibrate_at_blackout_entry(v5_speed, gt["velocity_ms"], start_idx)
        ac_slice = ac_vibration[idx_slice] if ac_vibration is not None else None
        v5_s_cal = calibrator.apply_calibration(v5_s, bias, ac_vibration=ac_slice)

        v5_d = np.cumsum(v5_s_cal * dt_slice)
        v5_mx, v5_my = map_match_to_road(v5_d, road_x_ext, road_y_ext)
        models_trajectories["IDNN v5 + MapMatch (Calib)"] = (v5_mx, v5_my, v5_s_cal, True)

    # Compute Error Metrics
    results_models = {}
    for m_name, (mx, my, ms, is_map) in models_trajectories.items():
        pos_err_t = np.sqrt((mx - gt_x)**2 + (my - gt_y)**2)
        final_2d_err = np.sqrt((mx[-1] - true_exit_x)**2 + (my[-1] - true_exit_y)**2)
        drift_2d_pct = (final_2d_err / true_distance) * 100.0 if true_distance > 0 else 0.0

        pred_dist = np.sum(ms * dt_slice)
        dist_1d_err = abs(pred_dist - true_distance)
        drift_1d_pct = (dist_1d_err / true_distance) * 100.0 if true_distance > 0 else 0.0

        results_models[m_name] = {
            "x": mx, "y": my,
            "speed": ms,
            "pos_err_t": pos_err_t,
            "final_2d_err": final_2d_err,
            "drift_2d_pct": drift_2d_pct,
            "dist_1d_err": dist_1d_err,
            "drift_1d_pct": drift_1d_pct,
            "is_map": is_map,
        }

    return {
        "prefix": prefix,
        "name": name,
        "duration": actual_duration,
        "distance": true_distance,
        "t_slice": t_slice,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "models": results_models,
    }


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_clean_scenario_map(res, filename):
    fig, ax = plt.subplots(figsize=(11, 8))

    gt_x = res["gt_x"]
    gt_y = res["gt_y"]

    ax.plot(gt_x, gt_y, "k-", lw=3.5, label="Ground Truth (Real Road)", zorder=10)
    ax.scatter([gt_x[0]], [gt_y[0]], c="green", s=140, marker="o", label="Tunnel Entrance (GPS Cut)", zorder=15)
    ax.scatter([gt_x[-1]], [gt_y[-1]], c="red", s=160, marker="X", label="True Tunnel Exit", zorder=15)

    colors = {
        "Baseline INS": "#e66101",
        "IDNN v1": "#d7191c",
        "IDNN v2": "#fdae61",
        "IDNN v3 + MapMatch": "#2b83ba",
        "IDNN v4 + MapMatch": "#0571b0",
        "IDNN v5 + MapMatch (Calib)": "#008837",
    }

    styles = {
        "Baseline INS": ":",
        "IDNN v1": "-.",
        "IDNN v2": "-.",
        "IDNN v3 + MapMatch": "--",
        "IDNN v4 + MapMatch": "-",
        "IDNN v5 + MapMatch (Calib)": "-",
    }

    for m_name, m_data in res["models"].items():
        c = colors.get(m_name, "tab:purple")
        st = styles.get(m_name, "-")

        if "Baseline" in m_name:
            # Clip exploding baseline so real road geometry remains prominent
            rx_min, rx_max = np.min(gt_x), np.max(gt_x)
            ry_min, ry_max = np.min(gt_y), np.max(gt_y)
            pad_x = max(abs(rx_max - rx_min) * 0.35, 70)
            pad_y = max(abs(ry_max - ry_min) * 0.35, 70)
            bx = np.clip(m_data["x"], rx_min - pad_x, rx_max + pad_x)
            by = np.clip(m_data["y"], ry_min - pad_y, ry_max + pad_y)
            ax.plot(bx, by, st, color=c, lw=1.2, alpha=0.5, label=f"{m_name} ({m_data['final_2d_err']:.0f}m)")
        else:
            lw = 2.4 if "v5" in m_name else (2.0 if "v4" in m_name or "v3" in m_name else 1.2)
            ax.plot(m_data["x"], m_data["y"], st, color=c, lw=lw,
                    label=f"{m_name} ({m_data['final_2d_err']:.1f}m / {m_data['drift_2d_pct']:.1f}%)")

    ax.set_title(f"2D Blackout Simulation: {res['name']} ({res['duration']:.0f}s, {res['distance']:.0f}m)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Local East (Meters)")
    ax.set_ylabel("Local North (Meters)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_time(scenarios, filename):
    n_sc = len(scenarios)
    cols = 3
    rows = (n_sc + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    colors = {
        "IDNN v1": "#d7191c",
        "IDNN v2": "#fdae61",
        "IDNN v3 + MapMatch": "#2b83ba",
        "IDNN v4 + MapMatch": "#0571b0",
        "IDNN v5 + MapMatch (Calib)": "#008837",
    }

    for idx, (sc_name, res) in enumerate(scenarios.items()):
        ax = axes[idx]
        t = res["t_slice"]
        for m_name, m_data in res["models"].items():
            if "Baseline" not in m_name:
                c = colors.get(m_name, "tab:purple")
                lw = 2.2 if "v5" in m_name else 1.5
                ax.plot(t, m_data["pos_err_t"], lw=lw, color=c, label=m_name)

        ax.set_title(res["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time inside Outage (s)")
        ax.set_ylabel("Position Error (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide extra subplots if any
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Position Error Progression vs Outage Time Across Scenarios", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_along_track_summary(scenarios, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

    sc_names = [res["name"].split(" (")[0] for res in scenarios.values()]
    models_to_plot = ["IDNN v1", "IDNN v2", "IDNN v3 + MapMatch", "IDNN v4 + MapMatch", "IDNN v5 + MapMatch (Calib)"]

    colors = {
        "IDNN v1": "#d7191c",
        "IDNN v2": "#fdae61",
        "IDNN v3 + MapMatch": "#2b83ba",
        "IDNN v4 + MapMatch": "#0571b0",
        "IDNN v5 + MapMatch (Calib)": "#008837",
    }

    x = np.arange(len(sc_names))
    width = 0.15

    # Panel 1: Pure Along-Track Speed Integration Drift % (1D)
    for i, m_name in enumerate(models_to_plot):
        drift_1d = [scenarios[k]["models"][m_name]["drift_1d_pct"] for k in scenarios.keys()]
        offset = (i - len(models_to_plot)/2 + 0.5) * width
        rects = ax1.bar(x + offset, drift_1d, width, label=m_name, color=colors[m_name],
                        edgecolor="black" if "v5" in m_name else "none", lw=1.2 if "v5" in m_name else 0)
        for rect, val in zip(rects, drift_1d):
            h = rect.get_height()
            if h > 0 and h < 95:
                ax1.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                             xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                             fontsize=7.5, fontweight="bold" if val <= 10.0 else "normal",
                             color="#006d2c" if val <= 10.0 else "black")

    ax1.axhline(10.0, color="red", ls="--", lw=1.5, label="SIH Target (< 10%)")
    ax1.set_title("Along-Track Speed Drift % (1D Velocity Integration)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Drift Percentage (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_names, fontsize=9, rotation=10)
    ax1.set_ylim(0, 85)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)

    # Panel 2: 2D Position Exit Drift % (With Map Matching)
    for i, m_name in enumerate(models_to_plot):
        drift_2d = [scenarios[k]["models"][m_name]["drift_2d_pct"] for k in scenarios.keys()]
        offset = (i - len(models_to_plot)/2 + 0.5) * width
        rects = ax2.bar(x + offset, drift_2d, width, label=m_name, color=colors[m_name],
                        edgecolor="black" if "v5" in m_name else "none", lw=1.2 if "v5" in m_name else 0)
        for rect, val in zip(rects, drift_2d):
            h = rect.get_height()
            if h > 0 and h < 95:
                ax2.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                             xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                             fontsize=7.5, fontweight="bold" if val <= 10.0 else "normal",
                             color="#006d2c" if val <= 10.0 else "black")

    ax2.axhline(10.0, color="red", ls="--", lw=1.5, label="SIH Target (< 10%)")
    ax2.set_title("2D Position Exit Drift % (Map-Matched to Road Polyline)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drift Percentage (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_names, fontsize=9, rotation=10)
    ax2.set_ylim(0, 95)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper left", fontsize=8.5, ncol=2)

    fig.suptitle("GNSS Outage Drift Benchmark: Along-Track vs Map-Matched 2D Exit Drift",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close(fig)


def save_clean_blackout_table(table_rows, headers, filename):
    fig, ax = plt.subplots(figsize=(19, 13))
    ax.axis("off")
    col_widths = [0.24, 0.28, 0.10, 0.10, 0.10, 0.10, 0.10]
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center",
                     colWidths=col_widths, bbox=[0.02, 0.03, 0.96, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.1, 2.0)

    # Left-align Scenario and Model columns
    for i in range(len(table_rows) + 1):
        table[i, 0].set_text_props(ha="left")
        table[i, 1].set_text_props(ha="left")

    for j in range(len(headers)):
        table[0, j].set_facecolor("#1B365D")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(table_rows, 1):
        m_label = row[1]
        c = "#FFFFFF"
        if "Baseline" in m_label:
            c = "#FFF2CC"
        elif "v1" in m_label:
            c = "#FCE4EC"
        elif "v2" in m_label:
            c = "#FFF3E0"
        elif "v3" in m_label:
            c = "#E1F5FE"
        elif "v4" in m_label:
            c = "#E0F2F1"
        elif "v5" in m_label and "Calib" in m_label:
            c = "#C8E6C9"
        elif "v5" in m_label:
            c = "#E8F5E9"

        for j in range(len(headers)):
            cell = table[i, j]
            cell.set_facecolor(c)
            if "Calib" in m_label and ("2.8%" in row[4] or "3.5%" in row[4]):
                if j in [4, 6]:
                    cell.set_text_props(color="#006d2c", fontweight="bold")

    fig.suptitle("GNSS Blackout Simulation Summary Across All Models", fontsize=16, fontweight="bold", y=0.965)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close(fig)


# ============================================================
# MAIN SIMULATION RUNNER
# ============================================================

def run_blackout_simulation():
    print("=" * 80)
    print("RUNNING CLEAN GNSS BLACKOUT SIMULATION ACROSS ALL MODELS")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Purge old duplicate files
    old_files = glob.glob(f"{OUTPUT_DIR}/*.png")
    for f in old_files:
        try:
            os.remove(f)
        except Exception:
            pass
    print(f"Purged {len(old_files)} old duplicate plots from {OUTPUT_DIR}")

    pairs = find_paired_files()
    primary = pairs[0]

    s_df = pd.read_csv(primary[0], low_memory=False, encoding="latin1")
    v_df = pd.read_csv(primary[1], low_memory=False, encoding="latin1")
    s_cols = detect_smartphone_columns(s_df)
    v_cols = detect_vehicle_columns(v_df)

    v_time = numeric_series(v_df, v_cols["time"]).to_numpy()
    v_time = v_time - v_time[0]
    n_full = len(v_time)
    gt = compute_vehicle_ground_truth(v_df, v_cols, v_time)

    # Baseline Inertial
    ax = numeric_series(s_df, s_cols["ax"]).iloc[:n_full].to_numpy()
    ay = numeric_series(s_df, s_cols["ay"]).iloc[:n_full].to_numpy()
    az = numeric_series(s_df, s_cols["az"]).iloc[:n_full].to_numpy()
    gx_gr = numeric_series(s_df, s_cols.get("grav_x")).iloc[:n_full].to_numpy()
    gy_gr = numeric_series(s_df, s_cols.get("grav_y")).iloc[:n_full].to_numpy()
    gz_gr = numeric_series(s_df, s_cols.get("grav_z")).iloc[:n_full].to_numpy()
    inertial = compute_baseline_inertial(ax, ay, az, gx_gr, gy_gr, gz_gr, v_time)

    # Phone Compass Heading
    yaw_col = None
    for c in s_df.columns:
        if "ORIENTATION" in c.upper() and "YAW" in c.upper():
            yaw_col = c
            break
    phone_yaw_deg = pd.Series(numeric_series(s_df, yaw_col)).interpolate().ffill().bfill().to_numpy()[:n_full]

    # Process models
    m_data_v3 = process_file_v3(primary[0], primary[1], for_training=False, stride=1)
    v1_speed = load_v1_predictions(m_data_v3["ts_feats"], v_time)
    v2_speed = load_v2_predictions(primary[0], primary[1], v_time)

    # Load v3
    ckpt3 = torch.load(V3_MODEL_PATH, weights_only=False, map_location=DEVICE)
    cfg3 = ckpt3["config"]
    m3 = IDNNv3(cfg3["n_features"], cfg3["delay_taps"], cfg3["n_outputs"], cfg3["hidden_sizes"], cfg3["dropout"]).to(DEVICE)
    m3.load_state_dict(ckpt3["model_state"])
    m3.X_mean = ckpt3["X_mean"]; m3.X_std = ckpt3["X_std"]; m3.y_mean = ckpt3["y_mean"]; m3.y_std = ckpt3["y_std"]
    v3_p = predict_v3(m3, m_data_v3["X"])
    v3_raw = np.zeros(n_full)
    v3_raw[V3_DELAY_TAPS:] = v3_p
    v3_speed = zupt_v3(kf_v3(v3_raw), m_data_v3["ts_feats"][:, :3])

    # Load v4
    V4_MODEL_PATH = "results/crossdrive_v4_model.pth"
    from src.idnn_model_v4 import IDNNv4
    from src.crossdrive_pipeline_v4 import (
        process_file_v4, predict_v4,
        kinematic_plausibility_gate as gate_v4,
        ekf_with_acceleration_feedforward as ekf_v4,
        apply_cruise_lock as cruise_v4,
        apply_enhanced_zupt as zupt_v4,
        DELAY_TAPS as V4_DELAY_TAPS,
    )
    ckpt4 = torch.load(V4_MODEL_PATH, weights_only=False, map_location=DEVICE)
    cfg4 = ckpt4["config"]
    m4 = IDNNv4(cfg4["n_features"], cfg4["delay_taps"], cfg4["n_outputs"], cfg4["hidden_sizes"], cfg4["dropout"]).to(DEVICE)
    m4.load_state_dict(ckpt4["model_state"])
    m4.X_mean = ckpt4["X_mean"]; m4.X_std = ckpt4["X_std"]; m4.y_mean = ckpt4["y_mean"]; m4.y_std = ckpt4["y_std"]
    m_data_v4 = process_file_v4(primary[0], primary[1], for_training=False, stride=1)
    v4_p = predict_v4(m4, m_data_v4["X"])
    v4_raw = np.zeros(n_full)
    v4_raw[V4_DELAY_TAPS:] = v4_p
    v4_speed = zupt_v4(cruise_v4(ekf_v4(gate_v4(v4_raw, m_data_v4["lin_ax"]), m_data_v4["lin_ax"]), m_data_v4["lin_ax"], m_data_v4["gz"]), m_data_v4["ts_feats"])

    # Load v5
    V5_MODEL_PATH = "results/crossdrive_v5_model.pth"
    v5_speed = None
    if os.path.exists(V5_MODEL_PATH):
        from src.idnn_model_v5 import IDNNv5
        from src.crossdrive_pipeline_v5 import (
            process_file_v5, predict_v5,
            kinematic_plausibility_gate as gate_v5,
            ekf_with_acceleration_feedforward as ekf_v5,
            apply_cruise_lock as cruise_v5,
            apply_enhanced_zupt as zupt_v5,
            DELAY_TAPS as V5_DELAY_TAPS,
        )
        ckpt5 = torch.load(V5_MODEL_PATH, weights_only=False, map_location=DEVICE)
        cfg5 = ckpt5["config"]
        m5 = IDNNv5(cfg5["n_features"], cfg5["delay_taps"], cfg5["n_outputs"], cfg5["hidden_sizes"], cfg5["dropout"]).to(DEVICE)
        m5.load_state_dict(ckpt5["model_state"])
        m5.X_mean = ckpt5["X_mean"]; m5.X_std = ckpt5["X_std"]; m5.y_mean = ckpt5["y_mean"]; m5.y_std = ckpt5["y_std"]
        m_data_v5 = process_file_v5(primary[0], primary[1], for_training=False, stride=1)
        v5_p = predict_v5(m5, m_data_v5["X"])
        v5_raw = np.zeros(n_full)
        v5_raw[V5_DELAY_TAPS:] = v5_p
        v5_speed = zupt_v5(cruise_v5(ekf_v5(gate_v5(v5_raw, m_data_v5["lin_ax"]), m_data_v5["lin_ax"]), m_data_v5["lin_ax"], m_data_v5["gz"]), m_data_v5["ts_feats"])
        lin_norm = np.sqrt(m_data_v5["lin_ax"]**2 + m_data_v5["ts_feats"][:, 1]**2 + m_data_v5["ts_feats"][:, 2]**2)
        v5_ac_var = pd.Series(lin_norm).rolling(10, min_periods=1).var().fillna(0.0).to_numpy()
    else:
        v5_ac_var = None

    # 5 Distinct, Non-Overlapping Blackout Scenarios
    scenarios_config = [
        ("01_straight_highway_tunnel_30s", "Straight Highway Tunnel", 92100, 30.0),
        ("02_short_underpass_30s", "Short Underpass", 15000, 30.0),
        ("03_medium_tunnel_60s", "Medium Tunnel", 35000, 60.0),
        ("04_long_mountain_tunnel_120s", "Long Mountain Tunnel", 50000, 120.0),
        ("05_complex_city_canyon_90s", "Complex City Canyon", 75000, 90.0),
    ]

    scenarios_results = {}
    table_rows = []
    headers = ["Scenario", "Model", "Distance", "Along-Track Err", "Along-Track Drift %", "2D Exit Err", "2D Drift %"]

    for prefix, name, start_idx, dur in scenarios_config:
        res = run_blackout_scenario(
            prefix, name, start_idx, dur,
            v_time, gt, inertial,
            v1_speed, v2_speed, v3_speed, v4_speed, v5_speed,
            phone_yaw_deg,
            ac_vibration=v5_ac_var,
        )
        scenarios_results[prefix] = res
        plot_clean_scenario_map(res, f"{prefix}_map.png")

        dist_m = res["distance"]
        for m_name, m_stats in res["models"].items():
            table_rows.append([
                f"{name} ({dur:.0f}s)",
                m_name,
                f"{dist_m:.0f} m",
                f"{m_stats['dist_1d_err']:.1f} m",
                f"{m_stats['drift_1d_pct']:.1f}%",
                f"{m_stats['final_2d_err']:.1f} m",
                f"{m_stats['drift_2d_pct']:.1f}%",
            ])

    plot_error_vs_time(scenarios_results, "06_error_vs_time.png")
    plot_along_track_summary(scenarios_results, "07_along_track_vs_mapmatch_comparison.png")
    save_clean_blackout_table(table_rows, headers, "08_comprehensive_blackout_table.png")

    print(f"\nAll 8 clean, non-repeating plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_blackout_simulation()