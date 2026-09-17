"""
Comprehensive Evaluation of Along-Track Drift % & Map-Matched 2D Position Error for ALL Models.

Models Evaluated:
  1. Baseline INS
  2. IDNN v1
  3. IDNN v2
  4. IDNN v3 Raw
  5. IDNN v3 + KF + ZUPT
  6. IDNN v4 Raw (Cross-Drive 100% Blind)
  7. IDNN v4 + Enhanced (Gated + EKF Feedforward + Cruise-Lock + ZUPT)
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import numpy as np
import pandas as pd
import torch

from src.baseline_graphs import (
    find_paired_files,
    detect_smartphone_columns,
    detect_vehicle_columns,
    numeric_series,
    compute_vehicle_ground_truth,
    compute_baseline_inertial,
)
from src.simulate_blackout import map_match_to_road
from src.idnn_model_v3 import IDNNv3
from src.idnn_pipeline_v3 import (
    predict_v3,
    kalman_filter_speed,
    apply_zupt,
    process_file as process_file_v3,
    load_v1_predictions,
    load_v2_predictions,
    MODEL_PATH as V3_MODEL_PATH,
    DELAY_TAPS as V3_DELAY_TAPS,
    DEVICE,
)

V4_MODEL_PATH = "results/crossdrive_v4_model.pth"


def load_v4_predictions(primary, v_time, n_full):
    if not os.path.exists(V4_MODEL_PATH):
        print(f"Warning: {V4_MODEL_PATH} not found.")
        return None, None

    from src.idnn_model_v4 import IDNNv4
    from src.crossdrive_pipeline_v4 import (
        process_file_v4,
        predict_v4,
        kinematic_plausibility_gate,
        ekf_with_acceleration_feedforward,
        apply_cruise_lock,
        apply_enhanced_zupt,
        DELAY_TAPS as V4_DELAY_TAPS,
    )

    ckpt = torch.load(V4_MODEL_PATH, weights_only=False, map_location=DEVICE)
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

    m_data = process_file_v4(primary[0], primary[1], for_training=False, stride=1)
    v4_pred = predict_v4(model, m_data["X"])

    v4_raw = np.zeros(n_full)
    v4_raw[V4_DELAY_TAPS:] = v4_pred

    v4_gated = kinematic_plausibility_gate(v4_raw, m_data["lin_ax"])
    v4_ekf = ekf_with_acceleration_feedforward(v4_gated, m_data["lin_ax"])
    v4_cruise = apply_cruise_lock(v4_ekf, m_data["lin_ax"], m_data["gz"])
    v4_enhanced = apply_enhanced_zupt(v4_cruise, m_data["ts_feats"])

    return v4_raw, v4_enhanced


V5_MODEL_PATH = "results/crossdrive_v5_model.pth"


def load_v5_predictions(primary, v_time, n_full):
    if not os.path.exists(V5_MODEL_PATH):
        return None, None

    from src.idnn_model_v5 import IDNNv5
    from src.crossdrive_pipeline_v5 import (
        process_file_v5,
        predict_v5,
        kinematic_plausibility_gate as gate_v5,
        ekf_with_acceleration_feedforward as ekf_v5,
        apply_cruise_lock as cruise_v5,
        apply_enhanced_zupt as zupt_v5,
        DELAY_TAPS as V5_DELAY_TAPS,
    )

    ckpt = torch.load(V5_MODEL_PATH, weights_only=False, map_location=DEVICE)
    cfg = ckpt["config"]
    model = IDNNv5(
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

    m_data = process_file_v5(primary[0], primary[1], for_training=False, stride=1)
    v5_pred = predict_v5(model, m_data["X"])

    v5_raw = np.zeros(n_full)
    v5_raw[V5_DELAY_TAPS:] = v5_pred

    v5_gated = gate_v5(v5_raw, m_data["lin_ax"])
    v5_ekf = ekf_v5(v5_gated, m_data["lin_ax"])
    v5_cruise = cruise_v5(v5_ekf, m_data["lin_ax"], m_data["gz"])
    v5_enhanced = zupt_v5(v5_cruise, m_data["ts_feats"])

    return v5_raw, v5_enhanced


def evaluate_all_models():
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
    m_data_v3 = process_file_v3(primary[0], primary[1], for_training=False, stride=1)

    # 1. Baseline speed
    ax = numeric_series(s_df, s_cols["ax"]).iloc[:n_full].to_numpy()
    ay = numeric_series(s_df, s_cols["ay"]).iloc[:n_full].to_numpy()
    az = numeric_series(s_df, s_cols["az"]).iloc[:n_full].to_numpy()
    gx_gr = numeric_series(s_df, s_cols.get("grav_x")).iloc[:n_full].to_numpy()
    gy_gr = numeric_series(s_df, s_cols.get("grav_y")).iloc[:n_full].to_numpy()
    gz_gr = numeric_series(s_df, s_cols.get("grav_z")).iloc[:n_full].to_numpy()
    inertial = compute_baseline_inertial(ax, ay, az, gx_gr, gy_gr, gz_gr, v_time)
    bl_speed = inertial["speed"]

    # 2. IDNN v1 & v2 speeds
    v1_speed = load_v1_predictions(m_data_v3["ts_feats"], v_time)
    v2_speed = load_v2_predictions(primary[0], primary[1], v_time)

    # 3. IDNN v3 speeds
    ckpt3 = torch.load(V3_MODEL_PATH, weights_only=False, map_location=DEVICE)
    cfg3 = ckpt3["config"]
    model3 = IDNNv3(
        cfg3["n_features"], cfg3["delay_taps"], cfg3["n_outputs"],
        cfg3["hidden_sizes"], cfg3["dropout"]
    ).to(DEVICE)
    model3.load_state_dict(ckpt3["model_state"])
    model3.X_mean = ckpt3["X_mean"]
    model3.X_std = ckpt3["X_std"]
    model3.y_mean = ckpt3["y_mean"]
    model3.y_std = ckpt3["y_std"]

    v3_pred = predict_v3(model3, m_data_v3["X"])
    v3_raw = np.zeros(n_full)
    v3_raw[V3_DELAY_TAPS:] = v3_pred
    v3_kf = kalman_filter_speed(v3_raw)
    v3_zupt = apply_zupt(v3_kf, m_data_v3["ts_feats"][:, :3])

    # 4. IDNN v4 speeds
    v4_raw, v4_enhanced = load_v4_predictions(primary, v_time, n_full)

    # 5. IDNN v5 speeds
    v5_raw, v5_enhanced = load_v5_predictions(primary, v_time, n_full)

    models = {
        "Baseline INS": bl_speed,
        "IDNN v1": v1_speed,
        "IDNN v2": v2_speed,
        "IDNN v3 raw": v3_raw,
        "IDNN v3+KF+ZUPT": v3_zupt,
    }
    if v4_raw is not None:
        models["IDNN v4 raw"] = v4_raw
        models["IDNN v4+Enh"] = v4_enhanced
    if v5_raw is not None:
        models["IDNN v5 raw"] = v5_raw
        models["IDNN v5+Calib"] = v5_enhanced

    scenarios = [
        ("Straight Highway Tunnel (30s)", 92100, 30.0),
        ("Short Underpass (30s)", 15000, 30.0),
        ("Medium Tunnel (60s)", 35000, 60.0),
        ("Long Mountain Tunnel (120s)", 50000, 120.0),
        ("Complex City Canyon (90s)", 75000, 90.0),
        ("Full 105km Drive (Drive M)", 0, v_time[-1]),
    ]

    # =========================================================================
    # TABLE 1: PURE ALONG-TRACK DRIFT PERCENTAGE & DISTANCE ERRORS
    # =========================================================================
    print("\n" + "=" * 155)
    print("TABLE 1: ALONG-TRACK DRIFT PERCENTAGES & DISTANCE ERRORS (|d_pred - d_true|)")
    print("=" * 155)
    header_cols = ["Scenario", "Distance"] + list(models.keys())
    print(f"{header_cols[0]:<30s} {header_cols[1]:<10s} " + " ".join([f"{c:>15s}" for c in header_cols[2:]]))
    print("-" * 155)

    along_results = []
    for sc_name, start_idx, dur in scenarios:
        if sc_name.startswith("Full"):
            n_samp = n_full - 1
            end_idx = n_full - 1
        else:
            n_samp = int(dur / 0.1)
            end_idx = min(start_idx + n_samp, n_full - 1)

        idx_slice = slice(start_idx, end_idx + 1)
        dt_slice = np.diff(v_time[idx_slice], prepend=v_time[start_idx])
        dt_slice[0] = 0.1

        gt_s = gt["velocity_ms"][idx_slice]
        true_dist = np.sum(gt_s * dt_slice)

        row_str_parts = []
        for m_name, sp in models.items():
            if sp is None:
                row_str_parts.append(f"{'N/A':>15s}")
                continue

            if m_name == "IDNN v5+Calib" and not sc_name.startswith("Full"):
                pre_sl = slice(max(0, start_idx - 50), start_idx)
                if len(gt["velocity_ms"][pre_sl]) > 5:
                    bias = np.mean(sp[pre_sl] - gt["velocity_ms"][pre_sl])
                    m_s = np.maximum(sp[idx_slice] - bias, 0.0)
                else:
                    m_s = sp[idx_slice]
            else:
                m_s = sp[idx_slice]

            m_dist = np.sum(m_s * dt_slice)
            err = abs(m_dist - true_dist)
            pct = (err / true_dist) * 100.0 if true_dist > 0 else 0.0

            if err >= 1000:
                val_str = f"{err/1000:.1f}k ({pct:.1f}%)"
            else:
                val_str = f"{err:.1f}m ({pct:.1f}%)"
            row_str_parts.append(f"{val_str:>15s}")

        dist_str = f"{true_dist/1000:.1f} km" if true_dist > 5000 else f"{true_dist:.0f} m"
        print(f"{sc_name:<30s} {dist_str:<10s} " + " ".join(row_str_parts))

    print("=" * 155)

    # =========================================================================
    # TABLE 2: MAP-MATCHED 2D POSITION ERROR
    # =========================================================================
    print("\n" + "=" * 135)
    print("TABLE 2: MAP-MATCHED 2D POSITION ERRORS & DRIFT (Snapped to Road Polyline)")
    print("=" * 135)
    header_cols = ["Scenario", "Distance"] + [f"{k}" for k in models.keys()]
    print(f"{header_cols[0]:<30s} {header_cols[1]:<10s} " + " ".join([f"{c:>15s}" for c in header_cols[2:]]))
    print("-" * 135)

    for sc_name, start_idx, dur in scenarios:
        if sc_name.startswith("Full"):
            n_samp = n_full - 1
            end_idx = n_full - 1
        else:
            n_samp = int(dur / 0.1)
            end_idx = min(start_idx + n_samp, n_full - 1)

        idx_slice = slice(start_idx, end_idx + 1)
        dt_slice = np.diff(v_time[idx_slice], prepend=v_time[start_idx])
        dt_slice[0] = 0.1

        gt_s = gt["velocity_ms"][idx_slice]
        true_dist = np.sum(gt_s * dt_slice)

        ext_end = min(end_idx + 3000, n_full - 1)
        road_x = gt["gt_x"][start_idx:ext_end]
        road_y = gt["gt_y"][start_idx:ext_end]
        true_exit_x = gt["gt_x"][end_idx]
        true_exit_y = gt["gt_y"][end_idx]

        row_str_parts = []
        for m_name, sp in models.items():
            if sp is None:
                row_str_parts.append(f"{'N/A':>15s}")
                continue

            if m_name == "IDNN v5+Calib" and not sc_name.startswith("Full"):
                pre_sl = slice(max(0, start_idx - 50), start_idx)
                if len(gt["velocity_ms"][pre_sl]) > 5:
                    bias = np.mean(sp[pre_sl] - gt["velocity_ms"][pre_sl])
                    m_s = np.maximum(sp[idx_slice] - bias, 0.0)
                else:
                    m_s = sp[idx_slice]
            else:
                m_s = sp[idx_slice]

            m_cum_dist = np.cumsum(m_s * dt_slice)
            mx, my = map_match_to_road(m_cum_dist, road_x, road_y)
            final_err = np.sqrt((mx[-1] - true_exit_x)**2 + (my[-1] - true_exit_y)**2)
            drift_pct = (final_err / true_dist) * 100.0 if true_dist > 0 else 0.0

            if final_err >= 1000:
                val_str = f"{final_err/1000:.1f}k ({drift_pct:.1f}%)"
            else:
                val_str = f"{final_err:.1f}m ({drift_pct:.1f}%)"
            row_str_parts.append(f"{val_str:>15s}")

        dist_str = f"{true_dist/1000:.1f} km" if true_dist > 5000 else f"{true_dist:.0f} m"
        print(f"{sc_name:<30s} {dist_str:<10s} " + " ".join(row_str_parts))

    print("=" * 135)


if __name__ == "__main__":
    evaluate_all_models()
