"""
Cross-Drive Blind Test Pipeline for IDNN v3.

Rigorous evaluation protocol:
  - Training Data  : 100% external driving sessions (S1, S2, S3a, S3b, S3c, S4, Vfa01)
  - Validation Data: External driving session (Vfa02)
  - Test Data      : 100% of Drive M (S-M.csv / V-M.csv) -- NEVER SEEN DURING TRAINING!

This provides absolute proof against overfitting:
  - Different vehicle/mount
  - Different driver
  - Different route and road conditions

Outputs:
  - results/crossdrive_v3_model.pth
  - results/plots/crossdrive_test/ (all plots + comprehensive table)
"""

import os
import sys
import time
import traceback

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.idnn_model_v3 import IDNNv3
from src.baseline_graphs import (
    find_paired_files,
    detect_smartphone_columns,
    detect_vehicle_columns,
    numeric_series,
    compute_baseline_inertial,
    compute_vehicle_ground_truth,
    calculate_scalar_drift,
    G,
)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "results/plots/crossdrive_test"
MODEL_PATH = "results/crossdrive_v3_model.pth"
V1_PATH = "results/idnn_model.pth"
V2_PATH = "results/idnn_v2_model.pth"

N_FEATURES = 12       # lin_ax, lin_ay, lin_az, gx, gy, gz, jerk_ax, jerk_ay, jerk_az, jerk_gx, jerk_gy, jerk_gz
DELAY_TAPS = 20       # 2 seconds at 10 Hz
N_OUTPUTS = 1
HIDDEN_SIZES = (256, 128, 64)
DROPOUT = 0.2
SAMPLE_RATE = 10.0

EPOCHS = 80
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4

TRAIN_STRIDE = 2      # stride for external training files

# Kalman Filter
KF_PROCESS_NOISE = 0.5
KF_MEASUREMENT_NOISE = 4.0

# ZUPT
ZUPT_ENERGY_THRESHOLD = 0.3
ZUPT_WINDOW = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _interp(arr, n):
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    if isinstance(arr, pd.Series):
        arr = arr.interpolate().ffill().bfill().to_numpy(dtype=np.float64)
    else:
        arr = pd.Series(arr[:n]).interpolate().ffill().bfill().to_numpy(dtype=np.float64)
    return arr[:n]


def extract_features_from_file(s_df, s_cols, n):
    ax = _interp(numeric_series(s_df, s_cols["ax"]), n)
    ay = _interp(numeric_series(s_df, s_cols["ay"]), n)
    az = _interp(numeric_series(s_df, s_cols["az"]), n)

    grav_x = _interp(numeric_series(s_df, s_cols.get("grav_x")), n)
    grav_y = _interp(numeric_series(s_df, s_cols.get("grav_y")), n)
    grav_z = _interp(numeric_series(s_df, s_cols.get("grav_z")), n)

    mag = np.sqrt(ax**2 + ay**2 + az**2)
    if 0.5 <= np.nanmedian(mag) <= 2.0:
        ax *= G; ay *= G; az *= G
        grav_x *= G; grav_y *= G; grav_z *= G

    lin_ax = ax - grav_x
    lin_ay = ay - grav_y
    lin_az = az - grav_z

    gx = _interp(numeric_series(s_df, s_cols.get("gx")), n)
    gy = _interp(numeric_series(s_df, s_cols.get("gy")), n)
    gz = _interp(numeric_series(s_df, s_cols.get("gz")), n)

    dt = 1.0 / SAMPLE_RATE
    jerk_ax = np.gradient(lin_ax, dt)
    jerk_ay = np.gradient(lin_ay, dt)
    jerk_az = np.gradient(lin_az, dt)
    jerk_gx = np.gradient(gx, dt)
    jerk_gy = np.gradient(gy, dt)
    jerk_gz = np.gradient(gz, dt)

    features = np.column_stack([
        lin_ax, lin_ay, lin_az,
        gx, gy, gz,
        jerk_ax, jerk_ay, jerk_az,
        jerk_gx, jerk_gy, jerk_gz,
    ]).astype(np.float32)

    return features, (ax, ay, az, grav_x, grav_y, grav_z)


def create_delay_windows(features, target, delay_taps, stride=1):
    n_samples, n_feat = features.shape
    n_valid = n_samples - delay_taps

    indices = np.arange(0, n_valid, stride)
    M = len(indices)

    X = np.zeros((M, (delay_taps + 1) * n_feat), dtype=np.float32)
    y_out = np.zeros(M, dtype=np.float32)

    for j, i in enumerate(indices):
        t = i + delay_taps
        window = features[t - delay_taps: t + 1][::-1]
        X[j] = window.flatten()
        y_out[j] = target[t]

    return X, y_out


def process_file(s_path, v_path, for_training=False, stride=1):
    s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1")
    v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1")

    s_cols = detect_smartphone_columns(s_df)
    v_cols = detect_vehicle_columns(v_df)

    v_time = numeric_series(v_df, v_cols["time"]).to_numpy()
    v_time = v_time - v_time[0]
    v_time = pd.Series(v_time).interpolate().ffill().bfill().to_numpy()
    for i in range(1, len(v_time)):
        if v_time[i] <= v_time[i - 1]:
            v_time[i] = v_time[i - 1] + 0.1

    n = min(len(s_df), len(v_df), len(v_time))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)
    time_axis = v_time[:n]

    gt = compute_vehicle_ground_truth(v_df, v_cols, time_axis)
    target_speed = gt["velocity_ms"]

    ts_feats, raw_accel = extract_features_from_file(s_df, s_cols, n)
    X, y = create_delay_windows(ts_feats, target_speed, DELAY_TAPS, stride=stride)

    result = {
        "X": X,
        "y": y,
        "n_samples": len(X),
    }

    if not for_training:
        ax, ay, az, grav_x, grav_y, grav_z = raw_accel
        inertial = compute_baseline_inertial(
            ax.copy(), ay.copy(), az.copy(),
            grav_x.copy(), grav_y.copy(), grav_z.copy(),
            time_axis,
        )
        result["time_axis"] = time_axis
        result["gt"] = gt
        result["inertial"] = inertial
        result["ts_feats"] = ts_feats
        result["s_path"] = s_path
        result["v_path"] = v_path

    return result


# ============================================================
# TRAINING
# ============================================================

def train_crossdrive(X_train, y_train, X_val, y_val):
    model = IDNNv3(
        n_features=N_FEATURES,
        delay_taps=DELAY_TAPS,
        n_outputs=N_OUTPUTS,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n  IDNN v3 (Cross-Drive) Architecture:")
    print(f"    Input: {N_FEATURES} features x {DELAY_TAPS+1} taps = {N_FEATURES*(DELAY_TAPS+1)}")
    print(f"    Hidden: {HIDDEN_SIZES} + BatchNorm")
    print(f"    Parameters: {n_params:,}")
    print(f"    Device: {DEVICE} ({dev_name})")

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_tr = ((X_train - X_mean) / X_std).astype(np.float32)
    X_va = ((X_val - X_mean) / X_std).astype(np.float32)
    y_tr = ((y_train - y_mean) / y_std).astype(np.float32)
    y_va = ((y_val - y_mean) / y_std).astype(np.float32)

    use_cuda = torch.cuda.is_available()
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=use_cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6,
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    print(f"\n  Training {EPOCHS} epochs on external drives (train={len(X_train):,}, val={len(X_val):,})")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_total, t_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_total += loss.item() * len(xb)
            t_n += len(xb)
        train_loss = t_total / t_n

        model.eval()
        v_total, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                v_total += loss.item() * len(xb)
                v_n += len(xb)
        val_loss = v_total / v_n

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}  best={best_val:.6f}  lr={lr_now:.1e}  [{time.time()-t0:.1f}s]")

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  Cross-Drive Training complete in {time.time()-t0:.1f}s. Best val loss: {best_val:.6f}")

    model.X_mean = X_mean
    model.X_std = X_std
    model.y_mean = y_mean
    model.y_std = y_std

    return model, history


def predict(model, X, batch=8192):
    model.eval()
    X_norm = ((X - model.X_mean) / model.X_std).astype(np.float32)

    preds = []
    for i in range(0, len(X_norm), batch):
        xt = torch.from_numpy(X_norm[i:i+batch]).to(DEVICE)
        with torch.no_grad():
            p = model(xt).cpu().numpy().flatten()
        preds.append(p)

    pred = np.concatenate(preds)
    pred = pred * model.y_std + model.y_mean
    return np.maximum(pred, 0.0)


def kalman_filter_speed(raw_speed, Q=KF_PROCESS_NOISE, R=KF_MEASUREMENT_NOISE):
    n = len(raw_speed)
    out = np.zeros(n, dtype=np.float64)
    x = float(raw_speed[0])
    P = 1.0

    for i in range(n):
        x_pred = x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (raw_speed[i] - x_pred)
        P = (1.0 - K) * P_pred
        out[i] = max(x, 0.0)
    return out


def apply_zupt(speed, accel_features, threshold=ZUPT_ENERGY_THRESHOLD, window=ZUPT_WINDOW):
    n = len(speed)
    out = speed.copy()
    energy = np.sqrt(np.mean(accel_features**2, axis=1))

    if n >= window:
        kernel = np.ones(window) / window
        energy_smooth = np.convolve(energy, kernel, mode="same")
    else:
        energy_smooth = energy

    stopped = energy_smooth < threshold
    out[stopped] = 0.0
    return out


def speed_to_distance(speed, time_axis):
    dt_arr = np.diff(time_axis, prepend=time_axis[0])
    dt_med = np.median(np.diff(time_axis))
    dt_arr[0] = dt_med
    dist = np.zeros(len(time_axis))
    for i in range(1, len(time_axis)):
        dt_i = dt_arr[i] if (0 < dt_arr[i] < 10 and np.isfinite(dt_arr[i])) else dt_med
        dist[i] = dist[i-1] + speed[i] * dt_i
    return dist


def drift_on_split(speed_pred, speed_gt, dt=0.1):
    gt_d = np.sum(speed_gt * dt)
    if gt_d < 1:
        return float("nan")
    return abs(np.sum(speed_pred * dt) - gt_d) / gt_d * 100


# ============================================================
# LOAD V1 / V2
# ============================================================

def load_v1_predictions(ts_feats, time_axis):
    if not os.path.exists(V1_PATH):
        return None
    try:
        from src.idnn_model import IDNN
        from src.idnn_pipeline import create_delay_windows as v1_windows, predict as v1_predict

        ckpt = torch.load(V1_PATH, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        model = IDNN(cfg["n_features"], cfg["delay_taps"], cfg["n_outputs"],
                      cfg["hidden_sizes"], cfg["dropout"]).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.X_mean = ckpt["X_mean"]; model.X_std = ckpt["X_std"]
        model.y_mean = ckpt["y_mean"]; model.y_std = ckpt["y_std"]

        v1_raw = ts_feats[:, :6]
        X, _ = v1_windows(v1_raw, np.zeros(len(v1_raw)), cfg["delay_taps"])
        speed = v1_predict(model, X)

        full_speed = np.zeros(len(time_axis))
        full_speed[cfg["delay_taps"]:] = speed
        return full_speed
    except Exception as e:
        print(f"  V1 load failed: {e}")
        return None


def load_v2_predictions(s_path, v_path, time_axis):
    if not os.path.exists(V2_PATH):
        return None
    try:
        from src.idnn_model_v2 import IDNNv2
        from src.idnn_pipeline_v2 import prepare_all_data, predict_v2, WINDOW_SIZE as V2_WIN

        s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1")
        v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1")
        s_cols = detect_smartphone_columns(s_df)
        v_cols = detect_vehicle_columns(v_df)

        min_len = min(len(s_df), len(v_df), len(time_axis))
        s_df = s_df.iloc[:min_len].reset_index(drop=True)
        v_df = v_df.iloc[:min_len].reset_index(drop=True)
        t = time_axis[:min_len]

        ckpt = torch.load(V2_PATH, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        model = IDNNv2(cfg["n_features"], cfg["n_outputs"],
                        cfg["hidden_sizes"], cfg["dropout"]).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.X_mean = ckpt["X_mean"]; model.X_std = ckpt["X_std"]
        model.y_mean = ckpt["y_mean"]; model.y_std = ckpt["y_std"]

        feat_v2, _, _, _, _ = prepare_all_data(s_df, v_df, s_cols, v_cols, t)
        speed = predict_v2(model, feat_v2)

        full_speed = np.zeros(len(time_axis))
        full_speed[V2_WIN - 1:] = speed
        return full_speed
    except Exception as e:
        print(f"  V2 load failed: {e}")
        return None


# ============================================================
# PLOTTING
# ============================================================

def generate_crossdrive_plots(
    time_axis, gt, inertial,
    v1_speed, v2_speed,
    v3_speed, v3_kf_speed, v3_zupt_speed,
    v3_dist, v3_kf_dist, v3_zupt_dist,
    history,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    gt_s = gt["velocity_ms"]
    gt_d = gt["gt_distance"]
    bl_s = inertial["speed"]
    bl_d = inertial["cumulative_distance"]

    C = {"gt": "tab:blue", "bl": "tab:orange", "v1": "tab:red",
         "v2": "tab:purple", "v3": "tab:green", "kf": "darkgreen", "zupt": "teal"}

    # 1. Training loss on external drives
    fig, ax = plt.subplots(figsize=(10, 6))
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="Train (External Drives S1..S4, Vfa01)", lw=1.5)
    ax.plot(ep, history["val_loss"], label="Val (External Drive Vfa02)", lw=1.5)
    ax.set(title="Cross-Drive Training Loss (Zero Samples from Drive M)", xlabel="Epoch", ylabel="MSE (normalized)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Speed comparison across 100% blind Drive M
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(time_axis, gt_s*3.6, lw=1.5, alpha=0.8, label="Vehicle Ground Truth", color=C["gt"])
    ax.plot(time_axis, bl_s*3.6, lw=0.6, alpha=0.15, label="Baseline INS", color=C["bl"])
    if v1_speed is not None:
        ax.plot(time_axis, v1_speed*3.6, lw=0.8, alpha=0.35, label="IDNN v1 (Single-Drive)", color=C["v1"])
    if v2_speed is not None:
        ax.plot(time_axis, v2_speed*3.6, lw=0.8, alpha=0.35, label="IDNN v2 (Single-Drive)", color=C["v2"])
    ax.plot(time_axis, v3_zupt_speed*3.6, lw=1.2, alpha=0.9, label="IDNN v3+KF+ZUPT (100% Blind)", color=C["zupt"])
    ax.set(title="Speed on 100% Blind Drive M (Never Seen During Training)", xlabel="Time (s)", ylabel="km/h")
    ax.set_ylim(-5, max(np.max(gt_s*3.6)*1.3, 120))
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_speed_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Zoomed v3+KF+ZUPT vs Ground Truth
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, gt_s*3.6, lw=1.5, label="Vehicle Ground Truth", color=C["gt"])
    ax.plot(time_axis, v3_zupt_speed*3.6, lw=1.2, alpha=0.85, label="IDNN v3+KF+ZUPT (Blind)", color=C["zupt"])
    ax.set(title="Blind Cross-Drive Speed Prediction vs Vehicle Ground Truth", xlabel="Time (s)", ylabel="km/h")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_v3_vs_gt.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Cumulative distance
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, gt_d, lw=2, label="Vehicle Ground Truth", color=C["gt"])
    ax.plot(time_axis, bl_d, lw=1, alpha=0.2, label="Baseline INS", color=C["bl"])
    if v1_speed is not None:
        v1_d = speed_to_distance(v1_speed, time_axis)
        ax.plot(time_axis, v1_d, lw=1, alpha=0.4, label="IDNN v1", color=C["v1"])
    if v2_speed is not None:
        v2_d = speed_to_distance(v2_speed, time_axis)
        ax.plot(time_axis, v2_d, lw=1, alpha=0.4, label="IDNN v2", color=C["v2"])
    ax.plot(time_axis, v3_zupt_dist, lw=1.5, alpha=0.9, label="IDNN v3+KF+ZUPT (Blind)", color=C["zupt"])
    ax.set(title="Cumulative Distance on 100% Blind Drive M", xlabel="Time (s)", ylabel="Meters")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_cumulative_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Distance error
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, np.abs(bl_d - gt_d), lw=0.8, alpha=0.2, label="Baseline", color=C["bl"])
    if v1_speed is not None:
        ax.plot(time_axis, np.abs(v1_d - gt_d), lw=1, alpha=0.4, label="IDNN v1", color=C["v1"])
    if v2_speed is not None:
        ax.plot(time_axis, np.abs(v2_d - gt_d), lw=1, alpha=0.4, label="IDNN v2", color=C["v2"])
    ax.plot(time_axis, np.abs(v3_dist - gt_d), lw=1, alpha=0.5, label="IDNN v3 raw (Blind)", color=C["v3"])
    ax.plot(time_axis, np.abs(v3_zupt_dist - gt_d), lw=1.5, alpha=0.9, label="IDNN v3+KF+ZUPT (Blind)", color=C["zupt"])
    ax.set(title="Distance Error Over Time on Blind Drive M", xlabel="Time (s)", ylabel="Meters")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_distance_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6. Drift % vs distance
    _, bl_dr = calculate_scalar_drift(bl_d, gt_d)
    _, v3_dr = calculate_scalar_drift(v3_dist, gt_d)
    _, zupt_dr = calculate_scalar_drift(v3_zupt_dist, gt_d)

    fig, ax = plt.subplots(figsize=(14, 6))
    valid = np.isfinite(bl_dr)
    if np.any(valid):
        ax.plot(gt_d[valid], bl_dr[valid], lw=0.8, alpha=0.2, label="Baseline", color=C["bl"])
    if v1_speed is not None:
        _, v1_dr = calculate_scalar_drift(v1_d, gt_d)
        v = np.isfinite(v1_dr)
        if np.any(v):
            ax.plot(gt_d[v], v1_dr[v], lw=1, alpha=0.4, label="IDNN v1", color=C["v1"])
    if v2_speed is not None:
        _, v2_dr = calculate_scalar_drift(v2_d, gt_d)
        v = np.isfinite(v2_dr)
        if np.any(v):
            ax.plot(gt_d[v], v2_dr[v], lw=1, alpha=0.4, label="IDNN v2", color=C["v2"])
    v = np.isfinite(v3_dr)
    if np.any(v):
        ax.plot(gt_d[v], v3_dr[v], lw=1, alpha=0.5, label="IDNN v3 raw (Blind)", color=C["v3"])
    v = np.isfinite(zupt_dr)
    if np.any(v):
        ax.plot(gt_d[v], zupt_dr[v], lw=1.5, alpha=0.9, label="IDNN v3+KF+ZUPT (Blind)", color=C["zupt"])
    ax.axhline(10, ls="--", lw=1.5, color="red", label="10% Threshold")
    ax.axhline(6, ls="--", lw=1.0, color="gold", label="6% Target")
    max_dr = 30
    if np.any(np.isfinite(zupt_dr)):
        max_dr = min(50, max(np.nanmax(zupt_dr[np.isfinite(zupt_dr)])*1.5, 15))
    ax.set_ylim(0, max_dr)
    ax.set(title="Blind Drift % vs Distance on Drive M", xlabel="Distance (m)", ylabel="Drift (%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_drift_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 7. Speed error histogram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_i, name, sp, col in [
        (axes[0], "Baseline INS", bl_s, C["bl"]),
        (axes[1], "IDNN v3 raw (Blind)", v3_speed, C["v3"]),
        (axes[2], "IDNN v3+KF+ZUPT (Blind)", v3_zupt_speed, C["zupt"]),
    ]:
        err = (sp - gt_s) * 3.6
        rmse = np.sqrt(np.mean(err**2))
        ax_i.hist(err, bins=100, alpha=0.7, color=col)
        ax_i.set_title(f"{name} (RMSE={rmse:.1f} km/h)")
        ax_i.set_xlabel("Speed Error (km/h)"); ax_i.grid(True, alpha=0.3)
    fig.suptitle("Blind Speed Error Distribution on Drive M", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/07_speed_error_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 8. Summary bars
    models = ["Baseline INS", "IDNN v1", "IDNN v2", "v3 Blind Raw", "v3 Blind+KF+ZUPT"]
    speeds_list = [bl_s, v1_speed, v2_speed, v3_speed, v3_zupt_speed]
    dists_list = [bl_d,
                  v1_d if v1_speed is not None else None,
                  v2_d if v2_speed is not None else None,
                  v3_dist, v3_zupt_dist]
    colors = [C["bl"], C["v1"], C["v2"], C["v3"], C["zupt"]]

    rmses, fdrifts = [], []
    for sp in speeds_list:
        if sp is not None:
            rmses.append(np.sqrt(np.mean((sp - gt_s)**2)) * 3.6)
        else:
            rmses.append(0)
    for d in dists_list:
        if d is not None:
            fdrifts.append(abs(d[-1] - gt_d[-1]) / gt_d[-1] * 100 if gt_d[-1] > 0 else 0)
        else:
            fdrifts.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(models, rmses, color=colors)
    axes[0].set_title("Speed RMSE on 100% Blind Drive M"); axes[0].set_ylabel("km/h")
    for b, v in zip(bars, rmses):
        if v > 0:
            axes[0].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                         f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=15)

    bars = axes[1].bar(models, fdrifts, color=colors)
    axes[1].set_title("Final Cumulative Drift on Blind Drive M"); axes[1].set_ylabel("%")
    axes[1].axhline(10, ls="--", color="red", lw=1.5, label="10% Threshold")
    for b, v in zip(bars, fdrifts):
        if v > 0:
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                         f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Cross-Drive Generalization Performance on Blind Drive M", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_summary_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_drift_table_image(rows, headers):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.0)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#1B365D")
        table[0, j].set_text_props(color="white", fontweight="bold")
    cmap = {"Baseline": "#FFF2CC", "v1": "#FCE4EC", "v2": "#E8D5F0",
            "v3 Blind Raw": "#E8F5E9", "v3 Blind+KF": "#C8E6C9", "ZUPT": "#A5D6A7"}
    for i, row in enumerate(rows, 1):
        c = "#FFFFFF"
        for k, v in cmap.items():
            if k in row[0]:
                c = v; break
        for j in range(len(headers)):
            table[i, j].set_facecolor(c)
    ax.set_title("Cross-Drive Blind Test Drift Table (100% Unseen Drive M)",
                 fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/09_drift_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_crossdrive_pipeline():
    print("=" * 75)
    print("CROSS-DRIVE BLIND TEST PIPELINE -- 100% UNSEEN EVALUATION ON DRIVE M")
    print("=" * 75)

    pairs = find_paired_files()
    if not pairs:
        raise FileNotFoundError("No paired files found.")

    # Primary Blind Test Drive: Drive M (Pair 0)
    primary_test_pair = pairs[0]
    print(f"\n[1/7] Target Blind Test File: {os.path.basename(primary_test_pair[0])} (EXCLUDED FROM TRAINING)")

    # Training Drives: Pairs 1..7 (S1, S2, S3a, S3b, S3c, S4, Vfa01)
    train_pairs = pairs[1:8]
    # Validation Drive: Pair 8 (Vfa02)
    val_pair = pairs[8]

    print(f"  Training Files ({len(train_pairs)} pairs): {[os.path.basename(p[0]) for p in train_pairs]}")
    print(f"  Validation File: {os.path.basename(val_pair[0])}")

    # Process Training Drives
    print(f"\n[2/7] Processing {len(train_pairs)} training drives...")
    train_X_list, train_y_list = [], []
    for i, (sp, vp) in enumerate(train_pairs):
        fname = os.path.basename(sp)
        try:
            d = process_file(sp, vp, for_training=True, stride=TRAIN_STRIDE)
            train_X_list.append(d["X"])
            train_y_list.append(d["y"])
            print(f"  [{i+1}/{len(train_pairs)}] {fname}: {d['n_samples']:,} delay windows")
        except Exception as e:
            print(f"  [{i+1}/{len(train_pairs)}] {fname}: FAILED -- {e}")

    X_train = np.concatenate(train_X_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    # Process Validation Drive
    print(f"\n[3/7] Processing validation drive ({os.path.basename(val_pair[0])})...")
    val_data = process_file(val_pair[0], val_pair[1], for_training=True, stride=TRAIN_STRIDE)
    X_val = val_data["X"]
    y_val = val_data["y"]

    print(f"  Total Training Windows:   {len(X_train):,}")
    print(f"  Total Validation Windows: {len(X_val):,}")

    # Process 100% Blind Test Drive M
    print(f"\n[4/7] Processing 100% Blind Test Drive M...")
    m_data = process_file(primary_test_pair[0], primary_test_pair[1], for_training=False, stride=1)
    m_X = m_data["X"]
    m_y = m_data["y"]
    time_axis = m_data["time_axis"]
    gt = m_data["gt"]
    inertial = m_data["inertial"]
    gt_speed = gt["velocity_ms"]
    n_full = len(time_axis)
    print(f"  Drive M Full Samples: {n_full:,} (0 samples in train/val)")

    # Train Cross-Drive Model
    print("\n[5/7] Training IDNN v3 strictly on external drives...")
    model, history = train_crossdrive(X_train, y_train, X_val, y_val)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "X_mean": model.X_mean, "X_std": model.X_std,
        "y_mean": model.y_mean, "y_std": model.y_std,
        "config": {
            "n_features": N_FEATURES,
            "delay_taps": DELAY_TAPS,
            "n_outputs": N_OUTPUTS,
            "hidden_sizes": HIDDEN_SIZES,
            "dropout": DROPOUT,
        },
    }, MODEL_PATH)
    print(f"  Saved Cross-Drive Model: {MODEL_PATH}")

    # Evaluate on 100% Blind Drive M
    print("\n[6/7] Running Inference on 100% Blind Drive M...")
    v3_pred = predict(model, m_X)

    v3_speed = np.zeros(n_full)
    v3_speed[DELAY_TAPS:] = v3_pred

    v3_kf_speed = kalman_filter_speed(v3_speed)
    accel_for_zupt = m_data["ts_feats"][:, :3]
    v3_zupt_speed = apply_zupt(v3_kf_speed, accel_for_zupt)

    v3_dist = speed_to_distance(v3_speed, time_axis)
    v3_kf_dist = speed_to_distance(v3_kf_speed, time_axis)
    v3_zupt_dist = speed_to_distance(v3_zupt_speed, time_axis)

    # Load v1/v2 for comparison
    ts_feats = m_data["ts_feats"]
    v1_speed = load_v1_predictions(ts_feats, time_axis)
    v2_speed = load_v2_predictions(primary_test_pair[0], primary_test_pair[1], time_axis)

    # ----------------------------------------------------------
    # COMPUTE DRIFT TABLE
    # ----------------------------------------------------------
    print("\n[7/7] Computing Cross-Drive Drift Metrics on Drive M...")
    dt = 0.1
    bl_speed = inertial["speed"]

    # Segment boundaries for Drive M (70% / 15% / 15%)
    idx_70 = int(n_full * 0.70)
    idx_85 = int(n_full * 0.85)

    def d(sp, start, end):
        return drift_on_split(sp[start:end], gt_speed[start:end], dt)

    models_info = [
        ("Baseline INS", bl_speed),
        ("IDNN v1 (Single-Drive)", v1_speed),
        ("IDNN v2 (Single-Drive)", v2_speed),
        ("v3 Blind Raw", v3_speed),
        ("v3 Blind+KF", v3_kf_speed),
        ("v3 Blind+KF+ZUPT", v3_zupt_speed),
    ]

    headers = [
        "Model",
        "Full Drive M (100% Blind)",
        "Segment 1 (0-70%)",
        "Segment 2 (70-85%)",
        "Segment 3 (85-100%)",
        "Speed RMSE (Blind)",
    ]
    rows = []

    print(f"\n{'='*100}")
    print(f"{'Model':<24s} {'Full (100% Blind)':<20s} {'Seg 1 (0-70%)':<18s} {'Seg 2 (70-85%)':<18s} {'Seg 3 (85-100%)':<18s} {'Speed RMSE':>12s}")
    print(f"{'-'*100}")

    for name, sp in models_info:
        if sp is None:
            rows.append([name, "N/A", "N/A", "N/A", "N/A", "N/A"])
            print(f"{name:<24s} {'N/A':>16s}   {'N/A':>14s}   {'N/A':>14s}   {'N/A':>14s}   {'N/A':>10s}")
            continue

        full = d(sp, 0, n_full)
        s1 = d(sp, 0, idx_70)
        s2 = d(sp, idx_70, idx_85)
        s3 = d(sp, idx_85, n_full)
        rmse = np.sqrt(np.mean((sp - gt_speed)**2)) * 3.6

        rows.append([name, f"{full:.1f}%", f"{s1:.1f}%", f"{s2:.1f}%", f"{s3:.1f}%", f"{rmse:.1f} km/h"])
        print(f"{name:<24s} {full:>16.1f}%   {s1:>14.1f}%   {s2:>14.1f}%   {s3:>14.1f}%   {rmse:>10.1f} km/h")

    print(f"{'='*100}")

    final_blind_drift = d(v3_zupt_speed, 0, n_full)
    print(f"\n>>> 100% Blind Cross-Drive Drift on Full Drive M: {final_blind_drift:.1f}% <<<")

    # Generate plots
    print("\nGenerating Cross-Drive comparison plots...")
    generate_crossdrive_plots(
        time_axis, gt, inertial,
        v1_speed, v2_speed,
        v3_speed, v3_kf_speed, v3_zupt_speed,
        v3_dist, v3_kf_dist, v3_zupt_dist,
        history,
    )
    save_drift_table_image(rows, headers)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print(f"\n{'='*75}")
    print("CROSS-DRIVE BLIND TEST COMPLETE")
    print(f"{'='*75}")


if __name__ == "__main__":
    run_crossdrive_pipeline()

