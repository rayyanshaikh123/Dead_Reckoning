"""
IDNN v2 Pipeline — Hand-motion-robust features + 3-way comparison.

Generates comparison plots in results/plots/idnn_v2/ showing
Baseline vs IDNN v1 vs IDNN v2 performance.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.idnn_model_v2 import IDNNv2
from src.idnn_model import IDNN
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

OUTPUT_DIR = "results/plots/idnn_v2"
MODEL_SAVE_PATH = "results/idnn_v2_model.pth"
V1_MODEL_PATH = "results/idnn_model.pth"

# Window
WINDOW_SIZE = 20        # 2 seconds at 10 Hz
SAMPLE_RATE = 10.0      # Hz

# Features: 33 total
# 9 accel stats + 6 gyro stats + 9 spectral + 3 mag + 4 orient + 2 grav
N_FEATURES = 33
N_OUTPUTS = 1

# Model
HIDDEN_SIZES = (256, 128, 64)
DROPOUT = 0.2

# Training
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_window_features(
    lin_ax, lin_ay, lin_az,
    gx, gy, gz,
    mag_x, mag_y, mag_z,
    orient_yaw, orient_pitch, orient_roll,
    grav_mag,
    window_size=WINDOW_SIZE,
):
    """
    Extract hand-motion-robust features from sliding windows.

    Returns:
        features: (N - window_size + 1, 33) array
    """

    n = len(lin_ax)
    accel = np.column_stack([lin_ax, lin_ay, lin_az])    # (N, 3)
    gyro = np.column_stack([gx, gy, gz])                 # (N, 3)

    # Sliding windows: (N-W+1, W, C)
    accel_win = sliding_window_view(accel, window_shape=window_size, axis=0)
    # shape: (N-W+1, 3, W) — need to transpose last two
    accel_win = np.moveaxis(accel_win, -1, 1)  # (N-W+1, W, 3)

    gyro_win = sliding_window_view(gyro, window_shape=window_size, axis=0)
    gyro_win = np.moveaxis(gyro_win, -1, 1)

    n_win = accel_win.shape[0]

    # ---- Group 1: Acceleration Statistics (9 features) ----
    accel_mean = np.mean(accel_win, axis=1)       # (n_win, 3)
    accel_std = np.std(accel_win, axis=1)          # (n_win, 3)
    accel_median = np.median(accel_win, axis=1)    # (n_win, 3)

    # ---- Group 2: Gyroscope Statistics (6 features) ----
    gyro_mean = np.mean(gyro_win, axis=1)          # (n_win, 3)
    gyro_std = np.std(gyro_win, axis=1)            # (n_win, 3)

    # ---- Group 3: Spectral Band Energies (9 features) ----
    freqs = np.fft.rfftfreq(window_size, d=1.0 / SAMPLE_RATE)
    band1 = (freqs >= 0) & (freqs < 1.0)    # 0–1 Hz: vehicle dynamics
    band2 = (freqs >= 1.0) & (freqs < 3.0)  # 1–3 Hz: mixed
    band3 = (freqs >= 3.0)                    # 3–5 Hz: vibration + hand

    fft_accel = np.fft.rfft(accel_win, axis=1)       # (n_win, W//2+1, 3)
    power = np.abs(fft_accel) ** 2

    e1 = np.sum(power[:, band1, :], axis=1)   # (n_win, 3)
    e2 = np.sum(power[:, band2, :], axis=1)
    e3 = np.sum(power[:, band3, :], axis=1)

    # ---- Group 4: Magnetometer (3 features) ----
    mag = np.column_stack([mag_x, mag_y, mag_z])
    mag_win = sliding_window_view(mag, window_shape=window_size, axis=0)
    mag_win = np.moveaxis(mag_win, -1, 1)

    mag_mean = np.mean(mag_win, axis=1)   # (n_win, 3)
    mag_magnitude = np.sqrt(np.sum(mag_mean ** 2, axis=1, keepdims=True))  # (n_win, 1)

    # Heading from magnetometer: atan2(Y, X)
    mag_heading = np.arctan2(mag_mean[:, 1], mag_mean[:, 0])  # (n_win,)
    mag_sin = np.sin(mag_heading).reshape(-1, 1)
    mag_cos = np.cos(mag_heading).reshape(-1, 1)

    # ---- Group 5: Orientation (4 features) ----
    orient_yaw_rad = np.deg2rad(orient_yaw)
    orient_win_sin = sliding_window_view(np.sin(orient_yaw_rad), window_shape=window_size, axis=0)
    orient_win_cos = sliding_window_view(np.cos(orient_yaw_rad), window_shape=window_size, axis=0)
    orient_pitch_win = sliding_window_view(orient_pitch, window_shape=window_size, axis=0)
    orient_roll_win = sliding_window_view(orient_roll, window_shape=window_size, axis=0)

    yaw_sin_mean = np.mean(orient_win_sin, axis=1).reshape(-1, 1)
    yaw_cos_mean = np.mean(orient_win_cos, axis=1).reshape(-1, 1)
    pitch_mean = np.mean(orient_pitch_win, axis=1).reshape(-1, 1)
    roll_mean = np.mean(orient_roll_win, axis=1).reshape(-1, 1)

    # ---- Group 6: Gravity Stability (2 features) ----
    grav_win = sliding_window_view(grav_mag, window_shape=window_size, axis=0)
    grav_mean = np.mean(grav_win, axis=1).reshape(-1, 1)
    grav_std = np.std(grav_win, axis=1).reshape(-1, 1)

    # ---- Concatenate all features ----
    features = np.hstack([
        accel_mean,          # 3  (1-3)
        accel_std,           # 3  (4-6)
        accel_median,        # 3  (7-9)
        gyro_mean,           # 3  (10-12)
        gyro_std,            # 3  (13-15)
        e1,                  # 3  (16-18) spectral 0-1 Hz
        e2,                  # 3  (19-21) spectral 1-3 Hz
        e3,                  # 3  (22-24) spectral 3-5 Hz
        mag_magnitude,       # 1  (25)
        mag_sin,             # 1  (26)
        mag_cos,             # 1  (27)
        yaw_sin_mean,        # 1  (28)
        yaw_cos_mean,        # 1  (29)
        pitch_mean,          # 1  (30)
        roll_mean,           # 1  (31)
        grav_mean,           # 1  (32)
        grav_std,            # 1  (33)
    ]).astype(np.float32)

    assert features.shape[1] == N_FEATURES, f"Expected {N_FEATURES}, got {features.shape[1]}"

    return features


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_all_data(s_df, v_df, s_cols, v_cols, time_axis):
    """
    Extract raw sensor arrays, compute baseline & GT,
    and build v2 feature matrix.
    """

    # ---- Raw accelerometer ----
    ax = numeric_series(s_df, s_cols["ax"]).to_numpy().astype(float)
    ay = numeric_series(s_df, s_cols["ay"]).to_numpy().astype(float)
    az = numeric_series(s_df, s_cols["az"]).to_numpy().astype(float)

    grav_x = numeric_series(s_df, s_cols["grav_x"])
    grav_y = numeric_series(s_df, s_cols["grav_y"])
    grav_z = numeric_series(s_df, s_cols["grav_z"])

    gx = numeric_series(s_df, s_cols["gx"])
    gy = numeric_series(s_df, s_cols["gy"])
    gz = numeric_series(s_df, s_cols["gz"])

    # Unit check
    mag = np.sqrt(np.nan_to_num(ax)**2 + np.nan_to_num(ay)**2 + np.nan_to_num(az)**2)
    if 0.5 <= np.nanmedian(mag) <= 2.0:
        ax *= G; ay *= G; az *= G
        if grav_x is not None:
            grav_x = grav_x * G; grav_y = grav_y * G; grav_z = grav_z * G

    # Interpolate all
    def interp(arr):
        if arr is None:
            return np.zeros(len(ax))
        if isinstance(arr, pd.Series):
            arr = arr.interpolate().ffill().bfill().to_numpy().astype(float)
        else:
            arr = pd.Series(arr).interpolate().ffill().bfill().to_numpy().astype(float)
        return arr

    ax = interp(ax); ay = interp(ay); az = interp(az)
    grav_x_arr = interp(grav_x); grav_y_arr = interp(grav_y); grav_z_arr = interp(grav_z)
    gx_arr = interp(gx); gy_arr = interp(gy); gz_arr = interp(gz)

    # Linear acceleration (gravity subtracted)
    lin_ax = ax - grav_x_arr
    lin_ay = ay - grav_y_arr
    lin_az = az - grav_z_arr

    # ---- Magnetometer ----
    mag_cols = {}
    for col in s_df.columns:
        cl = col.strip().upper()
        if "MAGNETIC" in cl and "X" in cl:
            mag_cols["mx"] = col
        elif "MAGNETIC" in cl and "Y" in cl:
            mag_cols["my"] = col
        elif "MAGNETIC" in cl and "Z" in cl:
            mag_cols["mz"] = col

    mag_x = interp(numeric_series(s_df, mag_cols.get("mx")))
    mag_y = interp(numeric_series(s_df, mag_cols.get("my")))
    mag_z = interp(numeric_series(s_df, mag_cols.get("mz")))

    # ---- Orientation ----
    orient_cols = {}
    for col in s_df.columns:
        cl = col.strip().upper()
        if "ORIENTATION" in cl and "YAW" in cl:
            orient_cols["yaw"] = col
        elif "ORIENTATION" in cl and "PITCH" in cl:
            orient_cols["pitch"] = col
        elif "ORIENTATION" in cl and "ROLL" in cl:
            orient_cols["roll"] = col

    orient_yaw = interp(numeric_series(s_df, orient_cols.get("yaw")))
    orient_pitch = interp(numeric_series(s_df, orient_cols.get("pitch")))
    orient_roll = interp(numeric_series(s_df, orient_cols.get("roll")))

    # ---- Gravity magnitude ----
    grav_mag = np.sqrt(grav_x_arr**2 + grav_y_arr**2 + grav_z_arr**2)

    # ---- Baseline inertial (for comparison) ----
    inertial = compute_baseline_inertial(
        ax.copy(), ay.copy(), az.copy(),
        grav_x_arr.copy(), grav_y_arr.copy(), grav_z_arr.copy(),
        time_axis,
    )

    # ---- Vehicle ground truth ----
    gt = compute_vehicle_ground_truth(v_df, v_cols, time_axis)

    # ---- V2 features ----
    features = extract_window_features(
        lin_ax, lin_ay, lin_az,
        gx_arr, gy_arr, gz_arr,
        mag_x, mag_y, mag_z,
        orient_yaw, orient_pitch, orient_roll,
        grav_mag,
    )

    # Target: vehicle speed at end of window
    target = gt["velocity_ms"][WINDOW_SIZE - 1:]

    # Also prepare V1-style features for comparison
    v1_features = np.column_stack([lin_ax, lin_ay, lin_az, gx_arr, gy_arr, gz_arr])

    return features, target, inertial, gt, v1_features


def numeric_series_safe(df, col_name):
    """Safely get a numeric series, returning None if column doesn't exist."""
    if col_name is None:
        return None
    try:
        return pd.to_numeric(df[col_name], errors="coerce")
    except (KeyError, TypeError):
        return None


# ============================================================
# TRAINING
# ============================================================

def train_v2(X_train, y_train, X_val, y_val):
    """Train IDNN v2 model."""

    model = IDNNv2(
        n_features=N_FEATURES,
        n_outputs=N_OUTPUTS,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nIDNN v2 Architecture:")
    print(f"  Input: {N_FEATURES} window features")
    print(f"  Hidden: {HIDDEN_SIZES} + BatchNorm")
    print(f"  Output: {N_OUTPUTS}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {DEVICE}\n")

    # Normalize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_tr = (X_train - X_mean) / X_std
    X_va = (X_val - X_mean) / X_std
    y_tr = (y_train - y_mean) / y_std
    y_va = (y_val - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float().unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va).float(),
        torch.from_numpy(y_va).float().unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    print(f"Training for {EPOCHS} epochs  (train={len(X_train):,}  val={len(X_val):,})")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        t_total, t_count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_total += loss.item() * len(xb)
            t_count += len(xb)
        train_loss = t_total / t_count

        model.eval()
        v_total, v_count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                v_total += loss.item() * len(xb)
                v_count += len(xb)
        val_loss = v_total / v_count

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  best={best_val:.6f}  [{time.time()-t0:.1f}s]")

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nTraining done in {time.time()-t0:.1f}s.  Best val: {best_val:.6f}")

    model.X_mean = X_mean
    model.X_std = X_std
    model.y_mean = y_mean
    model.y_std = y_std

    return model, history


def predict_v2(model, X):
    """Run inference with v2 model."""
    model.eval()
    X_norm = (X - model.X_mean) / model.X_std
    X_t = torch.from_numpy(X_norm).float().to(DEVICE)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy().flatten()
    pred = pred_norm * model.y_std + model.y_mean
    return np.maximum(pred, 0.0)


# ============================================================
# V1 MODEL LOADING & PREDICTION
# ============================================================

def load_and_predict_v1(v1_features, time_axis):
    """Load IDNN v1 checkpoint and predict."""

    if not os.path.exists(V1_MODEL_PATH):
        print(f"  V1 model not found at {V1_MODEL_PATH}, skipping.")
        return None, None

    ckpt = torch.load(V1_MODEL_PATH, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]

    model_v1 = IDNN(
        n_features=cfg["n_features"],
        delay_taps=cfg["delay_taps"],
        n_outputs=cfg["n_outputs"],
        hidden_sizes=cfg["hidden_sizes"],
        dropout=cfg["dropout"],
    )
    model_v1.load_state_dict(ckpt["model_state"])
    model_v1.X_mean = ckpt["X_mean"]
    model_v1.X_std = ckpt["X_std"]
    model_v1.y_mean = ckpt["y_mean"]
    model_v1.y_std = ckpt["y_std"]
    model_v1.eval()

    delay = cfg["delay_taps"]

    # Create v1-style delay windows
    from src.idnn_pipeline import create_delay_windows
    X_v1, _ = create_delay_windows(v1_features, np.zeros(len(v1_features)), delay)

    # Predict
    from src.idnn_pipeline import predict as predict_v1
    v1_speed = predict_v1(model_v1, X_v1)

    # Full-length speed (pad beginning)
    n = len(time_axis)
    v1_speed_full = np.zeros(n)
    v1_speed_full[delay:] = v1_speed

    # Cumulative distance
    dt_arr = np.diff(time_axis, prepend=time_axis[0])
    dt_med = np.median(np.diff(time_axis))
    dt_arr[0] = dt_med
    v1_dist = np.zeros(n)
    for i in range(1, n):
        dt_i = dt_arr[i] if (dt_arr[i] > 0 and np.isfinite(dt_arr[i])) else dt_med
        v1_dist[i] = v1_dist[i-1] + v1_speed_full[i] * dt_i

    return v1_speed_full, v1_dist


# ============================================================
# DISTANCE FROM SPEED
# ============================================================

def speed_to_distance(speed_array, time_axis):
    """Integrate speed to cumulative distance."""
    n = len(time_axis)
    dt_arr = np.diff(time_axis, prepend=time_axis[0])
    dt_med = np.median(np.diff(time_axis))
    dt_arr[0] = dt_med

    dist = np.zeros(n)
    for i in range(1, n):
        dt_i = dt_arr[i] if (dt_arr[i] > 0 and np.isfinite(dt_arr[i])) else dt_med
        dist[i] = dist[i-1] + speed_array[i] * dt_i
    return dist


# ============================================================
# DRIFT COMPUTATION ON SPLITS
# ============================================================

def compute_drift_on_split(speed_pred, speed_gt, dt=0.1):
    """Compute distance drift % for a given split."""
    gt_dist = np.sum(speed_gt * dt)
    pred_dist = np.sum(speed_pred * dt)
    if gt_dist < 1.0:
        return float("nan"), 0, 0
    drift_pct = abs(pred_dist - gt_dist) / gt_dist * 100
    return drift_pct, pred_dist, gt_dist


# ============================================================
# COMPARISON PLOTS
# ============================================================

def generate_all_plots(
    time_axis, gt, inertial,
    v1_speed, v1_dist,
    v2_speed, v2_dist,
    history_v2,
    split_indices,
):
    """Generate 3-way comparison plots."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11})

    gt_speed = gt["velocity_ms"]
    gt_dist = gt["gt_distance"]
    bl_speed = inertial["speed"]
    bl_dist = inertial["cumulative_distance"]

    train_end, val_end = split_indices

    # Colors
    C_GT = "tab:blue"
    C_BL = "tab:orange"
    C_V1 = "tab:red"
    C_V2 = "tab:green"

    # ========================================================
    # 1. Training Loss
    # ========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history_v2["train_loss"]) + 1)
    ax.plot(epochs, history_v2["train_loss"], label="Train", lw=1.5)
    ax.plot(epochs, history_v2["val_loss"], label="Val", lw=1.5)
    ax.set(title="IDNN v2 Training Loss", xlabel="Epoch", ylabel="MSE Loss (normalized)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_v2_training_loss.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 2. Speed: GT vs Baseline vs V1 vs V2
    # ========================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(time_axis, gt_speed*3.6, lw=1.5, alpha=0.9, label="Vehicle GT", color=C_GT)
    ax.plot(time_axis, bl_speed*3.6, lw=0.8, alpha=0.3, label="Baseline INS", color=C_BL)
    if v1_speed is not None:
        ax.plot(time_axis, v1_speed*3.6, lw=1, alpha=0.6, label="IDNN v1", color=C_V1)
    ax.plot(time_axis, v2_speed*3.6, lw=1, alpha=0.8, label="IDNN v2", color=C_V2)

    # Mark train/val/test regions
    ax.axvline(time_axis[train_end], ls="--", color="gray", alpha=0.5, label="Train|Val split")
    ax.axvline(time_axis[val_end], ls="--", color="black", alpha=0.5, label="Val|Test split")

    ax.set(title="Speed Comparison: GT vs All Models", xlabel="Time (s)", ylabel="Speed (km/h)")
    ax.set_ylim(-5, max(np.max(gt_speed*3.6)*1.3, 120))
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_speed_comparison_all.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 3. V2 speed vs GT zoomed
    # ========================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, gt_speed*3.6, lw=1.5, label="Vehicle GT", color=C_GT)
    ax.plot(time_axis, v2_speed*3.6, lw=1, alpha=0.8, label="IDNN v2", color=C_V2)
    ax.axvline(time_axis[val_end], ls="--", color="black", alpha=0.5, label="Test region →")
    ax.set(title="IDNN v2 Speed vs Ground Truth", xlabel="Time (s)", ylabel="Speed (km/h)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_v2_speed_vs_gt.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 4. Cumulative Distance
    # ========================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, gt_dist, lw=2, label="Vehicle GT", color=C_GT)
    ax.plot(time_axis, bl_dist, lw=1.5, alpha=0.5, label="Baseline INS", color=C_BL)
    if v1_dist is not None:
        ax.plot(time_axis, v1_dist, lw=1.5, alpha=0.7, label="IDNN v1", color=C_V1)
    ax.plot(time_axis, v2_dist, lw=1.5, alpha=0.8, label="IDNN v2", color=C_V2)
    ax.set(title="Cumulative Distance: All Models", xlabel="Time (s)", ylabel="Distance (m)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_cumulative_distance_all.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 5. Distance Error
    # ========================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    bl_err = np.abs(bl_dist - gt_dist)
    ax.plot(time_axis, bl_err, lw=1, alpha=0.5, label="Baseline", color=C_BL)
    if v1_dist is not None:
        v1_err = np.abs(v1_dist - gt_dist)
        ax.plot(time_axis, v1_err, lw=1.2, alpha=0.7, label="IDNN v1", color=C_V1)
    v2_err = np.abs(v2_dist - gt_dist)
    ax.plot(time_axis, v2_err, lw=1.2, alpha=0.8, label="IDNN v2", color=C_V2)
    ax.set(title="Distance Error Over Time", xlabel="Time (s)", ylabel="Error (m)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_distance_error_all.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 6. Drift %
    # ========================================================
    _, bl_drift = calculate_scalar_drift(bl_dist, gt_dist)
    _, v2_drift_arr = calculate_scalar_drift(v2_dist, gt_dist)

    fig, ax = plt.subplots(figsize=(14, 6))
    valid_b = np.isfinite(bl_drift)
    valid_v2 = np.isfinite(v2_drift_arr)

    if np.any(valid_b):
        ax.plot(gt_dist[valid_b], bl_drift[valid_b], lw=1, alpha=0.5, label="Baseline", color=C_BL)
    if v1_dist is not None:
        _, v1_drift_arr = calculate_scalar_drift(v1_dist, gt_dist)
        valid_v1 = np.isfinite(v1_drift_arr)
        if np.any(valid_v1):
            ax.plot(gt_dist[valid_v1], v1_drift_arr[valid_v1], lw=1.2, alpha=0.7, label="IDNN v1", color=C_V1)
    if np.any(valid_v2):
        ax.plot(gt_dist[valid_v2], v2_drift_arr[valid_v2], lw=1.2, alpha=0.8, label="IDNN v2", color=C_V2)

    ax.axhline(10, ls="--", lw=1.5, color="red", label="10% Threshold")
    max_v2 = np.nanmax(v2_drift_arr[valid_v2]) if np.any(valid_v2) else 100
    ax.set_ylim(0, min(200, max(max_v2 * 1.5, 25)))
    ax.set(title="Drift % vs Distance: All Models", xlabel="Distance (m)", ylabel="Drift (%)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_drift_comparison_all.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 7. Speed Error Histogram (3 panels)
    # ========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bl_se = (bl_speed - gt_speed) * 3.6
    axes[0].hist(bl_se, bins=100, alpha=0.7, color=C_BL)
    axes[0].set_title(f"Baseline (RMSE={np.sqrt(np.mean(bl_se**2)):.1f} km/h)")
    axes[0].set_xlabel("Speed Error (km/h)")
    axes[0].grid(True, alpha=0.3)

    if v1_speed is not None:
        v1_se = (v1_speed - gt_speed) * 3.6
        axes[1].hist(v1_se, bins=100, alpha=0.7, color=C_V1)
        axes[1].set_title(f"IDNN v1 (RMSE={np.sqrt(np.mean(v1_se**2)):.1f} km/h)")
    else:
        axes[1].set_title("IDNN v1 (not available)")
    axes[1].set_xlabel("Speed Error (km/h)")
    axes[1].grid(True, alpha=0.3)

    v2_se = (v2_speed - gt_speed) * 3.6
    axes[2].hist(v2_se, bins=100, alpha=0.7, color=C_V2)
    axes[2].set_title(f"IDNN v2 (RMSE={np.sqrt(np.mean(v2_se**2)):.1f} km/h)")
    axes[2].set_xlabel("Speed Error (km/h)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Speed Error Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_speed_error_hist.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 8. Summary Metrics Bar Chart
    # ========================================================
    bl_rmse = np.sqrt(np.mean((bl_speed - gt_speed)**2)) * 3.6
    v1_rmse = np.sqrt(np.mean((v1_speed - gt_speed)**2)) * 3.6 if v1_speed is not None else 0
    v2_rmse = np.sqrt(np.mean((v2_speed - gt_speed)**2)) * 3.6

    bl_fd = abs(bl_dist[-1] - gt_dist[-1]) / 1000
    v1_fd = abs(v1_dist[-1] - gt_dist[-1]) / 1000 if v1_dist is not None else 0
    v2_fd = abs(v2_dist[-1] - gt_dist[-1]) / 1000

    bl_dp = abs(bl_dist[-1] - gt_dist[-1]) / gt_dist[-1] * 100
    v1_dp = abs(v1_dist[-1] - gt_dist[-1]) / gt_dist[-1] * 100 if v1_dist is not None else 0
    v2_dp = abs(v2_dist[-1] - gt_dist[-1]) / gt_dist[-1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["Baseline", "IDNN v1", "IDNN v2"]
    colors = [C_BL, C_V1, C_V2]

    for ax, title, vals, unit in zip(
        axes,
        ["Speed RMSE", "Final Distance Error", "Final Drift (Full Data)"],
        [[bl_rmse, v1_rmse, v2_rmse], [bl_fd, v1_fd, v2_fd], [bl_dp, v1_dp, v2_dp]],
        ["km/h", "km", "%"],
    ):
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel(unit)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if "Drift" in title:
            ax.axhline(10, ls="--", color="red", label="10%")
            ax.legend()

    fig.suptitle("All Models — Summary (Full Dataset)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_summary_metrics_all.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========================================================
    # 9. Comprehensive Drift Table (image)
    # ========================================================
    return  # table printed in main, saved as separate plot below


def save_drift_table_plot(drift_table, output_dir):
    """Save the drift table as an image."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    headers = drift_table["headers"]
    rows = drift_table["rows"]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows
    for i, row in enumerate(rows, 1):
        if "Baseline" in row[0]:
            color = "#FFF2CC"
        elif "v1" in row[0]:
            color = "#FCE4EC"
        else:
            color = "#E8F5E9"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    ax.set_title("Comprehensive Drift Comparison Table",
                 fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_drift_table.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def run_v2_pipeline():

    print("=" * 65)
    print("IDNN v2 PIPELINE — Hand-Motion-Robust Features")
    print("=" * 65)

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n[1/7] Loading data...")

    pairs = find_paired_files()
    if not pairs:
        raise FileNotFoundError("No paired files found.")

    s_path, v_path = pairs[0]
    print(f"  S: {s_path}")
    print(f"  V: {v_path}")

    s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1")
    v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1")

    s_cols = detect_smartphone_columns(s_df)
    v_cols = detect_vehicle_columns(v_df)

    v_time = numeric_series(v_df, v_cols["time"]).to_numpy()
    v_time = v_time - v_time[0]
    v_time = pd.Series(v_time).interpolate().ffill().bfill().to_numpy()
    for i in range(1, len(v_time)):
        if v_time[i] <= v_time[i-1]:
            v_time[i] = v_time[i-1] + 0.1
    time_axis = v_time

    min_len = min(len(s_df), len(v_df), len(time_axis))
    s_df = s_df.iloc[:min_len].reset_index(drop=True)
    v_df = v_df.iloc[:min_len].reset_index(drop=True)
    time_axis = time_axis[:min_len]

    print(f"  Samples: {min_len:,}  Duration: {time_axis[-1]:.0f}s")

    # ----------------------------------------------------------
    # 2. Extract features
    # ----------------------------------------------------------
    print("\n[2/7] Extracting v2 features...")

    features_v2, target, inertial, gt, v1_raw = prepare_all_data(
        s_df, v_df, s_cols, v_cols, time_axis,
    )

    print(f"  V2 features: {features_v2.shape}  ({N_FEATURES} per window)")
    print(f"  Targets: {target.shape}")

    # ----------------------------------------------------------
    # 3. Split
    # ----------------------------------------------------------
    print("\n[3/7] Temporal split...")

    n = len(features_v2)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = features_v2[:train_end], target[:train_end]
    X_val, y_val = features_v2[train_end:val_end], target[train_end:val_end]
    X_test, y_test = features_v2[val_end:], target[val_end:]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Map back to full time indices (window features start at index WINDOW_SIZE-1)
    full_train_end = train_end + WINDOW_SIZE - 1
    full_val_end = val_end + WINDOW_SIZE - 1

    # ----------------------------------------------------------
    # 4. Train v2
    # ----------------------------------------------------------
    print("\n[4/7] Training IDNN v2...")

    model_v2, history = train_v2(X_train, y_train, X_val, y_val)

    # Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state": model_v2.state_dict(),
        "X_mean": model_v2.X_mean,
        "X_std": model_v2.X_std,
        "y_mean": model_v2.y_mean,
        "y_std": model_v2.y_std,
        "config": {
            "n_features": N_FEATURES,
            "n_outputs": N_OUTPUTS,
            "hidden_sizes": HIDDEN_SIZES,
            "dropout": DROPOUT,
        },
    }, MODEL_SAVE_PATH)
    print(f"  Saved: {MODEL_SAVE_PATH}")

    # ----------------------------------------------------------
    # 5. Load IDNN v1
    # ----------------------------------------------------------
    print("\n[5/7] Loading IDNN v1 for comparison...")

    v1_speed, v1_dist = load_and_predict_v1(v1_raw, time_axis)
    if v1_speed is not None:
        print(f"  V1 loaded. Speed range: {v1_speed.min()*3.6:.1f}–{v1_speed.max()*3.6:.1f} km/h")
    else:
        print("  V1 not available.")

    # ----------------------------------------------------------
    # 6. Evaluate all models
    # ----------------------------------------------------------
    print("\n[6/7] Evaluating all models...")

    # V2 full prediction
    v2_pred_all = predict_v2(model_v2, features_v2)

    # Full-length speed (pad start)
    v2_speed_full = np.zeros(len(time_axis))
    v2_speed_full[WINDOW_SIZE-1:] = v2_pred_all

    # Full-length distance
    v2_dist_full = speed_to_distance(v2_speed_full, time_axis)

    gt_speed = gt["velocity_ms"]
    gt_dist = gt["gt_distance"]
    bl_speed = inertial["speed"]

    # ---- Compute drift on every split ----
    dt = 0.1

    # Helper: get speed arrays for a split (by full-array indices)
    def split_drift(speed_arr, start, end):
        return compute_drift_on_split(speed_arr[start:end], gt_speed[start:end], dt)

    # Full data
    bl_full, _, _ = split_drift(bl_speed, 0, len(time_axis))
    v2_full, _, _ = split_drift(v2_speed_full, 0, len(time_axis))

    # Train
    bl_train, _, _ = split_drift(bl_speed, 0, full_train_end)
    v2_train, _, _ = split_drift(v2_speed_full, 0, full_train_end)

    # Val
    bl_val, _, _ = split_drift(bl_speed, full_train_end, full_val_end)
    v2_val, _, _ = split_drift(v2_speed_full, full_train_end, full_val_end)

    # Test
    bl_test, _, _ = split_drift(bl_speed, full_val_end, len(time_axis))
    v2_test, _, _ = split_drift(v2_speed_full, full_val_end, len(time_axis))

    # V1 drifts
    if v1_speed is not None:
        v1_full, _, _ = split_drift(v1_speed, 0, len(time_axis))
        v1_train, _, _ = split_drift(v1_speed, 0, full_train_end)
        v1_val, _, _ = split_drift(v1_speed, full_train_end, full_val_end)
        v1_test, _, _ = split_drift(v1_speed, full_val_end, len(time_axis))
    else:
        v1_full = v1_train = v1_val = v1_test = float("nan")

    # Speed RMSE on test
    bl_rmse_test = np.sqrt(np.mean((bl_speed[full_val_end:] - gt_speed[full_val_end:])**2)) * 3.6
    v2_rmse_test = np.sqrt(np.mean((v2_speed_full[full_val_end:] - gt_speed[full_val_end:])**2)) * 3.6
    v1_rmse_test = np.sqrt(np.mean((v1_speed[full_val_end:] - gt_speed[full_val_end:])**2)) * 3.6 if v1_speed is not None else float("nan")

    # ---- Print comprehensive table ----
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DRIFT TABLE")
    print("=" * 80)

    hdr = f"{'Model':<16s} {'Full (100%)':<14s} {'Train (70%)':<14s} {'Val (15%)':<14s} {'Test (15%)':<14s} {'Test RMSE':>10s}"
    print(hdr)
    print("-" * 80)
    print(f"{'Baseline INS':<16s} {bl_full:>10.1f}%    {bl_train:>10.1f}%    {bl_val:>10.1f}%    {bl_test:>10.1f}%    {bl_rmse_test:>8.1f} km/h")
    print(f"{'IDNN v1':<16s} {v1_full:>10.1f}%    {v1_train:>10.1f}%    {v1_val:>10.1f}%    {v1_test:>10.1f}%    {v1_rmse_test:>8.1f} km/h")
    print(f"{'IDNN v2':<16s} {v2_full:>10.1f}%    {v2_train:>10.1f}%    {v2_val:>10.1f}%    {v2_test:>10.1f}%    {v2_rmse_test:>8.1f} km/h")
    print("=" * 80)
    print(f"10% PS threshold: {'PASS' if v2_test <= 10 else 'FAIL'} (v2 test = {v2_test:.1f}%)")
    print()

    # Build table for image
    drift_table = {
        "headers": ["Model", "Full Data (100%)", "Train (70%)", "Val (15%)", "Test (15%)", "Test Speed RMSE"],
        "rows": [
            ["Baseline INS", f"{bl_full:.1f}%", f"{bl_train:.1f}%", f"{bl_val:.1f}%", f"{bl_test:.1f}%", f"{bl_rmse_test:.1f} km/h"],
            ["IDNN v1", f"{v1_full:.1f}%", f"{v1_train:.1f}%", f"{v1_val:.1f}%", f"{v1_test:.1f}%", f"{v1_rmse_test:.1f} km/h"],
            ["IDNN v2", f"{v2_full:.1f}%", f"{v2_train:.1f}%", f"{v2_val:.1f}%", f"{v2_test:.1f}%", f"{v2_rmse_test:.1f} km/h"],
        ],
    }

    # ----------------------------------------------------------
    # 7. Generate plots
    # ----------------------------------------------------------
    print("[7/7] Generating comparison plots...")

    generate_all_plots(
        time_axis, gt, inertial,
        v1_speed, v1_dist,
        v2_speed_full, v2_dist_full,
        history,
        (full_train_end, full_val_end),
    )

    save_drift_table_plot(drift_table, OUTPUT_DIR)

    print(f"\nSaved 9 plots to: {OUTPUT_DIR}")

    print("\n" + "=" * 65)
    print("IDNN v2 PIPELINE COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_v2_pipeline()

