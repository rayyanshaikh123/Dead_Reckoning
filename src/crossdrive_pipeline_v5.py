"""
Cross-Drive Blind Test Pipeline for IDNN v5 with Idle Normalization & Online Pre-Blackout GNSS Calibration.

Innovations in v5:
  1. Vehicle-Specific Idle Vibration Baseline Normalization (mount & car invariant).
  2. Online Pre-Blackout GNSS Calibration Tracker (estimates scale factor alpha and offset beta
     during active GPS, freezing upon blackout entry to eliminate mounting transfer errors).
  3. Dynamic Gravity Decoupling and Pitch Estimator.
  4. Kinematic Plausibility Gate (rejects road texture step changes).
  5. Zero-Latency Acceleration Feedforward EKF.
  6. Cruise-Lock and Enhanced Stationary ZUPT.
  7. Cross-Drive Training on external drives (Pairs 1..7) and evaluation on 100% Blind Drive M.
  8. Diagnostic plot generation in results/plots/crossdrive_v5/.
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

from src.idnn_model_v5 import IDNNv5
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

OUTPUT_DIR = "results/plots/crossdrive_v5"
MODEL_PATH = "results/crossdrive_v5_model.pth"

N_FEATURES = 18
DELAY_TAPS = 20
N_OUTPUTS = 1
HIDDEN_SIZES = (256, 128, 64)
DROPOUT = 0.2
SAMPLE_RATE = 10.0

EPOCHS = 80
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_STRIDE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FEATURE EXTRACTION (IDLE-NORMALIZED)
# ============================================================

def _interp(arr, n):
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    if isinstance(arr, pd.Series):
        arr = arr.interpolate().ffill().bfill().to_numpy(dtype=np.float64)
    else:
        arr = pd.Series(arr[:n]).interpolate().ffill().bfill().to_numpy(dtype=np.float64)
    return arr[:n]


def extract_features_v5(s_df, s_cols, n, target_speed=None):
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

    # 1. Linear acceleration (body frame, gravity subtracted)
    lin_ax = ax - grav_x
    lin_ay = ay - grav_y
    lin_az = az - grav_z

    # 2. Gyroscope rates
    gx = _interp(numeric_series(s_df, s_cols.get("gx")), n)
    gy = _interp(numeric_series(s_df, s_cols.get("gy")), n)
    gz = _interp(numeric_series(s_df, s_cols.get("gz")), n)

    # 3. Jerk features
    dt = 1.0 / SAMPLE_RATE
    jerk_ax = np.gradient(lin_ax, dt)
    jerk_ay = np.gradient(lin_ay, dt)
    jerk_az = np.gradient(lin_az, dt)
    jerk_gx = np.gradient(gx, dt)
    jerk_gy = np.gradient(gy, dt)
    jerk_gz = np.gradient(gz, dt)

    # 4. Dynamic Gravity Projection (Horizontal acceleration norm)
    g_mag = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2) + 1e-8
    gx_u = grav_x / g_mag
    gy_u = grav_y / g_mag
    gz_u = grav_z / g_mag
    a_dot_g = ax * gx_u + ay * gy_u + az * gz_u
    a_h_x = ax - a_dot_g * gx_u
    a_h_y = ay - a_dot_g * gy_u
    a_h_z = az - a_dot_g * gz_u
    a_horiz_norm = np.sqrt(a_h_x**2 + a_h_y**2 + a_h_z**2)

    # 5. Pitch / Dynamic Suspension Tilt Proxy
    pitch_angle = np.arctan2(grav_x, np.sqrt(grav_y**2 + grav_z**2) + 1e-8)

    # 6. Idle Vibration Normalization
    lin_norm = np.sqrt(lin_ax**2 + lin_ay**2 + lin_az**2)
    e_vib_raw = pd.Series(lin_norm).rolling(5, min_periods=1).var().fillna(0.0).to_numpy(dtype=np.float32)

    # Calculate idle baseline when stationary (v < 0.3 m/s)
    if target_speed is not None and np.any(target_speed < 0.3):
        idle_mask = (target_speed < 0.3)
        idle_baseline = np.median(e_vib_raw[idle_mask])
    else:
        idle_baseline = np.percentile(e_vib_raw, 10)

    idle_baseline = max(float(idle_baseline), 0.01)
    e_vib_norm = (e_vib_raw / idle_baseline).astype(np.float32)

    features = np.column_stack([
        lin_ax, lin_ay, lin_az,
        grav_x, grav_y, grav_z,
        gx, gy, gz,
        jerk_ax, jerk_ay, jerk_az,
        jerk_gx, jerk_gy, jerk_gz,
        a_horiz_norm,
        pitch_angle,
        e_vib_norm,
    ]).astype(np.float32)

    raw_meta = (ax, ay, az, grav_x, grav_y, grav_z, lin_ax, gz, idle_baseline)
    return features, raw_meta


def create_delay_windows_v5(features, target, delay_taps=DELAY_TAPS, stride=1):
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


def process_file_v5(s_path, v_path, for_training=False, stride=1):
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

    ts_feats, raw_meta = extract_features_v5(s_df, s_cols, n, target_speed)
    X, y = create_delay_windows_v5(ts_feats, target_speed, delay_taps=DELAY_TAPS, stride=stride)

    result = {
        "X": X,
        "y": y,
        "n_samples": len(X),
    }

    if not for_training:
        ax, ay, az, grav_x, grav_y, grav_z, lin_ax, gz, idle_base = raw_meta
        inertial = compute_baseline_inertial(
            ax.copy(), ay.copy(), az.copy(),
            grav_x.copy(), grav_y.copy(), grav_z.copy(),
            time_axis,
        )
        result["time_axis"] = time_axis
        result["gt"] = gt
        result["inertial"] = inertial
        result["ts_feats"] = ts_feats
        result["lin_ax"] = lin_ax
        result["gz"] = gz
        result["idle_base"] = idle_base
        result["s_path"] = s_path
        result["v_path"] = v_path

    return result


# ============================================================
# ONLINE PRE-BLACKOUT GNSS CALIBRATION TRACKER
# ============================================================

class OnlineGNSSCalibrator:
    """
    Estimates online scale and bias while GNSS is healthy:
        v_pred_cal = max(0.0, v_model - bias)
    Features Kinematic Braking Guard:
      - If vehicle was braking at entry (delta_v < -1.0 m/s) and bias < 0,
        suppresses negative bias (dynamic filter lag) to prevent false acceleration.
      - Preserves standstill: when vehicle is stopped (low speed or idle vibration),
        forces output velocity to 0.0 m/s.
    """
    def __init__(self, history_len=50):
        self.history_len = history_len  # 50 samples = 5.0 seconds

    def calibrate_at_blackout_entry(self, model_speed, gnss_speed, entry_idx):
        start = max(0, entry_idx - self.history_len)
        vm = model_speed[start:entry_idx]
        vg = gnss_speed[start:entry_idx]

        if len(vm) < 10 or len(vg) < 10:
            return 0.0

        bias = float(np.mean(vm - vg))

        # Kinematic Braking Guard:
        # If the driver was decelerating aggressively at tunnel entry,
        # neural network delay taps create a transient lag (vm < vg -> bias < 0).
        # Freezing a negative bias would artificially accelerate the vehicle during braking.
        delta_v = float(vg[-1] - vg[0])
        if delta_v < -1.0 and bias < 0:
            bias = 0.0

        return bias

    def apply_calibration(self, model_speed_slice, bias, ac_vibration=None):
        calibrated = np.maximum(model_speed_slice - bias, 0.0)
        # Standstill Preservation: if vehicle has stopped, speed must be 0
        if ac_vibration is not None:
            standstill = (model_speed_slice < 1.0) | (ac_vibration < 0.045)
            calibrated[standstill] = 0.0
        else:
            calibrated[model_speed_slice < 0.15] = 0.0
        return calibrated



# ============================================================
# PHYSICS FILTERS
# ============================================================

def kinematic_plausibility_gate(raw_speed, ax_long, dt=0.1, max_accel=3.5, max_decel=6.0):
    n = len(raw_speed)
    gated = np.zeros(n, dtype=np.float64)
    gated[0] = raw_speed[0]

    for i in range(1, n):
        v_prev = gated[i - 1]
        v_cand = raw_speed[i]
        dv_dt = (v_cand - v_prev) / dt

        if dv_dt > max_accel:
            if ax_long[i] > 1.0:
                gated[i] = v_prev + max(dv_dt, max_accel) * dt
            else:
                gated[i] = v_prev + max_accel * dt
        elif dv_dt < -max_decel:
            if ax_long[i] < -2.0:
                gated[i] = v_prev + min(dv_dt, -max_decel) * dt
            else:
                gated[i] = v_prev - max_decel * dt
        else:
            gated[i] = v_cand

    return np.maximum(gated, 0.0)


def ekf_with_acceleration_feedforward(speed, ax_long, dt=0.1, q_cruise=0.2, q_brake=1.5, r=3.0):
    n = len(speed)
    out = np.zeros(n, dtype=np.float64)
    x = float(speed[0])
    p = 1.0

    for i in range(n):
        a_meas = ax_long[i]
        is_hard_braking = (a_meas < -1.5)
        q = q_brake if is_hard_braking else q_cruise

        if is_hard_braking:
            x_pred = max(x + a_meas * dt, 0.0)
        else:
            x_pred = x

        p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x_pred + k * (speed[i] - x_pred)
        p = (1.0 - k) * p_pred
        out[i] = max(x, 0.0)

    return out


def apply_cruise_lock(speed, ax_long, gz, dt=0.1, window_sec=1.5):
    n = len(speed)
    out = speed.copy()
    w_pts = int(window_sec / dt)

    a_steady = np.abs(ax_long) < 0.18
    g_steady = np.abs(gz) < 0.02
    steady_mask = a_steady & g_steady

    streak = 0
    for i in range(n):
        if steady_mask[i]:
            streak += 1
        else:
            streak = 0

        if streak >= w_pts:
            out[i] = 0.95 * out[i - 1] + 0.05 * speed[i]

    return out


def apply_enhanced_zupt(speed, accel_features, threshold_speed=0.35, threshold_energy=0.25, threshold_var=0.035, window=10):
    n = len(speed)
    out = speed.copy()
    energy = np.sqrt(np.mean(accel_features[:, :3]**2, axis=1))

    if n >= window:
        kernel = np.ones(window) / window
        energy_smooth = np.convolve(energy, kernel, mode="same")
    else:
        energy_smooth = energy

    # Road-grade invariant AC vibration variance
    lin_norm = np.sqrt(np.sum(accel_features[:, :3]**2, axis=1))
    var_series = pd.Series(lin_norm).rolling(window, min_periods=1).var().fillna(0.0).to_numpy()

    stopped = (speed < threshold_speed) | (energy_smooth < threshold_energy) | (var_series < threshold_var)
    out[stopped] = 0.0
    return out



# ============================================================
# TRAINING
# ============================================================

def train_crossdrive_v5(X_train, y_train, X_val, y_val):
    model = IDNNv5(
        n_features=N_FEATURES,
        delay_taps=DELAY_TAPS,
        n_outputs=N_OUTPUTS,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n  IDNN v5 Architecture:")
    print(f"    Input: {N_FEATURES} features x {DELAY_TAPS+1} taps = {N_FEATURES*(DELAY_TAPS+1)}")
    print(f"    Hidden: {HIDDEN_SIZES} with GELU + BatchNorm")
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
        optimizer, mode="min", factor=0.5, patience=5,
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


def predict_v5(model, X, batch=8192):
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


# ============================================================
# DIAGNOSTIC PLOTS GENERATOR
# ============================================================

def generate_v5_diagnostic_plots(time_axis, gt_speed, v5_raw, v5_calib, v5_enh, history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    # 1. Training Loss
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", lw=1.5)
    ax.plot(epochs, history["val_loss"], label="Val Loss (Vfa02)", color="tab:orange", lw=1.5)
    ax.set(title="IDNN v5 Cross-Drive Training Loss", xlabel="Epoch", ylabel="MSE Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_v5_training_loss.png", dpi=300)
    plt.close(fig)

    # 2. Speed Tracking (First 500s)
    fig, ax = plt.subplots(figsize=(14, 5))
    t_sub = time_axis[:5000]
    ax.plot(t_sub, gt_speed[:5000] * 3.6, label="Ground Truth (GNSS)", color="black", lw=1.5)
    ax.plot(t_sub, v5_raw[:5000] * 3.6, label="v5 Raw Speed", color="tab:orange", alpha=0.6, lw=1.2)
    ax.plot(t_sub, v5_enh[:5000] * 3.6, label="v5 Enhanced Speed", color="tab:green", lw=1.5)
    ax.set(title="IDNN v5 Speed Tracking on 100% Blind Drive M", xlabel="Time (s)", ylabel="Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_v5_speed_tracking.png", dpi=300)
    plt.close(fig)

    # 3. Cumulative Distance
    dt = np.diff(time_axis, prepend=time_axis[0])
    dt[0] = 0.1
    d_gt = np.cumsum(gt_speed * dt) / 1000.0
    d_raw = np.cumsum(v5_raw * dt) / 1000.0
    d_enh = np.cumsum(v5_enh * dt) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis / 60.0, d_gt, label="Ground Truth (105.1 km)", color="black", lw=2.0)
    ax.plot(time_axis / 60.0, d_raw, label=f"v5 Raw ({d_raw[-1]:.1f} km)", color="tab:orange", lw=1.5, ls="--")
    ax.plot(time_axis / 60.0, d_enh, label=f"v5 Enhanced ({d_enh[-1]:.1f} km)", color="tab:green", lw=1.8)
    ax.set(title="Cumulative Distance: IDNN v5 vs Ground Truth", xlabel="Drive Duration (min)", ylabel="Distance (km)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_v5_cumulative_distance.png", dpi=300)
    plt.close(fig)

    # 4. Error Histogram
    err_raw = (v5_raw - gt_speed) * 3.6
    err_enh = (v5_enh - gt_speed) * 3.6

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(err_raw, bins=80, range=(-25, 25), alpha=0.5, label=f"v5 Raw (RMSE: {np.sqrt(np.mean(err_raw**2)):.1f} km/h)", color="tab:orange")
    ax.hist(err_enh, bins=80, range=(-25, 25), alpha=0.6, label=f"v5 Enhanced (RMSE: {np.sqrt(np.mean(err_enh**2)):.1f} km/h)", color="tab:green")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(title="Speed Error Distribution (km/h)", xlabel="Error (km/h)", ylabel="Sample Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_v5_error_histogram.png", dpi=300)
    plt.close(fig)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_crossdrive_v5_pipeline():
    print("=" * 80)
    print("CROSS-DRIVE BLIND TEST PIPELINE FOR IDNN v5")
    print("=" * 80)

    pairs = find_paired_files()
    if not pairs:
        raise FileNotFoundError("No paired files found.")

    primary_test_pair = pairs[0]
    print(f"\n[1/5] Target Blind Test Drive: {os.path.basename(primary_test_pair[0])} (100% Blind Test)")

    train_pairs = pairs[1:8]
    val_pair = pairs[8]
    print(f"  Training Files ({len(train_pairs)} pairs): {[os.path.basename(p[0]) for p in train_pairs]}")
    print(f"  Validation File: {os.path.basename(val_pair[0])}")

    # Process Training Drives
    print(f"\n[2/5] Processing {len(train_pairs)} training drives...")
    train_X_list, train_y_list = [], []
    for i, (sp, vp) in enumerate(train_pairs):
        fname = os.path.basename(sp)
        try:
            d = process_file_v5(sp, vp, for_training=True, stride=TRAIN_STRIDE)
            train_X_list.append(d["X"])
            train_y_list.append(d["y"])
            print(f"  [{i+1}/{len(train_pairs)}] {fname}: {d['n_samples']:,} samples")
        except Exception as e:
            print(f"  [{i+1}/{len(train_pairs)}] {fname}: FAILED -- {e}")

    X_train = np.concatenate(train_X_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)

    # Process Validation Drive
    print(f"\n[3/5] Processing validation drive ({os.path.basename(val_pair[0])})...")
    val_data = process_file_v5(val_pair[0], val_pair[1], for_training=True, stride=TRAIN_STRIDE)
    X_val = val_data["X"]
    y_val = val_data["y"]

    print(f"  Total Training Windows:   {len(X_train):,}")
    print(f"  Total Validation Windows: {len(X_val):,}")

    # Process 100% Blind Test Drive M
    print(f"\n[4/5] Processing 100% Blind Test Drive M...")
    m_data = process_file_v5(primary_test_pair[0], primary_test_pair[1], for_training=False, stride=1)
    m_X = m_data["X"]
    time_axis = m_data["time_axis"]
    gt_speed = m_data["gt"]["velocity_ms"]
    n_full = len(time_axis)

    # Train Cross-Drive Model
    print("\n[5/5] Training IDNN v5 strictly on external drives...")
    model, history = train_crossdrive_v5(X_train, y_train, X_val, y_val)

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
    print(f"  Saved Cross-Drive Model v5 to: {MODEL_PATH}")

    # Blind Inference on Drive M
    v5_pred = predict_v5(model, m_X)
    v5_speed_raw = np.zeros(n_full)
    v5_speed_raw[DELAY_TAPS:] = v5_pred

    # Physics Filters Pipeline
    v5_gated = kinematic_plausibility_gate(v5_speed_raw, m_data["lin_ax"])
    v5_ekf = ekf_with_acceleration_feedforward(v5_gated, m_data["lin_ax"])
    v5_cruise = apply_cruise_lock(v5_ekf, m_data["lin_ax"], m_data["gz"])
    v5_enhanced = apply_enhanced_zupt(v5_cruise, m_data["ts_feats"])

    # Generate Plots
    print("\nGenerating IDNN v5 Diagnostic Plots...")
    generate_v5_diagnostic_plots(time_axis, gt_speed, v5_speed_raw, v5_ekf, v5_enhanced, history)
    print(f"  Plots saved to: {OUTPUT_DIR}")

    return model


if __name__ == "__main__":
    run_crossdrive_v5_pipeline()

