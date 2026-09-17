"""
LSTM Pipeline — Multi-file training + jerk features + Kalman filter.

Combines best of v1 (temporal patterns) + v2 (window statistics)
with LSTM for sequential modeling and Kalman filter post-processing.

Outputs:
  - results/lstm_model.pth
  - results/plots/lstm/  (comparison plots: Baseline, v1, v2, LSTM, LSTM+KF)
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
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.lstm_model import LSTMSpeedPredictor, kalman_filter_speed
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
# CONFIG
# ============================================================

OUTPUT_DIR = "results/plots/lstm"
MODEL_PATH = "results/lstm_model.pth"
V1_PATH = "results/idnn_model.pth"
V2_PATH = "results/idnn_v2_model.pth"

SEQ_LEN = 30          # 3 seconds at 10 Hz
N_SEQ_FEAT = 20       # per-timestep features
N_STAT_FEAT = 26      # window statistics
SAMPLE_RATE = 10.0

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

EPOCHS = 100
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TRAIN_STRIDE = 3      # stride for training file sequences
MAX_EXTRA_FILES = 5    # additional files beyond M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PER-TIMESTEP FEATURE EXTRACTION  (20 features)
# ============================================================
# 0-2:   lin_ax, lin_ay, lin_az  (gravity-subtracted)
# 3-5:   gx, gy, gz             (gyroscope)
# 6-8:   mag_x, mag_y, mag_z    (magnetometer)
# 9:     sin(orient_yaw)
# 10:    cos(orient_yaw)
# 11:    orient_pitch
# 12:    orient_roll
# 13:    gravity_magnitude
# 14-16: jerk_ax, jerk_ay, jerk_az   (d(lin_accel)/dt)
# 17-19: jerk_gx, jerk_gy, jerk_gz   (d(gyro)/dt)

def _interp(arr, n):
    """Fill NaNs via interpolation, return float64 array of length n."""
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    s = pd.Series(arr[:n]).interpolate().ffill().bfill()
    return s.to_numpy(dtype=np.float64)


def _find_col(df, keywords, exclude=None):
    """Find column whose uppercase name contains ALL keywords."""
    for col in df.columns:
        cu = col.strip().upper()
        if all(k in cu for k in keywords):
            if exclude and any(e in cu for e in exclude):
                continue
            return col
    return None


def extract_timestep_features(s_df, s_cols, n):
    """Extract 20 per-timestep features from smartphone data."""

    # --- Accelerometer + gravity ---
    ax = _interp(numeric_series(s_df, s_cols["ax"]), n)
    ay = _interp(numeric_series(s_df, s_cols["ay"]), n)
    az = _interp(numeric_series(s_df, s_cols["az"]), n)

    grav_x = _interp(numeric_series(s_df, s_cols.get("grav_x")), n)
    grav_y = _interp(numeric_series(s_df, s_cols.get("grav_y")), n)
    grav_z = _interp(numeric_series(s_df, s_cols.get("grav_z")), n)

    # Unit check (g → m/s²)
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    if 0.5 <= np.nanmedian(mag) <= 2.0:
        ax *= G; ay *= G; az *= G
        grav_x *= G; grav_y *= G; grav_z *= G

    lin_ax = ax - grav_x
    lin_ay = ay - grav_y
    lin_az = az - grav_z

    # --- Gyroscope ---
    gx = _interp(numeric_series(s_df, s_cols.get("gx")), n)
    gy = _interp(numeric_series(s_df, s_cols.get("gy")), n)
    gz = _interp(numeric_series(s_df, s_cols.get("gz")), n)

    # --- Magnetometer ---
    mx = _interp(numeric_series(s_df, _find_col(s_df, ["MAGNETIC", "X"])), n)
    my = _interp(numeric_series(s_df, _find_col(s_df, ["MAGNETIC", "Y"])), n)
    mz = _interp(numeric_series(s_df, _find_col(s_df, ["MAGNETIC", "Z"])), n)

    # --- Orientation ---
    oy = _interp(numeric_series(s_df, _find_col(s_df, ["ORIENTATION", "YAW"])), n)
    op = _interp(numeric_series(s_df, _find_col(s_df, ["ORIENTATION", "PITCH"])), n)
    orl = _interp(numeric_series(s_df, _find_col(s_df, ["ORIENTATION", "ROLL"])), n)
    oy_rad = np.deg2rad(oy)

    # --- Gravity magnitude ---
    grav_mag = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2)

    # --- Jerk (d/dt) ---
    dt = 1.0 / SAMPLE_RATE
    jerk_ax = np.gradient(lin_ax, dt)
    jerk_ay = np.gradient(lin_ay, dt)
    jerk_az = np.gradient(lin_az, dt)
    jerk_gx = np.gradient(gx, dt)
    jerk_gy = np.gradient(gy, dt)
    jerk_gz = np.gradient(gz, dt)

    features = np.column_stack([
        lin_ax, lin_ay, lin_az,         # 0-2
        gx, gy, gz,                     # 3-5
        mx, my, mz,                     # 6-8
        np.sin(oy_rad), np.cos(oy_rad), # 9-10
        op, orl,                        # 11-12
        grav_mag,                       # 13
        jerk_ax, jerk_ay, jerk_az,      # 14-16
        jerk_gx, jerk_gy, jerk_gz,      # 17-19
    ]).astype(np.float32)

    assert features.shape == (n, N_SEQ_FEAT), f"Expected ({n}, {N_SEQ_FEAT}), got {features.shape}"
    return features


# ============================================================
# WINDOW STATISTICS  (26 features)
# ============================================================

def compute_window_stats(timestep_feats, seq_len):
    """
    Compute v2-style window statistics for every sliding window.

    Returns: (N - seq_len + 1, 26)
    """
    accel = timestep_feats[:, :3]
    gyro = timestep_feats[:, 3:6]
    grav = timestep_feats[:, 13]

    # Sliding windows
    a_win = sliding_window_view(accel, window_shape=seq_len, axis=0)
    a_win = np.moveaxis(a_win, -1, 1)   # (W, S, 3)

    g_win = sliding_window_view(gyro, window_shape=seq_len, axis=0)
    g_win = np.moveaxis(g_win, -1, 1)

    gv_win = sliding_window_view(grav, window_shape=seq_len, axis=0)

    # Statistics
    a_mean = np.mean(a_win, axis=1)
    a_std = np.std(a_win, axis=1)
    a_med = np.median(a_win, axis=1)
    g_mean = np.mean(g_win, axis=1)
    g_std = np.std(g_win, axis=1)

    # Spectral bands
    freqs = np.fft.rfftfreq(seq_len, d=1.0 / SAMPLE_RATE)
    b1 = (freqs >= 0) & (freqs < 1.0)
    b2 = (freqs >= 1.0) & (freqs < 3.0)
    b3 = (freqs >= 3.0)

    fft_a = np.fft.rfft(a_win, axis=1)
    pwr = np.abs(fft_a) ** 2
    e1 = np.sum(pwr[:, b1, :], axis=1)
    e2 = np.sum(pwr[:, b2, :], axis=1)
    e3 = np.sum(pwr[:, b3, :], axis=1)

    gv_m = np.mean(gv_win, axis=1).reshape(-1, 1)
    gv_s = np.std(gv_win, axis=1).reshape(-1, 1)

    return np.hstack([
        a_mean, a_std, a_med,     # 9
        g_mean, g_std,            # 6
        e1, e2, e3,               # 9
        gv_m, gv_s,               # 2
    ]).astype(np.float32)         # total 26


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(s_path, v_path, for_training=False, stride=1):
    """
    Load & process one S/V file pair.

    Returns dict with:
        x_seq:    (M, SEQ_LEN, 20)
        x_stat:   (M, 26)
        y:        (M,)
        time_axis, gt, inertial  (full-length, only if not for_training)
    """
    s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1")
    v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1")

    s_cols = detect_smartphone_columns(s_df)
    v_cols = detect_vehicle_columns(v_df)

    # Time axis from vehicle
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

    # Ground truth speed
    gt = compute_vehicle_ground_truth(v_df, v_cols, time_axis)
    target_speed = gt["velocity_ms"]   # (n,)

    # Per-timestep features
    ts_feats = extract_timestep_features(s_df, s_cols, n)

    # Window statistics
    win_stats = compute_window_stats(ts_feats, SEQ_LEN)  # (n-S+1, 26)

    # Create sequences  (using sliding_window_view on ts_feats)
    ts_win = sliding_window_view(ts_feats, window_shape=SEQ_LEN, axis=0)
    ts_win = np.moveaxis(ts_win, -1, 1)  # (n-S+1, SEQ_LEN, 20)

    # Targets aligned to end of each window
    y = target_speed[SEQ_LEN - 1:]       # (n-S+1,)

    n_win = ts_win.shape[0]

    # Apply stride
    idx = np.arange(0, n_win, stride)
    x_seq = ts_win[idx]
    x_stat = win_stats[idx]
    y_out = y[idx]

    result = {
        "x_seq": x_seq.astype(np.float32),
        "x_stat": x_stat.astype(np.float32),
        "y": y_out.astype(np.float32),
        "n_samples": len(idx),
    }

    if not for_training:
        # Baseline inertial (need raw accel + gravity)
        ax = _interp(numeric_series(s_df, s_cols["ax"]), n)
        ay = _interp(numeric_series(s_df, s_cols["ay"]), n)
        az = _interp(numeric_series(s_df, s_cols["az"]), n)
        gx = _interp(numeric_series(s_df, s_cols.get("grav_x")), n)
        gy = _interp(numeric_series(s_df, s_cols.get("grav_y")), n)
        gz = _interp(numeric_series(s_df, s_cols.get("grav_z")), n)
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        if 0.5 <= np.nanmedian(mag) <= 2.0:
            ax *= G; ay *= G; az *= G; gx *= G; gy *= G; gz *= G

        inertial = compute_baseline_inertial(ax, ay, az, gx, gy, gz, time_axis)
        result["time_axis"] = time_axis
        result["gt"] = gt
        result["inertial"] = inertial
        result["ts_feats_full"] = ts_feats       # for stride-1 prediction
        result["win_stats_full"] = win_stats

    return result


# ============================================================
# TRAINING
# ============================================================

def train_lstm(X_seq_train, X_stat_train, y_train,
               X_seq_val, X_stat_val, y_val):
    """Train LSTM model, return model + history."""

    model = LSTMSpeedPredictor(
        n_seq_features=N_SEQ_FEAT,
        n_stat_features=N_STAT_FEAT,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  LSTM Architecture:")
    print(f"    Seq input: {N_SEQ_FEAT} x {SEQ_LEN} timesteps -> LSTM({HIDDEN_SIZE}, {NUM_LAYERS} layers)")
    print(f"    Stat input: {N_STAT_FEAT} window statistics")
    print(f"    MLP head: {HIDDEN_SIZE}+{N_STAT_FEAT} -> 128 -> 64 -> 1")
    print(f"    Parameters: {n_params:,}")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"    Device: {DEVICE} ({dev_name})")

    # Normalize
    seq_mean = X_seq_train.mean(axis=(0, 1))
    seq_std = X_seq_train.std(axis=(0, 1)) + 1e-8
    stat_mean = X_stat_train.mean(axis=0)
    stat_std = X_stat_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    def norm_seq(x): return (x - seq_mean) / seq_std
    def norm_stat(x): return (x - stat_mean) / stat_std
    def norm_y(y): return (y - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(norm_seq(X_seq_train)).float(),
        torch.from_numpy(norm_stat(X_stat_train)).float(),
        torch.from_numpy(norm_y(y_train)).float().unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(norm_seq(X_seq_val)).float(),
        torch.from_numpy(norm_stat(X_stat_val)).float(),
        torch.from_numpy(norm_y(y_val)).float().unsqueeze(1),
    )

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=use_cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    print(f"\n  Training {EPOCHS} epochs  (train={len(X_seq_train):,}  val={len(X_seq_val):,})")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_total, t_n = 0.0, 0
        for xseq, xstat, yb in train_loader:
            xseq = xseq.to(DEVICE)
            xstat = xstat.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xseq, xstat)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_total += loss.item() * len(yb)
            t_n += len(yb)
        train_loss = t_total / t_n

        model.eval()
        v_total, v_n = 0.0, 0
        with torch.no_grad():
            for xseq, xstat, yb in val_loader:
                xseq = xseq.to(DEVICE)
                xstat = xstat.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xseq, xstat)
                loss = criterion(pred, yb)
                v_total += loss.item() * len(yb)
                v_n += len(yb)
        val_loss = v_total / v_n

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  best={best_val:.6f}  [{time.time()-t0:.1f}s]")

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s.  Best val: {best_val:.6f}")

    # Store normalization
    model.seq_mean = seq_mean
    model.seq_std = seq_std
    model.stat_mean = stat_mean
    model.stat_std = stat_std
    model.y_mean = y_mean
    model.y_std = y_std

    return model, history


def predict_lstm(model, x_seq, x_stat, batch=4096):
    """Run LSTM inference."""
    model.eval()
    x_seq_n = (x_seq - model.seq_mean) / model.seq_std
    x_stat_n = (x_stat - model.stat_mean) / model.stat_std

    preds = []
    for i in range(0, len(x_seq_n), batch):
        xs = torch.from_numpy(x_seq_n[i:i+batch]).float().to(DEVICE)
        xst = torch.from_numpy(x_stat_n[i:i+batch]).float().to(DEVICE)
        with torch.no_grad():
            p = model(xs, xst).cpu().numpy().flatten()
        preds.append(p)

    pred = np.concatenate(preds)
    pred = pred * model.y_std + model.y_mean
    return np.maximum(pred, 0.0)


# ============================================================
# V1 / V2 LOADING
# ============================================================

def load_v1_predictions(ts_feats, time_axis):
    """Load IDNN v1 and predict."""
    if not os.path.exists(V1_PATH):
        return None
    try:
        from src.idnn_model import IDNN
        from src.idnn_pipeline import create_delay_windows, predict as pred_v1

        ckpt = torch.load(V1_PATH, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        model = IDNN(cfg["n_features"], cfg["delay_taps"], cfg["n_outputs"],
                      cfg["hidden_sizes"], cfg["dropout"])
        model.load_state_dict(ckpt["model_state"])
        model.X_mean = ckpt["X_mean"]; model.X_std = ckpt["X_std"]
        model.y_mean = ckpt["y_mean"]; model.y_std = ckpt["y_std"]

        v1_raw = ts_feats[:, :6]  # first 6 = lin_accel + gyro
        X, _ = create_delay_windows(v1_raw, np.zeros(len(v1_raw)), cfg["delay_taps"])
        speed = pred_v1(model, X)

        full_speed = np.zeros(len(time_axis))
        full_speed[cfg["delay_taps"]:] = speed
        return full_speed
    except Exception as e:
        print(f"  V1 load failed: {e}")
        return None


def load_v2_predictions(s_path, v_path, time_axis):
    """Load IDNN v2 and predict using its exact features."""
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
                        cfg["hidden_sizes"], cfg["dropout"])
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
# DISTANCE
# ============================================================

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
# PLOTS
# ============================================================

def generate_plots(
    time_axis, gt, inertial,
    v1_speed, v2_speed,
    lstm_speed, lstm_kf_speed,
    lstm_dist, lstm_kf_dist,
    history, split_idx,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    gt_s = gt["velocity_ms"]
    gt_d = gt["gt_distance"]
    bl_s = inertial["speed"]
    bl_d = inertial["cumulative_distance"]

    C = {"gt": "tab:blue", "bl": "tab:orange", "v1": "tab:red",
         "v2": "tab:purple", "lstm": "tab:green", "kf": "darkgreen"}
    train_end, val_end = split_idx

    # 1. Training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="Train", lw=1.5)
    ax.plot(ep, history["val_loss"], label="Val", lw=1.5)
    ax.set(title="LSTM Training Loss", xlabel="Epoch", ylabel="MSE (normalized)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Speed: all models
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(time_axis, gt_s*3.6, lw=1.5, alpha=0.8, label="Vehicle GT", color=C["gt"])
    ax.plot(time_axis, bl_s*3.6, lw=0.6, alpha=0.2, label="Baseline INS", color=C["bl"])
    if v1_speed is not None:
        ax.plot(time_axis, v1_speed*3.6, lw=0.8, alpha=0.4, label="IDNN v1", color=C["v1"])
    if v2_speed is not None:
        ax.plot(time_axis, v2_speed*3.6, lw=0.8, alpha=0.4, label="IDNN v2", color=C["v2"])
    ax.plot(time_axis, lstm_speed*3.6, lw=1, alpha=0.7, label="LSTM", color=C["lstm"])
    ax.plot(time_axis, lstm_kf_speed*3.6, lw=1.2, alpha=0.9, label="LSTM+KF", color=C["kf"], ls="--")
    ax.axvline(time_axis[train_end], ls=":", color="gray", alpha=0.5)
    ax.axvline(time_axis[val_end], ls=":", color="black", alpha=0.5)
    ax.set(title="Speed: All Models", xlabel="Time (s)", ylabel="km/h")
    ax.set_ylim(-5, max(np.max(gt_s*3.6)*1.3, 120))
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_speed_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. LSTM+KF vs GT zoomed
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, gt_s*3.6, lw=1.5, label="Vehicle GT", color=C["gt"])
    ax.plot(time_axis, lstm_kf_speed*3.6, lw=1.2, alpha=0.85, label="LSTM+KF", color=C["kf"])
    ax.axvline(time_axis[val_end], ls="--", color="black", alpha=0.5, label="Test ->")
    ax.set(title="LSTM+KF Speed vs Ground Truth", xlabel="Time (s)", ylabel="km/h")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_lstm_kf_vs_gt.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Cumulative distance
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, gt_d, lw=2, label="Vehicle GT", color=C["gt"])
    ax.plot(time_axis, bl_d, lw=1, alpha=0.3, label="Baseline", color=C["bl"])
    if v1_speed is not None:
        v1_d = speed_to_distance(v1_speed, time_axis)
        ax.plot(time_axis, v1_d, lw=1, alpha=0.5, label="IDNN v1", color=C["v1"])
    if v2_speed is not None:
        v2_d = speed_to_distance(v2_speed, time_axis)
        ax.plot(time_axis, v2_d, lw=1, alpha=0.5, label="IDNN v2", color=C["v2"])
    ax.plot(time_axis, lstm_dist, lw=1.2, alpha=0.7, label="LSTM", color=C["lstm"])
    ax.plot(time_axis, lstm_kf_dist, lw=1.5, alpha=0.9, label="LSTM+KF", color=C["kf"], ls="--")
    ax.set(title="Cumulative Distance: All Models", xlabel="Time (s)", ylabel="m")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_cumulative_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Distance error
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_axis, np.abs(bl_d - gt_d), lw=0.8, alpha=0.3, label="Baseline", color=C["bl"])
    if v1_speed is not None:
        ax.plot(time_axis, np.abs(v1_d - gt_d), lw=1, alpha=0.5, label="v1", color=C["v1"])
    if v2_speed is not None:
        ax.plot(time_axis, np.abs(v2_d - gt_d), lw=1, alpha=0.5, label="v2", color=C["v2"])
    ax.plot(time_axis, np.abs(lstm_dist - gt_d), lw=1.2, alpha=0.7, label="LSTM", color=C["lstm"])
    ax.plot(time_axis, np.abs(lstm_kf_dist - gt_d), lw=1.5, alpha=0.9, label="LSTM+KF", color=C["kf"], ls="--")
    ax.set(title="Distance Error Over Time", xlabel="Time (s)", ylabel="m")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_distance_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6. Drift %
    _, bl_dr = calculate_scalar_drift(bl_d, gt_d)
    _, lstm_dr = calculate_scalar_drift(lstm_dist, gt_d)
    _, kf_dr = calculate_scalar_drift(lstm_kf_dist, gt_d)

    fig, ax = plt.subplots(figsize=(14, 6))
    valid = np.isfinite(bl_dr)
    if np.any(valid):
        ax.plot(gt_d[valid], bl_dr[valid], lw=0.8, alpha=0.3, label="Baseline", color=C["bl"])
    if v1_speed is not None:
        _, v1_dr = calculate_scalar_drift(v1_d, gt_d)
        v = np.isfinite(v1_dr)
        if np.any(v):
            ax.plot(gt_d[v], v1_dr[v], lw=1, alpha=0.5, label="v1", color=C["v1"])
    if v2_speed is not None:
        _, v2_dr = calculate_scalar_drift(v2_d, gt_d)
        v = np.isfinite(v2_dr)
        if np.any(v):
            ax.plot(gt_d[v], v2_dr[v], lw=1, alpha=0.5, label="v2", color=C["v2"])
    v = np.isfinite(lstm_dr)
    if np.any(v):
        ax.plot(gt_d[v], lstm_dr[v], lw=1.2, alpha=0.7, label="LSTM", color=C["lstm"])
    v = np.isfinite(kf_dr)
    if np.any(v):
        ax.plot(gt_d[v], kf_dr[v], lw=1.5, alpha=0.9, label="LSTM+KF", color=C["kf"], ls="--")
    ax.axhline(10, ls="--", lw=1.5, color="red", label="10% Threshold")
    ax.set_ylim(0, min(50, max(np.nanmax(kf_dr[np.isfinite(kf_dr)])*1.5, 15) if np.any(np.isfinite(kf_dr)) else 30))
    ax.set(title="Drift % vs Distance", xlabel="Distance (m)", ylabel="Drift (%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_drift_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 7. Speed error histogram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name, sp, col in [
        (axes[0], "Baseline", bl_s, C["bl"]),
        (axes[1], "LSTM", lstm_speed, C["lstm"]),
        (axes[2], "LSTM+KF", lstm_kf_speed, C["kf"]),
    ]:
        err = (sp - gt_s) * 3.6
        rmse = np.sqrt(np.mean(err**2))
        ax.hist(err, bins=100, alpha=0.7, color=col)
        ax.set_title(f"{name}  (RMSE={rmse:.1f} km/h)")
        ax.set_xlabel("Speed Error (km/h)"); ax.grid(True, alpha=0.3)
    fig.suptitle("Speed Error Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/07_speed_error_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 8. Summary bar chart
    models = ["Baseline", "IDNN v1", "IDNN v2", "LSTM", "LSTM+KF"]
    speeds = [bl_s, v1_speed, v2_speed, lstm_speed, lstm_kf_speed]
    dists = [bl_d, v1_d if v1_speed is not None else None,
             v2_d if v2_speed is not None else None, lstm_dist, lstm_kf_dist]
    colors = [C["bl"], C["v1"], C["v2"], C["lstm"], C["kf"]]

    rmses, fdrifts = [], []
    for sp in speeds:
        if sp is not None:
            rmses.append(np.sqrt(np.mean((sp - gt_s)**2)) * 3.6)
        else:
            rmses.append(0)
    for d in dists:
        if d is not None:
            fdrifts.append(abs(d[-1] - gt_d[-1]) / gt_d[-1] * 100)
        else:
            fdrifts.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(models, rmses, color=colors)
    axes[0].set_title("Speed RMSE (Full Data)"); axes[0].set_ylabel("km/h")
    for b, v in zip(bars, rmses):
        if v > 0:
            axes[0].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                         f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    bars = axes[1].bar(models, fdrifts, color=colors)
    axes[1].set_title("Final Drift % (Full Data)"); axes[1].set_ylabel("%")
    axes[1].axhline(10, ls="--", color="red", lw=1.5)
    for b, v in zip(bars, fdrifts):
        if v > 0:
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                         f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("All Models Summary", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_summary_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_drift_table_image(rows, headers):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.0)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    cmap = {"Baseline": "#FFF2CC", "v1": "#FCE4EC", "v2": "#E8D5F0",
            "LSTM": "#E8F5E9", "KF": "#C8E6C9"}
    for i, row in enumerate(rows, 1):
        c = "#FFFFFF"
        for k, v in cmap.items():
            if k in row[0]:
                c = v; break
        for j in range(len(headers)):
            table[i, j].set_facecolor(c)
    ax.set_title("Comprehensive Drift Table — All Models",
                 fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/09_drift_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def run_lstm_pipeline():

    print("=" * 70)
    print("LSTM PIPELINE — Multi-file + Jerk + Kalman Filter")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Find files
    # ----------------------------------------------------------
    print("\n[1/8] Finding data files...")
    pairs = find_paired_files()
    if not pairs:
        raise FileNotFoundError("No paired files found.")

    print(f"  Found {len(pairs)} file pairs.")

    # File 0 is M (primary test file)
    primary = pairs[0]
    extra_pairs = pairs[1:1+MAX_EXTRA_FILES]
    print(f"  Primary (test): {os.path.basename(primary[0])}")
    print(f"  Extra training files: {len(extra_pairs)}")

    # ----------------------------------------------------------
    # 2. Process primary file (file M)
    # ----------------------------------------------------------
    print("\n[2/8] Processing primary file M...")
    m_data = process_file(primary[0], primary[1], for_training=False, stride=1)
    print(f"  Samples: {m_data['n_samples']:,}")

    # ----------------------------------------------------------
    # 3. Process extra training files
    # ----------------------------------------------------------
    print(f"\n[3/8] Processing {len(extra_pairs)} extra files...")
    extra_seqs, extra_stats, extra_ys = [], [], []

    for i, (sp, vp) in enumerate(extra_pairs):
        fname = os.path.basename(sp)
        try:
            d = process_file(sp, vp, for_training=True, stride=TRAIN_STRIDE)
            extra_seqs.append(d["x_seq"])
            extra_stats.append(d["x_stat"])
            extra_ys.append(d["y"])
            print(f"  [{i+1}] {fname}: {d['n_samples']:,} sequences")
        except Exception as e:
            print(f"  [{i+1}] {fname}: FAILED — {e}")
            traceback.print_exc()

    # ----------------------------------------------------------
    # 4. Build train / val / test splits
    # ----------------------------------------------------------
    print("\n[4/8] Building data splits...")

    n_m = m_data["n_samples"]
    train_end = int(n_m * TRAIN_RATIO)
    val_end = int(n_m * (TRAIN_RATIO + VAL_RATIO))

    # M train portion
    m_seq = m_data["x_seq"]
    m_stat = m_data["x_stat"]
    m_y = m_data["y"]

    # Subsample M train for diversity (stride 3 to match extra files)
    m_train_idx = np.arange(0, train_end, TRAIN_STRIDE)
    m_train_seq = m_seq[m_train_idx]
    m_train_stat = m_stat[m_train_idx]
    m_train_y = m_y[m_train_idx]

    # Combine all training data
    all_train_seq = [m_train_seq] + extra_seqs
    all_train_stat = [m_train_stat] + extra_stats
    all_train_y = [m_train_y] + extra_ys

    X_seq_train = np.concatenate(all_train_seq, axis=0)
    X_stat_train = np.concatenate(all_train_stat, axis=0)
    y_train = np.concatenate(all_train_y, axis=0)

    # Val/test from M only
    X_seq_val = m_seq[train_end:val_end]
    X_stat_val = m_stat[train_end:val_end]
    y_val = m_y[train_end:val_end]

    X_seq_test = m_seq[val_end:]
    X_stat_test = m_stat[val_end:]
    y_test = m_y[val_end:]

    print(f"  Train: {len(X_seq_train):,}  (M: {len(m_train_seq):,}  Extra: {len(X_seq_train)-len(m_train_seq):,})")
    print(f"  Val:   {len(X_seq_val):,}")
    print(f"  Test:  {len(X_seq_test):,}")

    # Full-length indices for M
    full_train_end = train_end + SEQ_LEN - 1
    full_val_end = val_end + SEQ_LEN - 1

    # ----------------------------------------------------------
    # 5. Train
    # ----------------------------------------------------------
    print("\n[5/8] Training LSTM...")
    model, history = train_lstm(
        X_seq_train, X_stat_train, y_train,
        X_seq_val, X_stat_val, y_val,
    )

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "seq_mean": model.seq_mean, "seq_std": model.seq_std,
        "stat_mean": model.stat_mean, "stat_std": model.stat_std,
        "y_mean": model.y_mean, "y_std": model.y_std,
        "config": {
            "n_seq_features": N_SEQ_FEAT,
            "n_stat_features": N_STAT_FEAT,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
        },
    }, MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

    # ----------------------------------------------------------
    # 6. Predict on M (full stride-1)
    # ----------------------------------------------------------
    print("\n[6/8] Evaluating on file M...")

    # Full-stride prediction
    lstm_pred = predict_lstm(model, m_seq, m_stat)

    # Pad to full length
    time_axis = m_data["time_axis"]
    n_full = len(time_axis)
    lstm_speed = np.zeros(n_full)
    lstm_speed[SEQ_LEN-1:] = lstm_pred

    # Kalman filter
    lstm_kf_speed = kalman_filter_speed(lstm_speed, process_noise=0.5, measurement_noise=4.0)

    # Distances
    lstm_dist = speed_to_distance(lstm_speed, time_axis)
    lstm_kf_dist = speed_to_distance(lstm_kf_speed, time_axis)

    gt = m_data["gt"]
    inertial = m_data["inertial"]
    gt_speed = gt["velocity_ms"]
    gt_dist = gt["gt_distance"]
    bl_speed = inertial["speed"]

    # Load v1/v2
    print("\n  Loading v1/v2 for comparison...")
    ts_full = m_data["ts_feats_full"]
    v1_speed = load_v1_predictions(ts_full, time_axis)
    v2_speed = load_v2_predictions(primary[0], primary[1], time_axis)

    if v1_speed is not None:
        print(f"  V1 loaded.")
    else:
        print(f"  V1 not available.")
    if v2_speed is not None:
        print(f"  V2 loaded.")
    else:
        print(f"  V2 not available.")

    # ----------------------------------------------------------
    # 7. Comprehensive drift table
    # ----------------------------------------------------------
    print("\n[7/8] Computing drift metrics...")

    dt = 0.1

    def d(sp, start, end):
        return drift_on_split(sp[start:end], gt_speed[start:end], dt)

    # Models
    models_info = [
        ("Baseline INS", bl_speed),
        ("IDNN v1", v1_speed),
        ("IDNN v2", v2_speed),
        ("LSTM", lstm_speed),
        ("LSTM + KF", lstm_kf_speed),
    ]

    headers = ["Model", "Full (100%)", "Train (70%)", "Val (15%)",
               "Test (15%)", "Test Speed RMSE"]
    rows = []

    print(f"\n{'='*85}")
    print(f"{'Model':<16s} {'Full':<13s} {'Train':<13s} {'Val':<13s} {'Test':<13s} {'Test RMSE':>10s}")
    print(f"{'-'*85}")

    for name, sp in models_info:
        if sp is None:
            rows.append([name, "N/A", "N/A", "N/A", "N/A", "N/A"])
            print(f"{name:<16s} {'N/A':>10s}     {'N/A':>10s}     {'N/A':>10s}     {'N/A':>10s}     {'N/A':>10s}")
            continue

        full = d(sp, 0, n_full)
        train = d(sp, 0, full_train_end)
        val = d(sp, full_train_end, full_val_end)
        test = d(sp, full_val_end, n_full)
        rmse = np.sqrt(np.mean((sp[full_val_end:] - gt_speed[full_val_end:])**2)) * 3.6

        rows.append([name, f"{full:.1f}%", f"{train:.1f}%", f"{val:.1f}%",
                      f"{test:.1f}%", f"{rmse:.1f} km/h"])
        print(f"{name:<16s} {full:>10.1f}%    {train:>10.1f}%    {val:>10.1f}%    {test:>10.1f}%    {rmse:>8.1f} km/h")

    print(f"{'='*85}")

    # Check if LSTM+KF passes
    kf_test = drift_on_split(lstm_kf_speed[full_val_end:], gt_speed[full_val_end:], dt)
    print(f"\nLSTM+KF test drift: {kf_test:.1f}%  -> {'PASS [OK]' if kf_test <= 10 else 'FAIL [X]'}")

    # ----------------------------------------------------------
    # 8. Generate plots
    # ----------------------------------------------------------
    print("\n[8/8] Generating plots...")

    generate_plots(
        time_axis, gt, inertial,
        v1_speed, v2_speed,
        lstm_speed, lstm_kf_speed,
        lstm_dist, lstm_kf_dist,
        history, (full_train_end, full_val_end),
    )

    save_drift_table_image(rows, headers)

    print(f"\nSaved 9 plots to: {OUTPUT_DIR}")

    print(f"\n{'='*70}")
    print("LSTM PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_lstm_pipeline()

