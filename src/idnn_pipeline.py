"""
IDNN Pipeline — Train, evaluate, and compare against baseline.

This script:
1. Loads paired smartphone + vehicle data (reusing baseline_graphs logic)
2. Prepares time-delayed input windows for the IDNN
3. Trains the IDNN to predict vehicle speed from smartphone sensors
4. Evaluates and computes improved distance / drift metrics
5. Generates comparison plots (baseline vs IDNN) in results/plots/idnn/
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.idnn_model import IDNN
from src.baseline_graphs import (
    find_paired_files,
    detect_smartphone_columns,
    detect_vehicle_columns,
    numeric_series,
    project_gnss_to_local_xy,
    compute_baseline_inertial,
    compute_vehicle_ground_truth,
    calculate_scalar_drift,
    butter_lowpass,
    G, LOWPASS_CUTOFF_HZ,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "results/plots/idnn"
MODEL_SAVE_PATH = "results/idnn_model.pth"

# IDNN hyperparameters
DELAY_TAPS = 20          # 20 past samples = 2 seconds at 10Hz
N_FEATURES = 6           # lin_ax, lin_ay, lin_az, gx, gy, gz
N_OUTPUTS = 1            # predict vehicle speed (m/s)
HIDDEN_SIZES = (128, 64, 32)
DROPOUT = 0.2

# Training
EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  (remainder)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(s_df, v_df, s_cols, v_cols, time_axis):
    """
    Extract and align smartphone sensor features and vehicle targets.

    Returns:
        features: (N, 6) array — [lin_ax, lin_ay, lin_az, gx, gy, gz]
        target_speed: (N,) array — vehicle speed in m/s
        inertial: dict from baseline computation
        gt: dict from vehicle ground truth computation
    """

    # ---- Smartphone accel ----
    ax = numeric_series(s_df, s_cols["ax"]).to_numpy().astype(float)
    ay = numeric_series(s_df, s_cols["ay"]).to_numpy().astype(float)
    az = numeric_series(s_df, s_cols["az"]).to_numpy().astype(float)

    grav_x = numeric_series(s_df, s_cols["grav_x"])
    grav_y = numeric_series(s_df, s_cols["grav_y"])
    grav_z = numeric_series(s_df, s_cols["grav_z"])

    gx = numeric_series(s_df, s_cols["gx"])
    gy = numeric_series(s_df, s_cols["gy"])
    gz = numeric_series(s_df, s_cols["gz"])

    # Unit detection
    magnitude = np.sqrt(
        np.nan_to_num(ax)**2 + np.nan_to_num(ay)**2 + np.nan_to_num(az)**2
    )
    if 0.5 <= np.nanmedian(magnitude) <= 2.0:
        ax *= G; ay *= G; az *= G
        if grav_x is not None:
            grav_x = grav_x * G; grav_y = grav_y * G; grav_z = grav_z * G

    # Interpolate
    ax = pd.Series(ax).interpolate().ffill().bfill().to_numpy()
    ay = pd.Series(ay).interpolate().ffill().bfill().to_numpy()
    az = pd.Series(az).interpolate().ffill().bfill().to_numpy()

    if grav_x is not None:
        grav_x = grav_x.interpolate().ffill().bfill().to_numpy().astype(float)
        grav_y = grav_y.interpolate().ffill().bfill().to_numpy().astype(float)
        grav_z = grav_z.interpolate().ffill().bfill().to_numpy().astype(float)

    if gx is not None:
        gx = gx.interpolate().ffill().bfill().to_numpy().astype(float)
    else:
        gx = np.zeros(len(ax))
    if gy is not None:
        gy = gy.interpolate().ffill().bfill().to_numpy().astype(float)
    else:
        gy = np.zeros(len(ax))
    if gz is not None:
        gz = gz.interpolate().ffill().bfill().to_numpy().astype(float)
    else:
        gz = np.zeros(len(ax))

    # ---- Compute baseline inertial ----
    inertial = compute_baseline_inertial(
        ax.copy(), ay.copy(), az.copy(),
        grav_x.copy() if grav_x is not None else None,
        grav_y.copy() if grav_y is not None else None,
        grav_z.copy() if grav_z is not None else None,
        time_axis,
    )

    # ---- Vehicle ground truth ----
    gt = compute_vehicle_ground_truth(v_df, v_cols, time_axis)

    # ---- Build feature matrix ----
    # Use linear acceleration (gravity subtracted, filtered)
    lin_ax = inertial["lin_ax"]
    lin_ay = inertial["lin_ay"]
    lin_az = inertial["lin_az"]

    features = np.column_stack([lin_ax, lin_ay, lin_az, gx, gy, gz])

    # ---- Target: vehicle speed ----
    target_speed = gt["velocity_ms"]

    return features, target_speed, inertial, gt


def create_delay_windows(features, target, delay_taps):
    """
    Create time-delayed input windows.

    For each timestep t (where t >= delay_taps), creates:
        X[t] = [features[t], features[t-1], ..., features[t-delay_taps]]
        y[t] = target[t]

    Returns:
        X: (N - delay_taps, (delay_taps+1) * n_features)
        y: (N - delay_taps,)
    """

    n_samples, n_features = features.shape
    n_valid = n_samples - delay_taps

    X = np.zeros((n_valid, (delay_taps + 1) * n_features), dtype=np.float32)

    for i in range(n_valid):
        # Window: [features[i+delay_taps], features[i+delay_taps-1], ..., features[i]]
        # Most recent first
        t = i + delay_taps
        window = features[t - delay_taps: t + 1][::-1]  # reverse so current is first
        X[i] = window.flatten()

    y = target[delay_taps:].astype(np.float32)

    return X, y


# ============================================================
# TRAINING
# ============================================================

def train_idnn(X_train, y_train, X_val, y_val):
    """
    Train the IDNN model.

    Returns:
        model: trained IDNN
        history: dict with train_loss and val_loss per epoch
    """

    model = IDNN(
        n_features=N_FEATURES,
        delay_taps=DELAY_TAPS,
        n_outputs=N_OUTPUTS,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"\nIDNN Architecture:")
    print(f"  Input dim: {N_FEATURES} × {DELAY_TAPS + 1} = {N_FEATURES * (DELAY_TAPS + 1)}")
    print(f"  Hidden: {HIDDEN_SIZES}")
    print(f"  Output: {N_OUTPUTS}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")
    print()

    # Normalize inputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # Datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(y_train_norm).float().unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_norm).float(),
        torch.from_numpy(y_val_norm).float().unsqueeze(1),
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

    print(f"Training IDNN for {EPOCHS} epochs...")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples:   {len(X_val):,}")
    print()

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):

        # --- Train ---
        model.train()
        train_total = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item() * len(xb)
            train_count += len(xb)

        train_loss = train_total / train_count

        # --- Validate ---
        model.eval()
        val_total = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_total += loss.item() * len(xb)
                val_count += len(xb)

        val_loss = val_total / val_count

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"best_val={best_val:.6f}  [{elapsed:.1f}s]"
            )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s.  Best val loss: {best_val:.6f}")

    # Store normalization params on model for inference
    model.X_mean = X_mean
    model.X_std = X_std
    model.y_mean = y_mean
    model.y_std = y_std

    return model, history


def predict(model, X):
    """
    Run inference with the trained IDNN.
    """
    model.eval()

    X_norm = (X - model.X_mean) / model.X_std
    X_tensor = torch.from_numpy(X_norm).float().to(DEVICE)

    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy().flatten()

    pred = pred_norm * model.y_std + model.y_mean

    # Clamp speed to non-negative
    pred = np.maximum(pred, 0.0)

    return pred


# ============================================================
# EVALUATION & DISTANCE COMPUTATION
# ============================================================

def compute_idnn_distance(predicted_speed, time_axis, delay_taps):
    """
    Compute cumulative distance from IDNN-predicted speed.

    The IDNN output starts at index `delay_taps`, so we pad the
    beginning with zeros and integrate speed over time.
    """

    n = len(time_axis)

    # Pad the beginning (no prediction for first `delay_taps` samples)
    full_speed = np.zeros(n)
    full_speed[delay_taps:] = predicted_speed

    dt_array = np.diff(time_axis, prepend=time_axis[0])
    dt_med = np.median(np.diff(time_axis))
    dt_array[0] = dt_med

    cumulative_distance = np.zeros(n)
    for i in range(1, n):
        dt_i = dt_array[i]
        if dt_i <= 0 or not np.isfinite(dt_i):
            dt_i = dt_med
        cumulative_distance[i] = cumulative_distance[i - 1] + full_speed[i] * dt_i

    return full_speed, cumulative_distance


# ============================================================
# COMPARISON PLOTS
# ============================================================

def generate_comparison_plots(
    time_axis,
    gt,
    inertial,
    idnn_speed,
    idnn_distance,
    history,
):
    """
    Generate comparison plots: Baseline vs IDNN.
    Saved to results/plots/idnn/.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
    })

    gt_speed = gt["velocity_ms"]
    gt_distance = gt["gt_distance"]
    baseline_speed = inertial["speed"]
    baseline_distance = inertial["cumulative_distance"]

    # Drift metrics
    baseline_error, baseline_drift = calculate_scalar_drift(
        baseline_distance, gt_distance,
    )
    idnn_error, idnn_drift = calculate_scalar_drift(
        idnn_distance, gt_distance,
    )

    # ========================================================
    # FIGURE 1 — Training loss
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)

    ax.set_title("IDNN Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "01_training_loss.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 2 — Speed comparison: GT vs Baseline vs IDNN
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        time_axis, gt_speed * 3.6,
        linewidth=1.5, alpha=0.9,
        label="Vehicle Speed (GT)", color="tab:blue",
    )
    ax.plot(
        time_axis, baseline_speed * 3.6,
        linewidth=1, alpha=0.5,
        label="Baseline IMU Speed", color="tab:orange",
    )
    ax.plot(
        time_axis, idnn_speed * 3.6,
        linewidth=1, alpha=0.8,
        label="IDNN Predicted Speed", color="tab:green",
    )

    ax.set_title("Speed Comparison: Vehicle GT vs Baseline vs IDNN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "02_speed_comparison.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 3 — Speed: IDNN vs GT (zoomed)
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        time_axis, gt_speed * 3.6,
        linewidth=1.5, alpha=0.9,
        label="Vehicle Speed (GT)", color="tab:blue",
    )
    ax.plot(
        time_axis, idnn_speed * 3.6,
        linewidth=1.2, alpha=0.8,
        label="IDNN Predicted Speed", color="tab:green",
    )

    ax.set_title("IDNN Speed Prediction vs Vehicle Ground Truth")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "03_idnn_speed_vs_gt.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 4 — Cumulative distance comparison
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        time_axis, gt_distance,
        linewidth=2, label="Vehicle Distance (GT)",
    )
    ax.plot(
        time_axis, baseline_distance,
        linewidth=1.5, alpha=0.7,
        label="Baseline Inertial Distance",
    )
    ax.plot(
        time_axis, idnn_distance,
        linewidth=1.5, alpha=0.8,
        label="IDNN Distance", color="tab:green",
    )

    ax.set_title("Cumulative Distance: GT vs Baseline vs IDNN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "04_cumulative_distance.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 5 — Distance error comparison
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        time_axis, baseline_error,
        linewidth=1.5, alpha=0.7,
        label="Baseline Error", color="tab:orange",
    )
    ax.plot(
        time_axis, idnn_error,
        linewidth=1.5, alpha=0.8,
        label="IDNN Error", color="tab:green",
    )

    ax.set_title("Distance Error Over Time: Baseline vs IDNN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "05_distance_error.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 6 — Drift percentage comparison
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    valid_b = np.isfinite(baseline_drift)
    valid_i = np.isfinite(idnn_drift)

    if np.any(valid_b):
        ax.plot(
            gt_distance[valid_b], baseline_drift[valid_b],
            linewidth=1.5, alpha=0.7,
            label="Baseline Drift", color="tab:orange",
        )
    if np.any(valid_i):
        ax.plot(
            gt_distance[valid_i], idnn_drift[valid_i],
            linewidth=1.5, alpha=0.8,
            label="IDNN Drift", color="tab:green",
        )

    ax.axhline(10, linestyle="--", linewidth=1.5, color="red", label="10% PS Threshold")

    ax.set_title("Drift Percentage: Baseline vs IDNN")
    ax.set_xlabel("Distance Travelled (m)")
    ax.set_ylabel("Drift (%)")

    # Smart y-axis limit
    max_idnn = np.nanmax(idnn_drift[valid_i]) if np.any(valid_i) else 100
    ax.set_ylim(bottom=0, top=min(200, max(max_idnn * 1.2, 20)))

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "06_drift_comparison.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 7 — Speed error histogram
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    baseline_speed_err = (baseline_speed - gt_speed) * 3.6  # km/h
    idnn_speed_err = (idnn_speed - gt_speed) * 3.6

    axes[0].hist(
        baseline_speed_err, bins=100, alpha=0.7,
        label=f"Baseline (RMSE={np.sqrt(np.mean(baseline_speed_err**2)):.1f} km/h)",
        color="tab:orange",
    )
    axes[0].set_title("Baseline Speed Error Distribution")
    axes[0].set_xlabel("Speed Error (km/h)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        idnn_speed_err, bins=100, alpha=0.7,
        label=f"IDNN (RMSE={np.sqrt(np.mean(idnn_speed_err**2)):.1f} km/h)",
        color="tab:green",
    )
    axes[1].set_title("IDNN Speed Error Distribution")
    axes[1].set_xlabel("Speed Error (km/h)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "07_speed_error_histogram.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 8 — Summary metrics bar chart
    # ========================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Speed RMSE
    baseline_rmse = np.sqrt(np.mean((baseline_speed - gt_speed)**2)) * 3.6
    idnn_rmse = np.sqrt(np.mean((idnn_speed - gt_speed)**2)) * 3.6

    bars = axes[0].bar(
        ["Baseline", "IDNN"],
        [baseline_rmse, idnn_rmse],
        color=["tab:orange", "tab:green"],
    )
    axes[0].set_title("Speed RMSE (km/h)")
    axes[0].set_ylabel("RMSE (km/h)")
    for bar, val in zip(bars, [baseline_rmse, idnn_rmse]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", fontsize=12, fontweight="bold",
        )
    axes[0].grid(True, alpha=0.3, axis="y")

    # Final distance error
    baseline_final_err = abs(baseline_distance[-1] - gt_distance[-1])
    idnn_final_err = abs(idnn_distance[-1] - gt_distance[-1])

    bars = axes[1].bar(
        ["Baseline", "IDNN"],
        [baseline_final_err / 1000, idnn_final_err / 1000],
        color=["tab:orange", "tab:green"],
    )
    axes[1].set_title("Final Distance Error (km)")
    axes[1].set_ylabel("Error (km)")
    for bar, val in zip(bars, [baseline_final_err / 1000, idnn_final_err / 1000]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}", ha="center", fontsize=12, fontweight="bold",
        )
    axes[1].grid(True, alpha=0.3, axis="y")

    # Final drift %
    baseline_final_drift = baseline_final_err / gt_distance[-1] * 100
    idnn_final_drift = idnn_final_err / gt_distance[-1] * 100

    bars = axes[2].bar(
        ["Baseline", "IDNN"],
        [baseline_final_drift, idnn_final_drift],
        color=["tab:orange", "tab:green"],
    )
    axes[2].axhline(10, linestyle="--", color="red", label="10% threshold")
    axes[2].set_title("Final Drift (%)")
    axes[2].set_ylabel("Drift (%)")
    for bar, val in zip(bars, [baseline_final_drift, idnn_final_drift]):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold",
        )
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("IDNN vs Baseline — Summary Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "08_summary_metrics.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nSaved 8 comparison plots to: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def run_idnn_pipeline():

    print("=" * 60)
    print("IDNN PIPELINE — Input Delay Neural Network")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load paired data
    # ----------------------------------------------------------

    print("\n[1/6] Loading paired smartphone + vehicle data...")

    pairs = find_paired_files()

    if not pairs:
        raise FileNotFoundError("No paired files found.")

    s_path, v_path = pairs[0]
    print(f"  Smartphone: {s_path}")
    print(f"  Vehicle:    {v_path}")

    try:
        s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        s_df = pd.read_csv(s_path, low_memory=False, encoding="unicode_escape")

    try:
        v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        v_df = pd.read_csv(v_path, low_memory=False, encoding="unicode_escape")

    s_cols = detect_smartphone_columns(s_df)
    v_cols = detect_vehicle_columns(v_df)

    # Time axis from vehicle
    v_time = numeric_series(v_df, v_cols["time"]).to_numpy()
    v_time = v_time - v_time[0]
    v_time = pd.Series(v_time).interpolate().ffill().bfill().to_numpy()
    for i in range(1, len(v_time)):
        if v_time[i] <= v_time[i - 1]:
            v_time[i] = v_time[i - 1] + 0.1
    time_axis = v_time

    # Align lengths
    min_len = min(len(s_df), len(v_df), len(time_axis))
    s_df = s_df.iloc[:min_len].reset_index(drop=True)
    v_df = v_df.iloc[:min_len].reset_index(drop=True)
    time_axis = time_axis[:min_len]

    print(f"  Samples: {min_len:,}")
    print(f"  Duration: {time_axis[-1]:.0f}s")

    # ----------------------------------------------------------
    # 2. Prepare features and targets
    # ----------------------------------------------------------

    print("\n[2/6] Preparing features and targets...")

    features, target_speed, inertial, gt = prepare_data(
        s_df, v_df, s_cols, v_cols, time_axis,
    )

    print(f"  Feature shape: {features.shape}")
    print(f"  Target shape: {target_speed.shape}")

    # ----------------------------------------------------------
    # 3. Create delay windows and split
    # ----------------------------------------------------------

    print(f"\n[3/6] Creating delay windows (taps={DELAY_TAPS})...")

    X, y = create_delay_windows(features, target_speed, DELAY_TAPS)

    print(f"  Window samples: {X.shape[0]:,}")
    print(f"  Input dim per sample: {X.shape[1]}")

    # Temporal split (no shuffling across time)
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ----------------------------------------------------------
    # 4. Train IDNN
    # ----------------------------------------------------------

    print("\n[4/6] Training IDNN...")

    model, history = train_idnn(X_train, y_train, X_val, y_val)

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "X_mean": model.X_mean,
        "X_std": model.X_std,
        "y_mean": model.y_mean,
        "y_std": model.y_std,
        "config": {
            "n_features": N_FEATURES,
            "delay_taps": DELAY_TAPS,
            "n_outputs": N_OUTPUTS,
            "hidden_sizes": HIDDEN_SIZES,
            "dropout": DROPOUT,
        },
    }, MODEL_SAVE_PATH)
    print(f"  Model saved to: {MODEL_SAVE_PATH}")

    # ----------------------------------------------------------
    # 5. Evaluate
    # ----------------------------------------------------------

    print("\n[5/6] Evaluating IDNN...")

    # Predict on ALL data (train + val + test) for full-length plots
    predicted_speed_all = predict(model, X)

    # Test-only metrics
    predicted_speed_test = predict(model, X_test)

    test_rmse = np.sqrt(np.mean((predicted_speed_test - y_test)**2))
    test_mae = np.mean(np.abs(predicted_speed_test - y_test))

    print(f"\n  Test Speed RMSE: {test_rmse:.4f} m/s ({test_rmse * 3.6:.2f} km/h)")
    print(f"  Test Speed MAE:  {test_mae:.4f} m/s ({test_mae * 3.6:.2f} km/h)")

    # Full-length speed and distance
    idnn_speed_full, idnn_distance_full = compute_idnn_distance(
        predicted_speed_all, time_axis, DELAY_TAPS,
    )

    # Baseline vs IDNN final metrics
    gt_distance = gt["gt_distance"]
    gt_speed_arr = gt["velocity_ms"]

    baseline_final_err = abs(inertial["cumulative_distance"][-1] - gt_distance[-1])
    idnn_final_err = abs(idnn_distance_full[-1] - gt_distance[-1])

    baseline_drift_pct = baseline_final_err / gt_distance[-1] * 100
    idnn_drift_pct = idnn_final_err / gt_distance[-1] * 100

    baseline_speed_rmse = np.sqrt(np.mean((inertial["speed"] - gt_speed_arr)**2)) * 3.6
    idnn_speed_rmse = np.sqrt(np.mean((idnn_speed_full - gt_speed_arr)**2)) * 3.6

    improvement_drift = (1 - idnn_drift_pct / baseline_drift_pct) * 100
    improvement_speed = (1 - idnn_speed_rmse / baseline_speed_rmse) * 100

    print(f"\n  {'Metric':<30s} {'Baseline':>12s} {'IDNN':>12s} {'Improvement':>12s}")
    print(f"  {'-' * 66}")
    print(f"  {'Speed RMSE (km/h)':<30s} {baseline_speed_rmse:>12.2f} {idnn_speed_rmse:>12.2f} {improvement_speed:>11.1f}%")
    print(f"  {'Final Distance Error (m)':<30s} {baseline_final_err:>12.0f} {idnn_final_err:>12.0f} {(1-idnn_final_err/baseline_final_err)*100:>11.1f}%")
    print(f"  {'Final Drift (%)':<30s} {baseline_drift_pct:>12.1f}% {idnn_drift_pct:>12.1f}% {improvement_drift:>11.1f}%")

    # ----------------------------------------------------------
    # 6. Generate comparison plots
    # ----------------------------------------------------------

    print("\n[6/6] Generating comparison plots...")

    generate_comparison_plots(
        time_axis, gt, inertial,
        idnn_speed_full, idnn_distance_full,
        history,
    )

    print("\n" + "=" * 60)
    print("IDNN PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_idnn_pipeline()
