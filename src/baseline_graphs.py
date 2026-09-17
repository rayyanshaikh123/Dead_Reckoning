import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from pyproj import CRS, Transformer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = "data/raw/IO-VNBD"
OUTPUT_DIR = "results/plots/baseline"

# Filtering
LOWPASS_CUTOFF_HZ = 3.0
ACCEL_CLIP_MS2 = 40.0

# Minimum distance before drift percentage is reported.
MIN_DISTANCE_FOR_DRIFT_M = 50.0

# Number of initial samples used for approximate static bias.
INITIAL_CALIBRATION_SECONDS = 2.0

# Gravity constant
G = 9.80665


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(columns, patterns):
    """
    Find the first column matching any regex pattern.
    """
    for col in columns:
        name = str(col).strip().lower()

        for pattern in patterns:
            if re.search(pattern, name):
                return col

    return None


def detect_smartphone_columns(df):
    columns = list(df.columns)

    ax = find_column(columns, [
        r"accelerometer.*x", r"accel.*x", r"^ax$", r"acc[_\-\s]?x$",
    ])
    ay = find_column(columns, [
        r"accelerometer.*y", r"accel.*y", r"^ay$", r"acc[_\-\s]?y$",
    ])
    az = find_column(columns, [
        r"accelerometer.*z", r"accel.*z", r"^az$", r"acc[_\-\s]?z$",
    ])

    # Gravity sensor columns
    grav_x = find_column(columns, [r"gravity.*x"])
    grav_y = find_column(columns, [r"gravity.*y"])
    grav_z = find_column(columns, [r"gravity.*z"])

    gx = find_column(columns, [
        r"gyroscope.*yaw", r"gyroscope.*x", r"gyro.*x", r"^gx$",
    ])
    gy = find_column(columns, [
        r"gyroscope.*pitch", r"gyroscope.*y", r"gyro.*y", r"^gy$",
    ])
    gz = find_column(columns, [
        r"gyroscope.*roll", r"gyroscope.*z", r"gyro.*z", r"^gz$",
    ])

    lat = find_column(columns, [r"latitude", r"\blat\b"])
    lon = find_column(columns, [r"longitude", r"\blon\b", r"\blng\b"])

    gps_speed = find_column(columns, [r"gps.*speed"])

    timestamp = find_column(columns, [
        r"time.*since.*start", r"timestamp", r"time", r"elapsed",
    ])

    return {
        "ax": ax, "ay": ay, "az": az,
        "grav_x": grav_x, "grav_y": grav_y, "grav_z": grav_z,
        "gx": gx, "gy": gy, "gz": gz,
        "lat": lat, "lon": lon,
        "gps_speed": gps_speed,
        "timestamp": timestamp,
    }


def detect_vehicle_columns(df):
    columns = list(df.columns)

    time = find_column(columns, [r"time.*since.*start", r"time"])
    lat = find_column(columns, [r"latitude", r"\blat\b"])
    lon = find_column(columns, [r"longitude", r"\blon\b"])
    velocity = find_column(columns, [r"^.*velocity.*km", r"velocity"])
    ind_speed = find_column(columns, [r"indicated.*vehicle.*speed"])
    heading = find_column(columns, [r"heading"])
    long_accel = find_column(columns, [r"longitudinal.*accel"])
    lat_accel = find_column(columns, [r"lateral.*accel"])
    yaw_rate = find_column(columns, [r"yaw.*rate"])
    sample_period = find_column(columns, [r"sample.*period"])

    return {
        "time": time, "lat": lat, "lon": lon,
        "velocity": velocity, "ind_speed": ind_speed,
        "heading": heading,
        "long_accel": long_accel, "lat_accel": lat_accel,
        "yaw_rate": yaw_rate,
        "sample_period": sample_period,
    }


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def numeric_series(df, column):
    """
    Safely convert a dataframe column to numeric.
    """
    if column is None:
        return None
    return pd.to_numeric(df[column], errors="coerce")


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_paired_files():
    """
    Find paired smartphone (S-*) and vehicle (V-*) CSV files
    from the synchronised dataset.
    """

    # Search in synchronised categorised dataset first
    sync_root = os.path.join(
        DATA_ROOT,
        "Synchronised V abd S datasets",
        "Categorised IOVNB Dataset",
    )

    pairs = []

    if os.path.isdir(sync_root):
        for root, dirs, files in os.walk(sync_root):
            s_files = sorted([f for f in files if f.startswith("S-") and f.endswith(".csv")])
            v_files = sorted([f for f in files if f.startswith("V-") and f.endswith(".csv")])

            for sf in s_files:
                # Try to find matching V-file
                base = sf[2:]  # Remove "S-" prefix
                v_name = "V-" + base
                v_match = [vf for vf in v_files if vf.lower() == v_name.lower()]

                if v_match:
                    pairs.append((
                        os.path.join(root, sf),
                        os.path.join(root, v_match[0]),
                    ))

    if not pairs:
        # Fallback: search uncategorised
        s_dataset_dir = os.path.join(
            DATA_ROOT, "Synchronised V abd S datasets",
            "Uncategorised IOVNB Dataset", "S-Dataset",
        )
        v_dataset_dir = os.path.join(
            DATA_ROOT, "Synchronised V abd S datasets",
            "Uncategorised IOVNB Dataset", "V-Dataset",
        )

        if os.path.isdir(s_dataset_dir) and os.path.isdir(v_dataset_dir):
            for sf in os.listdir(s_dataset_dir):
                if sf.startswith("S-") and sf.endswith(".csv"):
                    base = sf[2:]
                    v_path = os.path.join(v_dataset_dir, "V-" + base)
                    if os.path.exists(v_path):
                        pairs.append((
                            os.path.join(s_dataset_dir, sf),
                            v_path,
                        ))

    return pairs


# ============================================================
# FILTERING
# ============================================================

def butter_lowpass(signal, cutoff_hz, fs, order=4):
    """
    Zero-phase low-pass filtering.
    """
    if len(signal) < 20:
        return signal.copy()

    nyquist = fs / 2.0

    if cutoff_hz >= nyquist:
        return signal.copy()

    normalized = cutoff_hz / nyquist
    b, a = butter(order, normalized, btype="low")

    try:
        return filtfilt(b, a, signal)
    except ValueError:
        return signal.copy()


# ============================================================
# GNSS PROJECTION
# ============================================================

def project_gnss_to_local_xy(lat, lon):
    """
    Convert latitude/longitude to a local azimuthal-equidistant
    coordinate system centered on the first valid GNSS point.
    """

    valid = np.isfinite(lat) & np.isfinite(lon)

    if not np.any(valid):
        raise ValueError("No valid GNSS coordinates found.")

    first_valid = np.where(valid)[0][0]

    lat0 = lat[first_valid]
    lon0 = lon[first_valid]

    local_crs = CRS.from_proj4(
        f"+proj=aeqd "
        f"+lat_0={lat0} "
        f"+lon_0={lon0} "
        f"+datum=WGS84 "
        f"+units=m"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    lon_clean = pd.Series(lon).interpolate().ffill().bfill().to_numpy()
    lat_clean = pd.Series(lat).interpolate().ffill().bfill().to_numpy()

    x, y = transformer.transform(lon_clean, lat_clean)

    x -= x[first_valid]
    y -= y[first_valid]

    return x, y


# ============================================================
# BASELINE INERTIAL ESTIMATION
# ============================================================

def compute_baseline_inertial(
    ax, ay, az,
    grav_x, grav_y, grav_z,
    time_axis,
):
    """
    Baseline diagnostic inertial estimator.

    Uses gravity sensor to subtract gravity from raw accelerometer,
    then integrates the linear acceleration magnitude.
    """

    n = len(time_axis)
    dt_array = np.diff(time_axis, prepend=time_axis[0])
    dt_med = np.median(np.diff(time_axis))

    if n > 1:
        dt_array[0] = dt_med

    fs = 1.0 / dt_med if dt_med > 0 else 10.0

    print(f"\nEstimated sampling frequency: {fs:.3f} Hz")
    print(f"Estimated dt: {dt_med:.6f} s")

    # ------------------------------------------
    # Clean NaN and clip spikes
    # ------------------------------------------

    ax = np.nan_to_num(ax, nan=0.0)
    ay = np.nan_to_num(ay, nan=0.0)
    az = np.nan_to_num(az, nan=0.0)

    ax = np.clip(ax, -ACCEL_CLIP_MS2, ACCEL_CLIP_MS2)
    ay = np.clip(ay, -ACCEL_CLIP_MS2, ACCEL_CLIP_MS2)
    az = np.clip(az, -ACCEL_CLIP_MS2, ACCEL_CLIP_MS2)

    # ------------------------------------------
    # Subtract gravity using gravity sensor
    # ------------------------------------------

    has_gravity = (
        grav_x is not None and
        grav_y is not None and
        grav_z is not None
    )

    if has_gravity:
        grav_x = np.nan_to_num(grav_x, nan=0.0)
        grav_y = np.nan_to_num(grav_y, nan=0.0)
        grav_z = np.nan_to_num(grav_z, nan=0.0)

        # Linear acceleration = raw accel - gravity
        lin_ax = ax - grav_x
        lin_ay = ay - grav_y
        lin_az = az - grav_z

        print("\nUsing gravity sensor for gravity subtraction.")
    else:
        # Fallback: naive bias removal from initial samples
        print("\nNo gravity sensor data. Using initial bias subtraction.")

        cal_mask = time_axis <= (time_axis[0] + INITIAL_CALIBRATION_SECONDS)
        if np.sum(cal_mask) < 5:
            cal_mask = np.zeros(n, dtype=bool)
            cal_mask[:min(20, n)] = True

        lin_ax = ax - np.median(ax[cal_mask])
        lin_ay = ay - np.median(ay[cal_mask])
        lin_az = az - np.median(az[cal_mask])

    # ------------------------------------------
    # Low-pass filter the linear acceleration
    # ------------------------------------------

    lin_ax_f = butter_lowpass(lin_ax, LOWPASS_CUTOFF_HZ, fs)
    lin_ay_f = butter_lowpass(lin_ay, LOWPASS_CUTOFF_HZ, fs)
    lin_az_f = butter_lowpass(lin_az, LOWPASS_CUTOFF_HZ, fs)

    # ------------------------------------------
    # Also filter the raw accelerometer for plots
    # ------------------------------------------

    ax_f = butter_lowpass(ax, LOWPASS_CUTOFF_HZ, fs)
    ay_f = butter_lowpass(ay, LOWPASS_CUTOFF_HZ, fs)
    az_f = butter_lowpass(az, LOWPASS_CUTOFF_HZ, fs)

    # ------------------------------------------
    # Horizontal acceleration magnitude
    # ------------------------------------------

    horizontal_accel = np.sqrt(lin_ax_f**2 + lin_ay_f**2)

    # Restore sign using strongest axis
    dominant = np.where(
        np.abs(lin_ax_f) >= np.abs(lin_ay_f),
        lin_ax_f, lin_ay_f,
    )
    horizontal_signed = np.sign(dominant) * horizontal_accel

    # Deadband
    deadband = 0.05
    horizontal_signed[np.abs(horizontal_signed) < deadband] = 0.0

    # ------------------------------------------
    # Integrate velocity and cumulative distance
    # ------------------------------------------

    velocity = np.zeros(n)
    speed = np.zeros(n)
    cumulative_distance = np.zeros(n)

    for i in range(1, n):
        dt_i = dt_array[i]
        if dt_i <= 0 or not np.isfinite(dt_i):
            dt_i = dt_med

        velocity[i] = velocity[i - 1] + horizontal_signed[i] * dt_i

        # Clip unreasonable velocity (70 m/s ≈ 252 km/h)
        velocity[i] = np.clip(velocity[i], -70.0, 70.0)

        speed[i] = np.abs(velocity[i])

        # Cumulative distance: integrate |velocity| (always increasing)
        cumulative_distance[i] = (
            cumulative_distance[i - 1]
            + speed[i - 1] * dt_i
            + 0.5 * np.abs(horizontal_signed[i]) * dt_i**2
        )

    return {
        "ax_raw": ax,
        "ay_raw": ay,
        "az_raw": az,
        "ax_filtered": ax_f,
        "ay_filtered": ay_f,
        "az_filtered": az_f,
        "lin_ax": lin_ax_f,
        "lin_ay": lin_ay_f,
        "lin_az": lin_az_f,
        "horizontal_accel": horizontal_signed,
        "velocity": velocity,
        "speed": speed,
        "cumulative_distance": cumulative_distance,
    }


# ============================================================
# GROUND TRUTH FROM VEHICLE DATA
# ============================================================

def compute_vehicle_ground_truth(v_df, v_cols, time_axis):
    """
    Extract ground truth metrics from vehicle dataset.
    """

    velocity_kmh = numeric_series(v_df, v_cols["velocity"])
    ind_speed_kmh = numeric_series(v_df, v_cols["ind_speed"])
    long_accel_g = numeric_series(v_df, v_cols["long_accel"])
    lat_accel_g = numeric_series(v_df, v_cols["lat_accel"])
    yaw_rate = numeric_series(v_df, v_cols["yaw_rate"])
    heading = numeric_series(v_df, v_cols["heading"])
    v_lat = numeric_series(v_df, v_cols["lat"])
    v_lon = numeric_series(v_df, v_cols["lon"])

    # Clean
    def clean(arr):
        if arr is None:
            return None
        s = pd.Series(arr).interpolate().ffill().bfill()
        return s.to_numpy()

    velocity_ms = clean(velocity_kmh) / 3.6 if velocity_kmh is not None else None
    ind_speed_ms = clean(ind_speed_kmh) / 3.6 if ind_speed_kmh is not None else None
    long_accel_ms2 = clean(long_accel_g) * G if long_accel_g is not None else None
    lat_accel_ms2 = clean(lat_accel_g) * G if lat_accel_g is not None else None
    yaw_rate_clean = clean(yaw_rate)
    heading_clean = clean(heading)

    # GNSS trajectory from vehicle data
    v_lat_clean = clean(v_lat)
    v_lon_clean = clean(v_lon)

    gt_x, gt_y = project_gnss_to_local_xy(v_lat_clean, v_lon_clean)

    # Cumulative distance from vehicle speed
    dt_array = np.diff(time_axis, prepend=time_axis[0])
    dt_med = np.median(np.diff(time_axis))
    dt_array[0] = dt_med

    gt_distance = np.zeros(len(time_axis))
    if velocity_ms is not None:
        for i in range(1, len(time_axis)):
            dt_i = dt_array[i]
            if dt_i <= 0 or not np.isfinite(dt_i):
                dt_i = dt_med
            gt_distance[i] = gt_distance[i - 1] + np.abs(velocity_ms[i]) * dt_i

    return {
        "gt_x": gt_x,
        "gt_y": gt_y,
        "velocity_ms": velocity_ms,
        "ind_speed_ms": ind_speed_ms,
        "long_accel_ms2": long_accel_ms2,
        "lat_accel_ms2": lat_accel_ms2,
        "yaw_rate": yaw_rate_clean,
        "heading": heading_clean,
        "gt_distance": gt_distance,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_scalar_drift(inertial_distance, gt_distance):
    """
    Diagnostic scalar drift: compare cumulative distances.
    """

    distance_error = np.abs(inertial_distance - gt_distance)

    drift_percentage = np.full(len(gt_distance), np.nan)
    mask = gt_distance >= MIN_DISTANCE_FOR_DRIFT_M
    drift_percentage[mask] = (
        distance_error[mask] / gt_distance[mask]
    ) * 100.0

    return distance_error, drift_percentage


# ============================================================
# SANITY CHECKS
# ============================================================

def print_sanity_checks(gt, inertial, time_axis):

    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    print(f"Total time: {time_axis[-1] - time_axis[0]:.1f} s")
    print(f"Total GNSS distance (vehicle): {gt['gt_distance'][-1]:.2f} m")

    if gt["velocity_ms"] is not None:
        print(
            f"Maximum vehicle speed: "
            f"{np.nanmax(gt['velocity_ms']):.2f} m/s "
            f"({np.nanmax(gt['velocity_ms']) * 3.6:.1f} km/h)"
        )

    print(
        f"Maximum inertial speed: "
        f"{np.nanmax(inertial['speed']):.2f} m/s "
        f"({np.nanmax(inertial['speed']) * 3.6:.1f} km/h)"
    )

    print(
        f"Final inertial distance: "
        f"{inertial['cumulative_distance'][-1]:.2f} m"
    )

    dist_err = np.abs(
        inertial["cumulative_distance"][-1] - gt["gt_distance"][-1]
    )
    if gt["gt_distance"][-1] > 0:
        print(
            f"Final distance error: {dist_err:.2f} m "
            f"({dist_err / gt['gt_distance'][-1] * 100:.1f}%)"
        )

    print("=" * 60)


# ============================================================
# PLOTTING
# ============================================================

def generate_plots(time_axis, gt, inertial, s_gx, s_gy, s_gz):
    """
    Generate all diagnostic plots:

    01 — Ground truth trajectory (vehicle GNSS)
    02 — Raw smartphone accelerometer vs vehicle accelerometer
    03 — Raw smartphone gyroscope vs vehicle yaw rate
    04 — Speed comparison (smartphone INS vs vehicle)
    05 — Cumulative distance comparison
    06 — Distance error over time
    07 — Drift percentage vs distance
    08 — Smartphone GPS speed vs vehicle speed
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
    })

    gt_x = gt["gt_x"]
    gt_y = gt["gt_y"]
    gt_distance = gt["gt_distance"]
    gt_speed = gt["velocity_ms"]
    inertial_speed = inertial["speed"]
    inertial_distance = inertial["cumulative_distance"]

    distance_error, drift_percentage = calculate_scalar_drift(
        inertial_distance, gt_distance,
    )

    # ========================================================
    # FIGURE 1 — Vehicle GNSS trajectory
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(gt_x, gt_y, linewidth=2, label="Ground Truth (Vehicle GNSS)")
    ax.scatter(gt_x[0], gt_y[0], s=60, zorder=5, label="Start")
    ax.scatter(gt_x[-1], gt_y[-1], s=60, zorder=5, label="End")

    ax.set_title("Ground-Truth Vehicle Trajectory")
    ax.set_xlabel("Local East / X (m)")
    ax.set_ylabel("Local North / Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "01_ground_truth_trajectory.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 2 — Raw Smartphone Accel vs Vehicle Accel
    # ========================================================

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Smartphone linear acceleration (gravity subtracted)
    axes[0].plot(time_axis, inertial["lin_ax"], alpha=0.8, linewidth=0.5, label="Phone Lin Ax")
    axes[0].plot(time_axis, inertial["lin_ay"], alpha=0.8, linewidth=0.5, label="Phone Lin Ay")

    axes[0].set_title("Smartphone Linear Acceleration (Gravity Subtracted)")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Vehicle acceleration
    if gt["long_accel_ms2"] is not None and gt["lat_accel_ms2"] is not None:
        axes[1].plot(
            time_axis, gt["long_accel_ms2"],
            alpha=0.8, linewidth=0.5,
            label="Vehicle Longitudinal Accel",
            color="tab:red",
        )
        axes[1].plot(
            time_axis, gt["lat_accel_ms2"],
            alpha=0.8, linewidth=0.5,
            label="Vehicle Lateral Accel",
            color="tab:purple",
        )

    axes[1].set_title("Vehicle Acceleration (from OBD/CAN)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "02_acceleration_comparison.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 3 — Raw Smartphone Gyro vs Vehicle Yaw Rate
    # ========================================================

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    if s_gx is not None:
        # Gyroscope is in rad/s, convert to deg/s for comparison
        axes[0].plot(
            time_axis, np.degrees(s_gx),
            alpha=0.7, linewidth=0.5, label="Phone Gyro Yaw (deg/s)",
        )
    if s_gy is not None:
        axes[0].plot(
            time_axis, np.degrees(s_gy),
            alpha=0.7, linewidth=0.5, label="Phone Gyro Pitch (deg/s)",
        )
    if s_gz is not None:
        axes[0].plot(
            time_axis, np.degrees(s_gz),
            alpha=0.7, linewidth=0.5, label="Phone Gyro Roll (deg/s)",
        )

    axes[0].set_title("Smartphone Gyroscope (Raw)")
    axes[0].set_ylabel("Angular Rate (deg/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    if gt["yaw_rate"] is not None:
        axes[1].plot(
            time_axis, gt["yaw_rate"],
            alpha=0.8, linewidth=0.5,
            label="Vehicle Yaw Rate",
            color="tab:red",
        )

    axes[1].set_title("Vehicle Yaw Rate (from OBD/CAN)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Yaw Rate (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "03_gyroscope_comparison.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 4 — Speed comparison
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    if gt_speed is not None:
        ax.plot(
            time_axis, gt_speed * 3.6,
            linewidth=1.5, alpha=0.9,
            label="Vehicle Speed (km/h)",
            color="tab:blue",
        )

    ax.plot(
        time_axis, inertial_speed * 3.6,
        linewidth=1, alpha=0.7,
        label="Baseline IMU Speed (km/h)",
        color="tab:orange",
    )

    ax.set_title("Vehicle Speed vs Baseline IMU Speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "04_speed_comparison.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 5 — Cumulative distance
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        time_axis, gt_distance,
        linewidth=2,
        label="Vehicle Distance (from speed)",
    )
    ax.plot(
        time_axis, inertial_distance,
        linewidth=1.5,
        label="Baseline Inertial Distance",
    )

    ax.set_title("Cumulative Distance Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "05_cumulative_distance.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 6 — Distance error
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        time_axis, distance_error,
        linewidth=2,
        label="Distance Error",
    )

    ax.set_title("Baseline Inertial Distance Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "06_distance_error.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 7 — Drift percentage
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    valid = np.isfinite(drift_percentage)

    if np.any(valid):
        ax.plot(
            gt_distance[valid],
            drift_percentage[valid],
            linewidth=2,
            label="Baseline Drift",
        )

    ax.axhline(
        10, linestyle="--", linewidth=1.5,
        color="red", label="10% PS Threshold",
    )

    ax.set_title("Baseline Drift Percentage vs Distance Travelled")
    ax.set_xlabel("Distance Travelled (m)")
    ax.set_ylabel("Drift (%)")
    ax.set_ylim(bottom=0, top=min(200, np.nanmax(drift_percentage[valid]) * 1.1) if np.any(valid) else 200)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "07_drift_percentage.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # FIGURE 8 — Raw accelerometer readings (all 3 axes)
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_axis, inertial["ax_filtered"], linewidth=0.5, alpha=0.8, label="Phone Ax (filtered)")
    ax.plot(time_axis, inertial["ay_filtered"], linewidth=0.5, alpha=0.8, label="Phone Ay (filtered)")
    ax.plot(time_axis, inertial["az_filtered"], linewidth=0.5, alpha=0.8, label="Phone Az (filtered)")

    ax.set_title("Filtered Smartphone Accelerometer (Raw, Including Gravity)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "08_raw_accelerometer.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nSaved {8} plots to: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def generate_baseline_graphs():

    print("\nSearching for paired smartphone + vehicle CSV files...")

    pairs = find_paired_files()

    if not pairs:
        raise FileNotFoundError(
            f"No paired S-*/V-* CSV files found inside {DATA_ROOT}"
        )

    print(f"\nFound {len(pairs)} paired file(s).")

    # Use first pair for now.
    s_path, v_path = pairs[0]

    print(f"\nSmartphone file: {s_path}")
    print(f"Vehicle file:    {v_path}")

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    try:
        s_df = pd.read_csv(s_path, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        s_df = pd.read_csv(s_path, low_memory=False, encoding="unicode_escape")

    try:
        v_df = pd.read_csv(v_path, low_memory=False, encoding="latin1")
    except UnicodeDecodeError:
        v_df = pd.read_csv(v_path, low_memory=False, encoding="unicode_escape")

    print(f"\nSmartphone shape: {s_df.shape}")
    print(f"Vehicle shape:    {v_df.shape}")

    # ----------------------------------------------------------
    # Detect columns
    # ----------------------------------------------------------

    s_cols = detect_smartphone_columns(s_df)
    v_cols = detect_vehicle_columns(v_df)

    print("\nSmartphone columns detected:")
    for key, value in s_cols.items():
        print(f"  {key:12s}: {value}")

    print("\nVehicle columns detected:")
    for key, value in v_cols.items():
        print(f"  {key:12s}: {value}")

    # ----------------------------------------------------------
    # Time axis: use VEHICLE time (reliable)
    # ----------------------------------------------------------

    v_time = numeric_series(v_df, v_cols["time"])

    if v_time is None:
        raise ValueError("Vehicle time column not found.")

    v_time = v_time.to_numpy()
    v_time = v_time - v_time[0]  # Start from zero

    # Clean vehicle time
    v_time_s = pd.Series(v_time).interpolate().ffill().bfill().to_numpy()

    # Enforce monotonicity
    for i in range(1, len(v_time_s)):
        if v_time_s[i] <= v_time_s[i - 1]:
            v_time_s[i] = v_time_s[i - 1] + 0.1

    time_axis = v_time_s

    print(f"\nTime axis: {time_axis[0]:.1f}s to {time_axis[-1]:.1f}s "
          f"({time_axis[-1] - time_axis[0]:.1f}s total)")

    # ----------------------------------------------------------
    # Align datasets (trim to same length)
    # ----------------------------------------------------------

    min_len = min(len(s_df), len(v_df), len(time_axis))

    s_df = s_df.iloc[:min_len].reset_index(drop=True)
    v_df = v_df.iloc[:min_len].reset_index(drop=True)
    time_axis = time_axis[:min_len]

    print(f"Aligned length: {min_len} samples")

    # ----------------------------------------------------------
    # Extract smartphone sensor data
    # ----------------------------------------------------------

    ax = numeric_series(s_df, s_cols["ax"])
    ay = numeric_series(s_df, s_cols["ay"])
    az = numeric_series(s_df, s_cols["az"])

    grav_x = numeric_series(s_df, s_cols["grav_x"])
    grav_y = numeric_series(s_df, s_cols["grav_y"])
    grav_z = numeric_series(s_df, s_cols["grav_z"])

    gx = numeric_series(s_df, s_cols["gx"])
    gy = numeric_series(s_df, s_cols["gy"])
    gz = numeric_series(s_df, s_cols["gz"])

    required = ["ax", "ay", "az"]
    missing = [key for key in required if s_cols[key] is None]
    if missing:
        raise ValueError(
            "Could not identify: " + ", ".join(missing)
        )

    # Convert to numpy and clean
    ax = ax.to_numpy().astype(float)
    ay = ay.to_numpy().astype(float)
    az = az.to_numpy().astype(float)

    if grav_x is not None:
        grav_x = grav_x.to_numpy().astype(float)
        grav_y = grav_y.to_numpy().astype(float)
        grav_z = grav_z.to_numpy().astype(float)

    if gx is not None:
        gx = gx.interpolate().ffill().bfill().to_numpy().astype(float)
    if gy is not None:
        gy = gy.interpolate().ffill().bfill().to_numpy().astype(float)
    if gz is not None:
        gz = gz.interpolate().ffill().bfill().to_numpy().astype(float)

    # Detect acceleration units
    magnitude = np.sqrt(
        np.nan_to_num(ax)**2 +
        np.nan_to_num(ay)**2 +
        np.nan_to_num(az)**2
    )
    median_mag = np.nanmedian(magnitude)

    print(f"\nAcceleration median magnitude: {median_mag:.2f}")

    if 0.5 <= median_mag <= 2.0:
        print("Detected unit: g → converting to m/s²")
        ax *= G
        ay *= G
        az *= G
        if grav_x is not None:
            grav_x *= G
            grav_y *= G
            grav_z *= G

    # Interpolate NaN
    for arr_name in ['ax', 'ay', 'az']:
        arr = locals()[arr_name]
        s = pd.Series(arr).interpolate().ffill().bfill()
        locals()[arr_name]  # just reference
        if arr_name == 'ax': ax = s.to_numpy()
        elif arr_name == 'ay': ay = s.to_numpy()
        elif arr_name == 'az': az = s.to_numpy()

    if grav_x is not None:
        grav_x = pd.Series(grav_x).interpolate().ffill().bfill().to_numpy()
        grav_y = pd.Series(grav_y).interpolate().ffill().bfill().to_numpy()
        grav_z = pd.Series(grav_z).interpolate().ffill().bfill().to_numpy()

    # ----------------------------------------------------------
    # Compute vehicle ground truth
    # ----------------------------------------------------------

    gt = compute_vehicle_ground_truth(v_df, v_cols, time_axis)

    # ----------------------------------------------------------
    # Compute baseline inertial
    # ----------------------------------------------------------

    inertial = compute_baseline_inertial(
        ax, ay, az,
        grav_x, grav_y, grav_z,
        time_axis,
    )

    # ----------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------

    print_sanity_checks(gt, inertial, time_axis)

    # ----------------------------------------------------------
    # Generate plots
    # ----------------------------------------------------------

    generate_plots(time_axis, gt, inertial, gx, gy, gz)


if __name__ == "__main__":
    generate_baseline_graphs()