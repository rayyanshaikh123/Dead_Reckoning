import 'dart:math' as math;

import 'frame.dart';

/// Which horizontal gyro axis the training logger called "Yaw" (the model
/// received `[Yaw, Yaw, Roll]`, see tools/app_export/CANONICAL_FRAME.md).
/// Drive M can't tell X from Y, so this is configurable and to be confirmed
/// on recorded drives.
enum GyroMapping { yawIsX, yawIsY }

/// Converts a live canonical frame (any phone orientation) into the frame
/// IDNN v5 was trained on:
///
///  1. rotate so gravity points along +z (the training phone lay flat);
///  2. turn the horizontal axes so the car's forward direction sits where it
///     was on the training phone: x-axis [trainingMountDeg] to the left of
///     the car's nose (measured on IO-VNBD Drive M);
///  3. rebuild the gyro columns as `[Yaw, Yaw, Roll]`.
class TrainingFrameAdapter {
  TrainingFrameAdapter({this.gyroMapping = GyroMapping.yawIsX});

  /// Angle of the training phone's +x axis, counter-clockwise (to the left)
  /// from the car's forward direction.
  static const trainingMountDeg = 28.5;

  final GyroMapping gyroMapping;

  /// Car forward direction in the flattened (gravity-up) phone frame.
  /// Default +y: a portrait phone upright in a dash cradle, screen facing
  /// the driver, has its top edge pointing up and its back pointing forward,
  /// which flattening turns into +y.
  (double, double) forward = (0.0, 1.0);

  SensorFrame apply(SensorFrame f) {
    final r = rotationToUp(f.gravX, f.gravY, f.gravZ);
    final a = r.apply(f.ax, f.ay, f.az);
    final g = r.apply(f.gravX, f.gravY, f.gravZ);
    final w = r.apply(f.gyroX, f.gyroY, f.gyroZ);

    final (fx, fy) = forward;
    const k = trainingMountDeg * math.pi / 180;
    // Training x = forward rotated left by k; training y = 90° further left.
    final xx = fx * math.cos(k) - fy * math.sin(k),
        xy = fx * math.sin(k) + fy * math.cos(k);
    final yx = -xy, yy = xx;
    double onX((double, double, double) v) => v.$1 * xx + v.$2 * xy;
    double onY((double, double, double) v) => v.$1 * yx + v.$2 * yy;

    final wx = onX(w), wy = onY(w);
    final (yaw, roll) = switch (gyroMapping) {
      GyroMapping.yawIsX => (wx, wy),
      GyroMapping.yawIsY => (wy, wx),
    };
    return SensorFrame(
      t: f.t,
      ax: onX(a),
      ay: onY(a),
      az: a.$3,
      gravX: onX(g),
      gravY: onY(g),
      gravZ: g.$3,
      gyroX: yaw,
      gyroY: yaw,
      gyroZ: roll,
    );
  }
}

/// A 3×3 rotation matrix (row-major).
class Rotation3 {
  const Rotation3(this.m);

  final List<double> m;

  (double, double, double) apply(double x, double y, double z) => (
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  );
}

/// Smallest rotation taking the gravity vector (pointing up) onto +z.
Rotation3 rotationToUp(double gx, double gy, double gz) {
  final n = math.sqrt(gx * gx + gy * gy + gz * gz);
  if (n < 1e-6) return const Rotation3([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  final ux = gx / n, uy = gy / n, uz = gz / n;
  if (uz < -0.999999) {
    // Face-down: 180° about x.
    return const Rotation3([1, 0, 0, 0, -1, 0, 0, 0, -1]);
  }
  // Rodrigues with axis v = u × z = (uy, -ux, 0), cos = uz.
  final vx = uy, vy = -ux;
  final f = 1 / (1 + uz);
  return Rotation3([
    1 - vy * vy * f,
    vx * vy * f,
    vy,
    vx * vy * f,
    1 - vx * vx * f,
    -vx,
    -vy,
    vx,
    1 - (vx * vx + vy * vy) * f,
  ]);
}

/// Learns which way the car's nose points in the flattened phone frame.
///
/// Uses cornering: centripetal acceleration points to the car's left and
/// equals speed × yaw rate. Both come from the phone's own clock (gyro and
/// accelerometer; speed changes slowly), so GNSS latency doesn't matter.
/// Old evidence decays so a re-mounted phone is re-learned.
class MountAlignment {
  MountAlignment({this.halfLifeSamples = 3000, this.minEvidence = 150});

  /// Evidence half-life (samples; 3000 = 5 min at 10 Hz).
  final int halfLifeSamples;

  /// Accumulated |centripetal|² (m²/s⁴) needed before trusting the estimate.
  final double minEvidence;

  double _lx = 0, _ly = 0, _evidence = 0;

  /// Adds one sample. [hx], [hy]: horizontal linear acceleration in the
  /// flattened frame; [omegaUp]: yaw rate about the up axis (rad/s, +left).
  void add({
    required double hx,
    required double hy,
    required double speedMs,
    required double omegaUp,
  }) {
    final decay = math.pow(0.5, 1 / halfLifeSamples).toDouble();
    _lx *= decay;
    _ly *= decay;
    _evidence *= decay;
    if (speedMs < 4) return;
    final cent = speedMs * omegaUp;
    if (cent.abs() < 0.3) return; // not really turning
    _lx += cent * hx;
    _ly += cent * hy;
    _evidence += cent * cent;
  }

  bool get learned => _evidence >= minEvidence && (_lx != 0 || _ly != 0);

  /// 0–1 progress towards a trusted estimate.
  double get progress => (_evidence / minEvidence).clamp(0.0, 1.0);

  /// Unit forward vector, or null until [learned].
  (double, double)? get forward {
    if (!learned) return null;
    final n = math.sqrt(_lx * _lx + _ly * _ly);
    // Left rotated 90° clockwise is forward.
    return (_ly / n, -_lx / n);
  }

  /// Forgets everything (the phone was re-mounted).
  void reset() {
    _lx = 0;
    _ly = 0;
    _evidence = 0;
  }

  /// Restores a previously learned direction with some starting evidence.
  void seed(double fx, double fy) {
    // forward (fx, fy) ⇒ left (-fy, fx)
    _lx = -fy * minEvidence;
    _ly = fx * minEvidence;
    _evidence = minEvidence;
  }
}
