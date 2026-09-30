/// One 10 Hz sample in the canonical sensor frame.
///
/// Canonical frame (what the model was trained on, Android conventions):
///   * acceleration in m/s², *including* gravity
///   * gravity vector in m/s², pointing away from the earth
///     (phone lying face-up reads ≈ (0, 0, +9.81))
///   * angular rate in rad/s, right-handed about the phone axes
class SensorFrame {
  const SensorFrame({
    required this.t,
    required this.ax,
    required this.ay,
    required this.az,
    required this.gravX,
    required this.gravY,
    required this.gravZ,
    required this.gyroX,
    required this.gyroY,
    required this.gyroZ,
  });

  /// Seconds since the stream started.
  final double t;
  final double ax, ay, az;
  final double gravX, gravY, gravZ;
  final double gyroX, gyroY, gyroZ;
}

/// A GNSS observation aligned to a sensor frame.
class GnssSample {
  const GnssSample({
    required this.speedMs,
    this.headingDeg,
    this.east,
    this.north,
  });

  final double speedMs;

  /// Course over ground, degrees clockwise from north.
  final double? headingDeg;

  /// Position in the local tangent plane (metres), if known.
  final double? east;
  final double? north;
}
