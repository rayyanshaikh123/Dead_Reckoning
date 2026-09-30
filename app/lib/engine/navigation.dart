import 'dart:math' as math;

import 'frame.dart';

/// Local east/north plane around an origin, using the WGS84 ellipsoid's
/// radii of curvature at the origin latitude. Over the few kilometres of a
/// tunnel this agrees with a proper azimuthal-equidistant projection to a few
/// centimetres (a spherical Earth would be ~0.2 % short at UK latitudes).
class LocalTangentPlane {
  factory LocalTangentPlane(double originLat, double originLon) {
    const a = 6378137.0; // WGS84 semi-major axis
    const e2 = 6.69437999014e-3; // first eccentricity squared
    final phi = originLat * math.pi / 180;
    final s = math.sin(phi);
    final w = 1 - e2 * s * s;
    final meridional = a * (1 - e2) / (w * math.sqrt(w));
    final primeVertical = a / math.sqrt(w);
    return LocalTangentPlane._(
      originLat,
      originLon,
      metresPerRadLat: meridional,
      metresPerRadLon: primeVertical * math.cos(phi),
    );
  }

  LocalTangentPlane._(
    this.originLat,
    this.originLon, {
    required double metresPerRadLat,
    required double metresPerRadLon,
  }) : _mLat = metresPerRadLat,
       _mLon = metresPerRadLon;

  final double originLat;
  final double originLon;
  final double _mLat;
  final double _mLon;

  (double east, double north) toLocal(double lat, double lon) => (
    (lon - originLon) * math.pi / 180 * _mLon,
    (lat - originLat) * math.pi / 180 * _mLat,
  );

  (double lat, double lon) toGeo(double east, double north) => (
    originLat + north / _mLat * 180 / math.pi,
    originLon + east / _mLon * 180 / math.pi,
  );
}

/// Propagates heading through a blackout from the gyroscope.
///
/// Anchored to the GNSS course at blackout entry, then integrates the yaw
/// rate about the gravity ("up") axis. With the canonical frame (gravity
/// vector points up) a positive rate about "up" is a left turn, i.e. the
/// compass heading *decreases*. Magnetometer yaw is not used: it is
/// unreliable inside steel-reinforced tunnels.
class HeadingTracker {
  HeadingTracker({this.dt = 0.1});

  final double dt;
  double? _heading;

  /// Gyro bias about the up axis (rad/s), learned while parked.
  double bias = 0;
  bool _biasKnown = false;

  double? get headingDeg => _heading;
  bool get biasKnown => _biasKnown;

  /// Sets heading from GNSS (only meaningful while moving).
  void anchor(double headingDeg) => _heading = headingDeg % 360;

  /// Yaw rate about the up axis (rad/s, positive = left turn).
  static double upRate(SensorFrame f) {
    final g = math.sqrt(
      f.gravX * f.gravX + f.gravY * f.gravY + f.gravZ * f.gravZ,
    );
    if (g < 1e-6) return 0;
    return (f.gyroX * f.gravX + f.gyroY * f.gravY + f.gyroZ * f.gravZ) / g;
  }

  int _stillSamples = 0;

  /// Learns the gyro bias while GNSS says the car is parked, where the true
  /// yaw rate is zero. Only after 3 s of continuous standstill, so the tail
  /// of a turn (GNSS reports "stopped" with some lag) isn't mistaken for
  /// bias. (Fitting bias and scale against the GNSS course while moving was
  /// tried on IO-VNBD Drive M and was too noisy to help.)
  void observe(SensorFrame f, {required double speedMs}) {
    if (speedMs >= 0.3) {
      _stillSamples = 0;
      return;
    }
    if (++_stillSamples < 30) return;
    final w = upRate(f);
    bias = _biasKnown ? bias + 0.01 * (w - bias) : w;
    _biasKnown = true;
  }

  /// Advances heading by one sample of gyro data. Returns the new heading.
  double? propagate(SensorFrame f) {
    final h = _heading;
    if (h == null) return null;
    final next = (h - (upRate(f) - bias) * 180 / math.pi * dt) % 360;
    _heading = next < 0 ? next + 360 : next;
    return _heading;
  }

  void reset() => _heading = null;
}

/// `dead_reckon_2d`: integrates speed along heading (x east, y north).
class DeadReckoner {
  DeadReckoner(this.east, this.north);

  double east;
  double north;

  /// Advances one step; heading in degrees clockwise from north.
  void step(double speedMs, double headingDeg, double dt) {
    final h = headingDeg * math.pi / 180;
    east += speedMs * math.sin(h) * dt;
    north += speedMs * math.cos(h) * dt;
  }
}

/// `map_match_to_road`: places a distance travelled along a road polyline
/// (`np.interp` semantics, clamped at both ends).
class PolylineMatcher {
  PolylineMatcher(List<double> xs, List<double> ys)
    : assert(xs.length == ys.length && xs.length >= 2),
      _x = xs,
      _y = ys,
      _cum = _cumulative(xs, ys);

  final List<double> _x;
  final List<double> _y;
  final List<double> _cum;

  double get length => _cum.last;

  static List<double> _cumulative(List<double> x, List<double> y) {
    final c = List<double>.filled(x.length, 0);
    for (var i = 1; i < x.length; i++) {
      final dx = x[i] - x[i - 1], dy = y[i] - y[i - 1];
      c[i] = c[i - 1] + math.sqrt(dx * dx + dy * dy);
    }
    return c;
  }

  (double x, double y) at(double distance) {
    if (distance <= _cum.first) return (_x.first, _y.first);
    if (distance >= _cum.last) return (_x.last, _y.last);
    var lo = 0, hi = _cum.length - 1;
    while (hi - lo > 1) {
      final mid = (lo + hi) >> 1;
      if (_cum[mid] <= distance) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    final span = _cum[hi] - _cum[lo];
    final f = span == 0 ? 0.0 : (distance - _cum[lo]) / span;
    return (_x[lo] + (_x[hi] - _x[lo]) * f, _y[lo] + (_y[hi] - _y[lo]) * f);
  }
}
