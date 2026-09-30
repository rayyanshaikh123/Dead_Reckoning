import 'dart:collection';
import 'dart:math' as math;

import 'frame.dart';

/// Tells whether the phone is held steady by a mount, from how much the
/// gravity direction wobbles.
///
/// A mounted phone only tilts as the car pitches and rolls: on IO-VNBD
/// Drive M (2.9 h, mounted) the 5 s RMS tilt wobble was 0.09° typically and
/// 1.0° at the 99.9th percentile. A phone in a hand, on a lap or sliding on
/// a seat wobbles far more. IDR's model and calibration assume a fixed
/// mount, so while the phone isn't steady its GPS-free estimate is flagged
/// as unreliable and calibration learning pauses.
///
/// Also reports when the phone settles at a clearly different orientation
/// ([remounted]), so the mount direction can be re-learned.
class MountStability {
  MountStability({
    this.window = 50,
    this.unsteadyDeg = 4.0,
    this.steadyDeg = 2.5,
    this.settleSamples = 100,
    this.remountDeg = 12.0,
  });

  /// Samples per wobble window (50 = 5 s at 10 Hz).
  final int window;

  /// RMS tilt wobble above which the phone counts as not mounted.
  final double unsteadyDeg;

  /// RMS wobble it must stay below for [settleSamples] to count as mounted
  /// again (hysteresis).
  final double steadyDeg;
  final int settleSamples;

  /// Change of resting orientation that counts as a new mount position.
  final double remountDeg;

  final _dirs = ListQueue<(double, double, double)>();
  bool _steady = true;
  int _calm = 0;
  double _wobble = 0;
  (double, double, double)? _reference;
  bool _remounted = false;

  /// Phone is held by a mount.
  bool get steady => _steady;

  /// Current 5 s RMS tilt wobble, degrees.
  double get wobbleDeg => _wobble;

  /// True for the one sample on which a new resting orientation was
  /// detected; cleared on the next [add].
  bool get remounted => _remounted;

  void add(SensorFrame f) {
    _remounted = false;
    final n = math.sqrt(
      f.gravX * f.gravX + f.gravY * f.gravY + f.gravZ * f.gravZ,
    );
    if (n < 1e-6 || n.isNaN) return;
    _dirs.addLast((f.gravX / n, f.gravY / n, f.gravZ / n));
    if (_dirs.length > window) _dirs.removeFirst();
    if (_dirs.length < window) return;

    final mean = _mean();
    var sq = 0.0;
    for (final d in _dirs) {
      final a = _angleDeg(d, mean);
      sq += a * a;
    }
    _wobble = math.sqrt(sq / _dirs.length);

    if (_wobble > unsteadyDeg) {
      _steady = false;
      _calm = 0;
      return;
    }
    if (!_steady) {
      _calm = _wobble < steadyDeg ? _calm + 1 : 0;
      if (_calm < settleSamples) return;
      _steady = true;
      final ref = _reference;
      if (ref != null && _angleDeg(ref, mean) > remountDeg) _remounted = true;
      _reference = mean;
      return;
    }
    // Steady: track the resting orientation (slowly, so hills don't move it).
    final ref = _reference;
    _reference = ref == null
        ? mean
        : _normalize((
            ref.$1 * 0.99 + mean.$1 * 0.01,
            ref.$2 * 0.99 + mean.$2 * 0.01,
            ref.$3 * 0.99 + mean.$3 * 0.01,
          ));
  }

  (double, double, double) _mean() {
    var x = 0.0, y = 0.0, z = 0.0;
    for (final d in _dirs) {
      x += d.$1;
      y += d.$2;
      z += d.$3;
    }
    return _normalize((x, y, z));
  }

  static (double, double, double) _normalize((double, double, double) v) {
    final n = math.sqrt(v.$1 * v.$1 + v.$2 * v.$2 + v.$3 * v.$3);
    return (v.$1 / n, v.$2 / n, v.$3 / n);
  }

  static double _angleDeg(
    (double, double, double) a,
    (double, double, double) b,
  ) {
    final dot = (a.$1 * b.$1 + a.$2 * b.$2 + a.$3 * b.$3).clamp(-1.0, 1.0);
    return math.acos(dot) * 180 / math.pi;
  }
}
