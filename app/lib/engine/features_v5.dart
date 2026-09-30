import 'dart:math' as math;
import 'dart:typed_data';

import 'frame.dart';
import 'numeric.dart';

/// The 18 IDNN v5 input features for one sample, plus the raw signals the
/// physics filters need.
class FeatureFrame {
  const FeatureFrame({
    required this.index,
    required this.values,
    required this.linAx,
    required this.gyroZ,
    required this.eVibRaw,
  });

  /// Sample index in the stream (0-based).
  final int index;

  /// 18 features in `feature_order`, rounded to float32 like the training data.
  final Float64List values;

  /// Longitudinal linear acceleration (float64), used by gate / EKF / cruise.
  final double linAx;

  /// Raw gyro z (float64), used by cruise lock.
  final double gyroZ;

  /// Un-normalised vibration energy, used to learn the idle baseline.
  final double eVibRaw;

  static const count = 18;
}

class _Pending {
  _Pending(
    this.index,
    this.lin,
    this.gyro,
    this.stat,
    this.linAx,
    this.gyroZ,
    this.eVibRaw,
  );

  final int index;
  final List<double> lin; // lin_ax, lin_ay, lin_az
  final List<double> gyro; // gx, gy, gz
  final List<double> stat; // grav xyz, a_horiz_norm, pitch (before e_vib)
  final double linAx;
  final double gyroZ;
  final double eVibRaw;
}

/// Streaming port of `extract_features_v5` (src/crossdrive_pipeline_v5.py).
///
/// Jerk uses numpy's central difference, so the frame for sample *k* is
/// emitted when sample *k+1* arrives (100 ms latency). [flush] emits the last
/// pending frame with numpy's one-sided end difference.
class FeatureExtractorV5 {
  FeatureExtractorV5({required double idleBaseline, this.dt = 0.1})
    : _idleBaseline = math.max(idleBaseline, minIdleBaseline);

  /// Floor applied in the Python pipeline (`max(idle_baseline, 0.01)`).
  static const minIdleBaseline = 0.01;

  final double dt;
  double _idleBaseline;
  final _vib = RollingVariance(5);

  _Pending? _prev2;
  _Pending? _prev;
  int _next = 0;

  double get idleBaseline => _idleBaseline;
  set idleBaseline(double v) => _idleBaseline = math.max(v, minIdleBaseline);

  /// Adds sample [f]; returns the features of the previous sample, if any.
  FeatureFrame? push(SensorFrame f) {
    final cur = _stage(f);
    final prev = _prev;
    FeatureFrame? out;
    if (prev != null) {
      final before = _prev2;
      out = before == null
          ? _emit(prev, cur, prev, dt) // first sample: forward difference
          : _emit(prev, cur, before, 2 * dt); // central difference
    }
    _prev2 = _prev;
    _prev = cur;
    return out;
  }

  /// Emits the final pending sample (backward difference). Call at stream end.
  FeatureFrame? flush() {
    final last = _prev;
    final before = _prev2;
    if (last == null || before == null) return null;
    _prev = null;
    _prev2 = null;
    return _emit(last, last, before, dt);
  }

  void reset() {
    _prev = null;
    _prev2 = null;
    _next = 0;
    _vib.reset();
  }

  _Pending _stage(SensorFrame f) {
    final linAx = f.ax - f.gravX;
    final linAy = f.ay - f.gravY;
    final linAz = f.az - f.gravZ;

    final gMag =
        math.sqrt(f.gravX * f.gravX + f.gravY * f.gravY + f.gravZ * f.gravZ) +
        1e-8;
    final ux = f.gravX / gMag, uy = f.gravY / gMag, uz = f.gravZ / gMag;
    final aDotG = f.ax * ux + f.ay * uy + f.az * uz;
    final hx = f.ax - aDotG * ux;
    final hy = f.ay - aDotG * uy;
    final hz = f.az - aDotG * uz;
    final aHoriz = math.sqrt(hx * hx + hy * hy + hz * hz);

    final pitch = math.atan2(
      f.gravX,
      math.sqrt(f.gravY * f.gravY + f.gravZ * f.gravZ) + 1e-8,
    );

    final linNorm = math.sqrt(linAx * linAx + linAy * linAy + linAz * linAz);
    final eVibRaw = f32(_vib.push(linNorm));

    return _Pending(
      _next++,
      [linAx, linAy, linAz],
      [f.gyroX, f.gyroY, f.gyroZ],
      [f.gravX, f.gravY, f.gravZ, aHoriz, pitch],
      linAx,
      f.gyroZ,
      eVibRaw,
    );
  }

  /// Features for [p], with jerk = ([after] - [before]) / [span].
  FeatureFrame _emit(_Pending p, _Pending after, _Pending before, double span) {
    final v = Float64List(FeatureFrame.count);
    v[0] = f32(p.lin[0]);
    v[1] = f32(p.lin[1]);
    v[2] = f32(p.lin[2]);
    v[3] = f32(p.stat[0]);
    v[4] = f32(p.stat[1]);
    v[5] = f32(p.stat[2]);
    v[6] = f32(p.gyro[0]);
    v[7] = f32(p.gyro[1]);
    v[8] = f32(p.gyro[2]);
    for (var k = 0; k < 3; k++) {
      v[9 + k] = f32((after.lin[k] - before.lin[k]) / span);
      v[12 + k] = f32((after.gyro[k] - before.gyro[k]) / span);
    }
    v[15] = f32(p.stat[3]);
    v[16] = f32(p.stat[4]);
    v[17] = f32(f32(p.eVibRaw / _idleBaseline));
    return FeatureFrame(
      index: p.index,
      values: v,
      linAx: p.linAx,
      gyroZ: p.gyroZ,
      eVibRaw: p.eVibRaw,
    );
  }
}
