import 'dart:collection';
import 'dart:math' as math;

import 'features_v5.dart';
import 'numeric.dart';

/// Streaming ports of the v5 physics filters (src/crossdrive_pipeline_v5.py).
/// Each keeps the same recursion as the batch version, so feeding a series
/// sample by sample reproduces the numpy output.

/// `kinematic_plausibility_gate`: caps implausible speed jumps.
class KinematicGate {
  KinematicGate({this.dt = 0.1, this.maxAccel = 3.5, this.maxDecel = 6.0});

  final double dt;
  final double maxAccel;
  final double maxDecel;
  double? _prev; // un-clamped, as in the batch recursion

  /// True if the last sample was capped.
  bool clamped = false;

  double push(double raw, double axLong) {
    final prev = _prev;
    double g;
    clamped = false;
    if (prev == null) {
      g = raw;
    } else {
      final dvdt = (raw - prev) / dt;
      if (dvdt > maxAccel) {
        clamped = true;
        g = axLong > 1.0
            ? prev + math.max(dvdt, maxAccel) * dt
            : prev + maxAccel * dt;
      } else if (dvdt < -maxDecel) {
        clamped = true;
        g = axLong < -2.0
            ? prev + math.min(dvdt, -maxDecel) * dt
            : prev - maxDecel * dt;
      } else {
        g = raw;
      }
    }
    _prev = g;
    return math.max(g, 0.0);
  }

  void reset() => _prev = null;
}

/// `ekf_with_acceleration_feedforward`: 1-state Kalman smoother that trusts
/// the accelerometer during hard braking.
class FeedforwardEkf {
  FeedforwardEkf({
    this.dt = 0.1,
    this.qCruise = 0.2,
    this.qBrake = 1.5,
    this.r = 3.0,
  });

  final double dt;
  final double qCruise;
  final double qBrake;
  final double r;
  double? _x;
  double _p = 1.0;

  bool braking = false;

  double push(double speed, double axLong) {
    var x = _x ?? speed;
    braking = axLong < -1.5;
    final q = braking ? qBrake : qCruise;
    final xPred = braking ? math.max(x + axLong * dt, 0.0) : x;
    final pPred = _p + q;
    final k = pPred / (pPred + r);
    x = xPred + k * (speed - xPred);
    _p = (1.0 - k) * pPred;
    _x = x;
    return math.max(x, 0.0);
  }

  void reset() {
    _x = null;
    _p = 1.0;
  }
}

/// `apply_cruise_lock`: holds speed steady after 1.5 s of no longitudinal
/// acceleration and no yaw.
class CruiseLock {
  CruiseLock({this.dt = 0.1, double windowSec = 1.5})
    // Same float arithmetic as `int(window_sec / dt)` in Python.
    : windowPts = (windowSec / dt).toInt();

  final double dt;
  final int windowPts;
  int _streak = 0;
  double? _prevOut;

  bool locked = false;

  double push(double speed, double axLong, double gyroZ) {
    final steady = axLong.abs() < 0.18 && gyroZ.abs() < 0.02;
    _streak = steady ? _streak + 1 : 0;
    var out = speed;
    locked = _streak >= windowPts && _prevOut != null;
    if (locked) out = 0.95 * _prevOut! + 0.05 * speed;
    _prevOut = out;
    return out;
  }

  void reset() {
    _streak = 0;
    _prevOut = null;
  }
}

/// Result of [EnhancedZupt] for one sample.
class ZuptSample {
  const ZuptSample(this.index, this.speed, this.stopped, this.acVariance);

  final int index;
  final double speed;
  final bool stopped;

  /// Rolling 10-sample variance of |lin acc| (also the calibrator's
  /// `ac_vibration`).
  final double acVariance;
}

/// `apply_enhanced_zupt`: forces speed to 0 when the car is stationary.
///
/// The Python version smooths vibration energy with a *centred* 10-sample
/// window (`np.convolve(mode="same")`, samples k-5 … k+4), so the output for
/// sample k is released 4 samples later. [flush] drains the tail using the
/// same zero padding numpy applies at the array end.
class EnhancedZupt {
  EnhancedZupt({
    this.thresholdSpeed = 0.35,
    this.thresholdEnergy = 0.25,
    this.thresholdVar = 0.035,
    this.window = 10,
  }) : _var = RollingVariance(window);

  final double thresholdSpeed;
  final double thresholdEnergy;
  final double thresholdVar;
  final int window;
  final RollingVariance _var;

  /// Samples after k that the centred window needs.
  int get lookahead => window - window ~/ 2 - 1;

  final _energy = ListQueue<double>(); // energy for samples (k-5) … newest
  final _pending = ListQueue<(int, double, double)>(); // (index, speed, var)
  int _energyStart = 0; // sample index of _energy.first

  /// Adds one sample; returns the output for the sample 4 steps back, if ready.
  ZuptSample? push(double speed, FeatureFrame f) {
    final a = f.values;
    final energy = f32(
      math.sqrt(
        f32((f32(a[0] * a[0]) + f32(a[1] * a[1]) + f32(a[2] * a[2])) / 3),
      ),
    );
    final linNorm = f32(
      math.sqrt(f32(f32(a[0] * a[0]) + f32(a[1] * a[1]) + f32(a[2] * a[2]))),
    );
    _energy.addLast(energy);
    _pending.addLast((f.index, speed, _var.push(linNorm)));
    return _ready(false);
  }

  /// Emits the remaining samples at end of stream.
  List<ZuptSample> flush() {
    final out = <ZuptSample>[];
    while (_pending.isNotEmpty) {
      out.add(_ready(true)!);
    }
    return out;
  }

  ZuptSample? _ready(bool draining) {
    if (_pending.isEmpty) return null;
    final (idx, speed, variance) = _pending.first;
    final before = window ~/ 2; // 5
    final after = lookahead; // 4
    final newest = _energyStart + _energy.length - 1;
    if (!draining && newest < idx + after) return null;

    var sum = 0.0;
    for (var j = idx - before; j <= idx + after; j++) {
      final pos = j - _energyStart;
      if (j >= 0 && pos >= 0 && pos < _energy.length) {
        sum += _energy.elementAt(pos);
      }
    }
    final smooth = sum / window;
    _pending.removeFirst();
    // Drop energies no longer needed by any pending sample.
    while (_energy.isNotEmpty && _energyStart < idx + 1 - before) {
      _energy.removeFirst();
      _energyStart++;
    }
    final stopped =
        speed < thresholdSpeed ||
        smooth < thresholdEnergy ||
        variance < thresholdVar;
    return ZuptSample(idx, stopped ? 0.0 : speed, stopped, variance);
  }

  void reset() {
    _energy.clear();
    _pending.clear();
    _energyStart = 0;
    _var.reset();
  }
}
