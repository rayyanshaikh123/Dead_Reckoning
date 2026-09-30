import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../engine/engine.dart';

/// Native motion stream (see ios/Runner/AppDelegate.swift and
/// android/.../MainActivity.kt), already in the canonical frame.
class MotionSource {
  static const _channel = EventChannel('idr/sensors');

  /// Raw samples at roughly [hz] (the OS may deliver slightly off-rate).
  Stream<SensorFrame> samples({double hz = 50}) =>
      _channel.receiveBroadcastStream({'hz': hz}).map(_parse);

  static SensorFrame _parse(dynamic event) {
    final v = [for (final x in event as List) (x as num).toDouble()];
    return SensorFrame(
      t: v[0],
      ax: v[1],
      ay: v[2],
      az: v[3],
      gravX: v[4],
      gravY: v[5],
      gravZ: v[6],
      gyroX: v[7],
      gyroY: v[8],
      gyroZ: v[9],
    );
  }
}

/// Raw motion samples; overridden in tests.
final motionSamplesProvider = Provider<Stream<SensorFrame>>(
  (ref) => MotionSource().samples(),
);

/// Fills in gravity by low-passing the accelerometer on devices without a
/// gravity sensor (the Android handler sends NaN).
class GravityFallback {
  GravityFallback({this.alpha = 0.02});

  final double alpha;
  double? _x, _y, _z;

  SensorFrame apply(SensorFrame f) {
    if (!f.gravX.isNaN) return f;
    _x = _x == null ? f.ax : _x! + alpha * (f.ax - _x!);
    _y = _y == null ? f.ay : _y! + alpha * (f.ay - _y!);
    _z = _z == null ? f.az : _z! + alpha * (f.az - _z!);
    return SensorFrame(
      t: f.t,
      ax: f.ax,
      ay: f.ay,
      az: f.az,
      gravX: _x!,
      gravY: _y!,
      gravZ: _z!,
      gyroX: f.gyroX,
      gyroY: f.gyroY,
      gyroZ: f.gyroZ,
    );
  }
}

/// Turns a ~50 Hz stream into 10 Hz frames by taking the first sample at or
/// after each 100 ms tick (sample-and-hold, like the IO-VNBD phone logger —
/// no averaging, which would shrink the vibration features).
class Decimator {
  Decimator({this.period = 0.1});

  final double period;
  double? _next;

  /// Returns the frame to emit for [s], or null to skip it.
  SensorFrame? push(SensorFrame s) {
    final next = _next;
    if (next == null || s.t >= next) {
      // Re-anchor after a pause instead of emitting a burst of catch-up frames.
      _next = (next == null || s.t - next > period)
          ? s.t + period
          : next + period;
      return s;
    }
    return null;
  }

  void reset() => _next = null;
}
