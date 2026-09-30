import 'dart:collection';
import 'dart:math' as math;

import 'features_v5.dart';
import 'numeric.dart';

/// Port of `OnlineGNSSCalibrator` (src/crossdrive_pipeline_v5.py).
///
/// While GNSS is healthy, [track] records (model, GNSS) speed pairs. At
/// blackout entry [freeze] computes the mean bias over the last
/// [historyLen] samples (5 s), with the braking guard; [apply] then removes it.
class GnssCalibrator {
  GnssCalibrator({this.historyLen = 50});

  final int historyLen;
  final _model = ListQueue<double>();
  final _gnss = ListQueue<double>();

  void track(double modelSpeed, double gnssSpeed) {
    _model.addLast(modelSpeed);
    _gnss.addLast(gnssSpeed);
    if (_model.length > historyLen) {
      _model.removeFirst();
      _gnss.removeFirst();
    }
  }

  /// Bias to subtract during the blackout that starts now.
  double freeze() {
    if (_model.length < 10) return 0.0;
    final vm = _model.toList();
    final vg = _gnss.toList();
    var sum = 0.0;
    for (var i = 0; i < vm.length; i++) {
      sum += vm[i] - vg[i];
    }
    var bias = sum / vm.length;
    // Braking guard: filter lag while decelerating into the tunnel looks like
    // a negative bias; freezing it would add phantom speed.
    final deltaV = vg.last - vg.first;
    if (deltaV < -1.0 && bias < 0) bias = 0.0;
    return bias;
  }

  /// Calibrated speed during a blackout.
  static double apply(double modelSpeed, double bias, {double? acVibration}) {
    final v = math.max(modelSpeed - bias, 0.0);
    final standstill = acVibration != null
        ? (modelSpeed < 1.0 || acVibration < 0.045)
        : modelSpeed < 0.15;
    return standstill ? 0.0 : v;
  }

  void reset() {
    _model.clear();
    _gnss.clear();
  }
}

/// Learns the vehicle's idle vibration level (σ_idle) from samples taken
/// while GNSS says the car is stationary (< 0.3 m/s), like the training
/// pipeline. Falls back to the 10th percentile of everything seen.
class IdleBaselineEstimator {
  IdleBaselineEstimator({this.maxSamples = 600});

  final int maxSamples;
  final _idle = ListQueue<double>();
  final _all = ListQueue<double>();

  void add(FeatureFrame f, {double? gnssSpeed}) {
    _push(_all, f.eVibRaw);
    if (gnssSpeed != null && gnssSpeed < 0.3) _push(_idle, f.eVibRaw);
  }

  void _push(ListQueue<double> q, double v) {
    q.addLast(v);
    if (q.length > maxSamples) q.removeFirst();
  }

  bool get hasIdleSamples => _idle.isNotEmpty;

  /// Current estimate, floored at 0.01; null until anything has been seen.
  double? get baseline {
    final double raw;
    if (_idle.isNotEmpty) {
      raw = median(_idle.toList());
    } else if (_all.isNotEmpty) {
      raw = percentile(_all.toList(), 10);
    } else {
      return null;
    }
    return math.max(raw, FeatureExtractorV5.minIdleBaseline);
  }
}
