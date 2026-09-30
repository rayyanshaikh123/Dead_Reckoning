import 'dart:math' as math;

import 'calibration.dart';
import 'features_v5.dart';
import 'filters.dart';
import 'frame.dart';
import 'idnn_v5.dart';
import 'navigation.dart';
import 'road_network.dart';

/// Which physics stages changed the speed on a given sample.
class StageFlags {
  const StageFlags({
    this.gateClamped = false,
    this.ekfBraking = false,
    this.cruiseLocked = false,
    this.zuptStopped = false,
    this.calibrated = false,
  });

  final bool gateClamped;
  final bool ekfBraking;
  final bool cruiseLocked;
  final bool zuptStopped;
  final bool calibrated;
}

/// Engine output for one sample.
class DrOutput {
  const DrOutput({
    required this.index,
    required this.t,
    required this.rawSpeed,
    required this.filteredSpeed,
    required this.speed,
    required this.blackout,
    required this.stages,
    this.gnssSpeed,
    this.bias,
    this.headingDeg,
    this.east,
    this.north,
    this.distanceSinceEntry = 0,
    this.blackoutSamples = 0,
    this.exitErrorM,
    this.onRoad = false,
  });

  final int index;
  final double t;

  /// IDNN v5 output before any filtering (m/s).
  final double rawSpeed;

  /// After gate → EKF → cruise lock → ZUPT (m/s).
  final double filteredSpeed;

  /// Best speed estimate: GNSS-calibrated model speed during a blackout,
  /// otherwise [filteredSpeed].
  final double speed;
  final double? gnssSpeed;
  final bool blackout;
  final StageFlags stages;

  /// Bias frozen at blackout entry (m/s).
  final double? bias;
  final double? headingDeg;

  /// Position in the local plane (m); GNSS while available, DR otherwise.
  final double? east;
  final double? north;
  final double distanceSinceEntry;
  final int blackoutSamples;

  /// Set on the first sample after GNSS returns: distance between the DR
  /// position and the GNSS fix (m).
  final double? exitErrorM;

  /// During an outage: the position is following a road rather than
  /// integrating heading.
  final bool onRoad;
}

class _Rec {
  _Rec(this.frame, this.gnss);

  final SensorFrame frame;
  final GnssSample? gnss;
  double raw = 0;
  bool gate = false, ekf = false, cruise = false;
}

/// Streaming IDNN v5 dead-reckoning pipeline.
///
/// Feed one canonical 10 Hz [SensorFrame] per call, with a [GnssSample]
/// when GNSS is healthy and `null` during an outage. Outputs lag the input
/// by 5 samples (0.5 s): 1 for central-difference jerk, 4 for ZUPT's
/// centred vibration window — the same maths the benchmark was run with.
class DrPipeline {
  DrPipeline({
    required IdnnV5 model,
    double idleBaseline = FeatureExtractorV5.minIdleBaseline,
    this.learnIdleBaseline = true,
    this.dt = 0.1,
    this.toModelFrame,
  }) : _features = FeatureExtractorV5(idleBaseline: idleBaseline, dt: dt),
       _speed = SpeedEstimator(model),
       _gate = KinematicGate(dt: dt),
       _ekf = FeedforwardEkf(dt: dt),
       _cruise = CruiseLock(dt: dt),
       _heading = HeadingTracker(dt: dt);

  final double dt;

  /// Update σ_idle from stationary samples as they arrive.
  final bool learnIdleBaseline;

  /// Maps a physical frame to the frame the model was trained on (see
  /// [TrainingFrameAdapter]); heading still uses the physical frame.
  /// Null for replays, which are already in the training frame.
  final SensorFrame Function(SensorFrame)? toModelFrame;

  final FeatureExtractorV5 _features;
  final SpeedEstimator _speed;
  final KinematicGate _gate;
  final FeedforwardEkf _ekf;
  final CruiseLock _cruise;
  final _zupt = EnhancedZupt();
  final _calibrator = GnssCalibrator();
  final _idle = IdleBaselineEstimator();
  final HeadingTracker _heading;
  final _recs = <int, _Rec>{};
  int _next = 0;

  bool _inBlackout = false;
  double _bias = 0;
  double _distance = 0;
  int _blackoutSamples = 0;
  DeadReckoner? _dr;
  double? _lastEast, _lastNorth;
  RoadGuide? _road;
  bool _onRoad = false;

  /// While true (phone not steadily mounted) the idle-vibration baseline and
  /// the gyro bias are not learned, so hand motion can't corrupt them.
  bool calibrationPaused = false;

  double get idleBaseline => _features.idleBaseline;
  bool get inBlackout => _inBlackout;

  /// Road data for the next outage. At blackout entry the guide snaps to
  /// the road the vehicle is on; if it finds one, positions follow the road,
  /// otherwise they are heading-integrated.
  void setRoadGuide(RoadGuide? guide) => _road = guide;

  List<DrOutput> push(SensorFrame frame, {GnssSample? gnss}) {
    _recs[_next++] = _Rec(frame, gnss);
    final f = _features.push(toModelFrame?.call(frame) ?? frame);
    return f == null ? const [] : _process(f);
  }

  /// Drains everything still buffered (end of a recording).
  List<DrOutput> flush() {
    final out = <DrOutput>[];
    final last = _features.flush();
    if (last != null) out.addAll(_process(last));
    for (final z in _zupt.flush()) {
      out.add(_finalize(z));
    }
    return out;
  }

  List<DrOutput> _process(FeatureFrame f) {
    final rec = _recs[f.index]!;
    if (learnIdleBaseline && !calibrationPaused) {
      _idle.add(f, gnssSpeed: rec.gnss?.speedMs);
      final b = _idle.baseline;
      if (b != null && _idle.hasIdleSamples) _features.idleBaseline = b;
    }
    rec.raw = _speed.push(f);
    final g = _gate.push(rec.raw, f.linAx);
    rec.gate = _gate.clamped;
    final e = _ekf.push(g, f.linAx);
    rec.ekf = _ekf.braking;
    final c = _cruise.push(e, f.linAx, f.gyroZ);
    rec.cruise = _cruise.locked;
    final z = _zupt.push(c, f);
    return z == null ? const [] : [_finalize(z)];
  }

  DrOutput _finalize(ZuptSample z) {
    final rec = _recs.remove(z.index)!;
    final gnss = rec.gnss;
    double? exitError;
    var speed = z.speed;

    if (gnss != null) {
      if (_inBlackout) {
        final dr = _dr;
        if (dr != null && gnss.east != null && gnss.north != null) {
          final de = dr.east - gnss.east!, dn = dr.north - gnss.north!;
          exitError = math.sqrt(de * de + dn * dn);
        }
        _inBlackout = false;
        _dr = null;
      }
      _calibrator.track(z.speed, gnss.speedMs);
      if (!calibrationPaused) {
        _heading.observe(rec.frame, speedMs: gnss.speedMs);
      }
      final h = gnss.headingDeg;
      if (h != null && gnss.speedMs > 2.0) _heading.anchor(h);
      if (gnss.east != null && gnss.north != null) {
        _lastEast = gnss.east;
        _lastNorth = gnss.north;
      }
    } else {
      if (!_inBlackout) {
        _inBlackout = true;
        _bias = _calibrator.freeze();
        _distance = 0;
        _blackoutSamples = 0;
        _dr = (_lastEast != null && _lastNorth != null)
            ? DeadReckoner(_lastEast!, _lastNorth!)
            : null;
        _onRoad =
            _dr != null &&
            (_road?.begin(_lastEast!, _lastNorth!, _heading.headingDeg) ??
                false);
      }
      speed = GnssCalibrator.apply(z.speed, _bias, acVibration: z.acVariance);
      _blackoutSamples++;
      _distance += speed * dt;
      final heading = _heading.propagate(rec.frame);
      final dr = _dr;
      final road = _road;
      if (dr != null && road != null && _onRoad) {
        final (x, y) = road.advance(speed * dt, heading);
        dr
          ..east = x
          ..north = y;
      } else if (dr != null && heading != null) {
        dr.step(speed, heading, dt);
      }
    }

    return DrOutput(
      index: z.index,
      t: rec.frame.t,
      rawSpeed: rec.raw,
      filteredSpeed: z.speed,
      speed: speed,
      gnssSpeed: gnss?.speedMs,
      blackout: _inBlackout,
      bias: _inBlackout ? _bias : null,
      headingDeg: _heading.headingDeg,
      east: _inBlackout ? _dr?.east : gnss?.east,
      north: _inBlackout ? _dr?.north : gnss?.north,
      distanceSinceEntry: _inBlackout ? _distance : 0,
      blackoutSamples: _inBlackout ? _blackoutSamples : 0,
      exitErrorM: exitError,
      onRoad: _inBlackout && _onRoad,
      stages: StageFlags(
        gateClamped: rec.gate,
        ekfBraking: rec.ekf,
        cruiseLocked: rec.cruise,
        zuptStopped: z.stopped,
        calibrated: _inBlackout,
      ),
    );
  }
}
