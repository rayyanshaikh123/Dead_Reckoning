import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/model_loader.dart';
import '../data/roads/road_repository.dart';
import '../data/sources/gnss_source.dart';
import '../data/sources/motion_source.dart';
import '../engine/engine.dart';
import 'nav_state.dart';
import 'settings.dart';

/// What the live engine is doing right now.
class LiveEngineState {
  const LiveEngineState({
    this.running = false,
    this.error,
    this.output,
    this.frameRateHz = 0,
    this.idleBaseline = FeatureExtractorV5.minIdleBaseline,
    this.alignmentProgress = 0,
    this.aligned = false,
    this.hadGnss = false,
    this.gnssUsed = false,
    this.estimateLat,
    this.estimateLon,
    this.gravityMagnitude = 0,
    this.frames = 0,
    this.roadsLoaded = false,
    this.gyroCheck = const GyroCheck(),
    this.mounted = true,
    this.wobbleDeg = 0,
  });

  final bool running;
  final String? error;

  /// Latest pipeline output (lags sensors by 0.5 s).
  final DrOutput? output;
  final double frameRateHz;
  final double idleBaseline;
  final double alignmentProgress;
  final bool aligned;

  /// GNSS has been healthy at least once this session (a blackout only means
  /// something after that).
  final bool hadGnss;

  /// Whether the latest frame was fed a GNSS fix.
  final bool gnssUsed;

  /// Dead-reckoned position during an outage.
  final double? estimateLat;
  final double? estimateLon;
  final double gravityMagnitude;
  final int frames;

  /// A road graph is ready for the next outage.
  final bool roadsLoaded;

  /// Running comparison of the two candidate gyro mappings against GPS.
  final GyroCheck gyroCheck;

  /// Phone is held steady by a mount (see [MountStability]). When false,
  /// GPS-free estimates are unreliable and calibration is paused.
  final bool mounted;

  /// Current tilt wobble, degrees (5 s RMS).
  final double wobbleDeg;

  double? get speedKmh => output == null ? null : output!.speed * 3.6;
  bool get inBlackout => output?.blackout ?? false;
}

/// Speed error of each candidate gyro mapping against GPS, accumulated over
/// every drive (see tools/app_export/CANONICAL_FRAME.md §5).
class GyroCheck {
  const GyroCheck({this.sseX = 0, this.sseY = 0, this.samples = 0});

  final double sseX;
  final double sseY;
  final int samples;

  /// RMS speed error, km/h.
  double get rmseXKmh => samples == 0 ? 0 : math.sqrt(sseX / samples) * 3.6;
  double get rmseYKmh => samples == 0 ? 0 : math.sqrt(sseY / samples) * 3.6;

  /// Minutes of GPS-verified driving behind the comparison.
  double get minutes => samples / 600;

  GyroCheck add(double errX, double errY) => GyroCheck(
    sseX: sseX + errX * errX,
    sseY: sseY + errY * errY,
    samples: samples + 1,
  );
}

/// One 10 Hz step of the live engine, for the drive recorder.
class LiveFrame {
  const LiveFrame({
    required this.frame,
    required this.fix,
    required this.output,
    required this.simulated,
    this.mounted = true,
  });

  /// Physical sensor frame (canonical axes).
  final SensorFrame frame;

  /// GNSS fix fed to the engine this step (null during an outage).
  final GnssFix? fix;

  /// Latest engine output (lags [frame] by 0.5 s).
  final DrOutput? output;

  /// Tunnel test active.
  final bool simulated;

  /// Phone held steady by a mount.
  final bool mounted;
}

/// Runs IDNN v5 on the phone's live sensors at 10 Hz, fed with GNSS while it
/// is healthy. Starts as soon as the model has loaded.
class LiveEngine extends Notifier<LiveEngineState> {
  static const _kIdle = 'idle_baseline';
  static const _kMountX = 'mount_forward_x';
  static const _kMountY = 'mount_forward_y';
  static const _kGyroSseX = 'gyro_check_sse_x';
  static const _kGyroSseY = 'gyro_check_sse_y';
  static const _kGyroN = 'gyro_check_n';

  /// A fix older than this, or less accurate than [_maxAccuracyM], counts as
  /// no GNSS.
  static const _maxFixAge = Duration(seconds: 2);
  static const _maxAccuracyM = 30.0;

  IdnnV5? _model;
  DrPipeline? _pipeline;

  /// Same engine with the other gyro mapping, only to compare against GPS.
  DrPipeline? _shadow;
  final _shadowAdapter = TrainingFrameAdapter(gyroMapping: GyroMapping.yawIsY);
  GyroCheck _gyroCheck = const GyroCheck();
  final _frameStream = StreamController<LiveFrame>.broadcast();
  StreamSubscription<SensorFrame>? _sub;
  final _gravity = GravityFallback();
  final _decimator = Decimator();
  final _adapter = TrainingFrameAdapter();
  final _alignment = MountAlignment();
  final _stability = MountStability();
  LocalTangentPlane? _plane;
  RoadData? _roads;
  RoadGuide? _guide;
  double? _lastT;
  final _frameTimes = <double>[];
  int _frames = 0;

  /// Every processed 10 Hz step.
  Stream<LiveFrame> get frames => _frameStream.stream;

  @override
  LiveEngineState build() {
    ref.onDispose(_stop);
    ref.onDispose(_frameStream.close);
    ref.listen(roadProvider, (_, next) {
      _roads = next.data;
      _applyRoads();
    }, fireImmediately: true);
    ref.listen(speedModelProvider, (_, next) {
      final m = next.value;
      if (m != null && _pipeline == null) _start(m.model);
    }, fireImmediately: true);
    return const LiveEngineState();
  }

  void _start(IdnnV5 model) {
    final prefs = ref.read(sharedPrefsProvider);
    final fx = prefs.getDouble(_kMountX), fy = prefs.getDouble(_kMountY);
    if (fx != null && fy != null) _alignment.seed(fx, fy);
    _model = model;
    _gyroCheck = GyroCheck(
      sseX: prefs.getDouble(_kGyroSseX) ?? 0,
      sseY: prefs.getDouble(_kGyroSseY) ?? 0,
      samples: prefs.getInt(_kGyroN) ?? 0,
    );
    _shadow = _newShadow(prefs.getDouble(_kIdle) ?? 0.0105);
    _pipeline = DrPipeline(
      model: model,
      idleBaseline:
          prefs.getDouble(_kIdle) ?? 0.0105, // Drive M's value until learned
      toModelFrame: _adapter.apply,
    )..setRoadGuide(_guide);
    _sub = ref
        .read(motionSamplesProvider)
        .listen(
          _onSample,
          onError: (Object e) {
            debugPrint('motion stream error: $e');
            state = LiveEngineState(error: e.toString());
          },
        );
    state = LiveEngineState(
      running: true,
      idleBaseline: _pipeline!.idleBaseline,
    );
  }

  void _stop() {
    _sub?.cancel();
    _sub = null;
  }

  void _onSample(SensorFrame raw) {
    final s = _gravity.apply(raw);
    final f = _decimator.push(s);
    if (f != null) _onFrame(f);
  }

  void _onFrame(SensorFrame f) {
    final pipeline = _pipeline!;
    // After a long pause (app backgrounded) the filters' history is stale.
    if (_lastT != null && f.t - _lastT! > 1.0) {
      _pipeline = DrPipeline(
        model: _model!,
        idleBaseline: pipeline.idleBaseline,
        toModelFrame: _adapter.apply,
      )..setRoadGuide(_guide);
      _shadow = _newShadow(pipeline.idleBaseline);
    }
    _lastT = f.t;
    _frames++;
    _frameTimes.add(f.t);
    while (_frameTimes.length > 1 && f.t - _frameTimes.first > 2) {
      _frameTimes.removeAt(0);
    }

    final fix = ref.read(gnssProvider).value;
    final fresh =
        fix != null &&
        DateTime.now().difference(fix.time) <= _maxFixAge &&
        fix.accuracyM <= _maxAccuracyM;
    final testing = ref.read(tunnelTestProvider).active;
    if (fresh && _plane == null) {
      _plane = LocalTangentPlane(fix.latitude, fix.longitude);
      _applyRoads();
    }

    GnssSample? gnss;
    if (fresh && !testing) {
      final (e, n) = _plane!.toLocal(fix.latitude, fix.longitude);
      gnss = GnssSample(
        speedMs: fix.speedMs ?? 0,
        headingDeg: fix.headingDeg,
        east: e,
        north: n,
      );
    }

    _stability.add(f);
    if (_stability.remounted) _onRemount();
    final steady = _stability.steady;
    _pipeline!.calibrationPaused = !steady;
    _shadow!.calibrationPaused = !steady;
    if (steady) _learnMount(f, fresh ? fix.speedMs : null);
    final outs = _pipeline!.push(f, gnss: gnss);
    _shadowAdapter.forward = _adapter.forward;
    final shadowOuts = _shadow!.push(f, gnss: gnss);
    _compareMappings(outs, shadowOuts);

    if (_frames % 300 == 0) _persist();

    final out = outs.isEmpty ? state.output : outs.last;
    double? lat, lon;
    if (out != null && out.blackout && out.east != null && _plane != null) {
      (lat, lon) = _plane!.toGeo(out.east!, out.north!);
    }
    final span = _frameTimes.length > 1
        ? _frameTimes.last - _frameTimes.first
        : 0.0;
    state = LiveEngineState(
      running: true,
      output: out,
      frameRateHz: span > 0 ? (_frameTimes.length - 1) / span : 0,
      idleBaseline: _pipeline!.idleBaseline,
      alignmentProgress: _alignment.progress,
      aligned: _alignment.learned,
      hadGnss: state.hadGnss || gnss != null,
      gnssUsed: gnss != null,
      estimateLat: lat,
      estimateLon: lon,
      gravityMagnitude: math.sqrt(
        f.gravX * f.gravX + f.gravY * f.gravY + f.gravZ * f.gravZ,
      ),
      frames: _frames,
      roadsLoaded: _guide != null,
      gyroCheck: _gyroCheck,
      mounted: steady,
      wobbleDeg: _stability.wobbleDeg,
    );
    if (_frameStream.hasListener) {
      _frameStream.add(
        LiveFrame(
          frame: f,
          fix: gnss == null ? null : fix,
          output: out,
          simulated: testing,
          mounted: steady,
        ),
      );
    }
  }

  DrPipeline _newShadow(double idleBaseline) => DrPipeline(
    model: _model!,
    idleBaseline: idleBaseline,
    toModelFrame: _shadowAdapter.apply,
  );

  /// Scores both gyro mappings against GPS speed while moving with GPS.
  void _compareMappings(List<DrOutput> main, List<DrOutput> shadow) {
    for (var i = 0; i < main.length && i < shadow.length; i++) {
      final v = main[i].gnssSpeed;
      if (v == null || v < 3) continue;
      _gyroCheck = _gyroCheck.add(
        main[i].filteredSpeed - v,
        shadow[i].filteredSpeed - v,
      );
    }
  }

  /// The phone settled at a new angle: the learned mount direction no
  /// longer applies, so start learning it again.
  void _onRemount() {
    _alignment.reset();
    _adapter.forward = TrainingFrameAdapter().forward;
    final prefs = ref.read(sharedPrefsProvider);
    prefs.remove(_kMountX);
    prefs.remove(_kMountY);
  }

  /// Builds the road graph for the current area in the engine's local plane.
  void _applyRoads() {
    final roads = _roads, plane = _plane;
    _guide = (roads == null || plane == null || roads.ways.isEmpty)
        ? null
        : GraphGuide(RoadGraph(roads, plane));
    _pipeline?.setRoadGuide(_guide);
  }

  /// Feeds cornering evidence to the mount-alignment learner.
  void _learnMount(SensorFrame f, double? gnssSpeed) {
    if (gnssSpeed == null) return;
    final r = rotationToUp(f.gravX, f.gravY, f.gravZ);
    final (hx, hy, _) = r.apply(f.ax - f.gravX, f.ay - f.gravY, f.az - f.gravZ);
    final g = math.sqrt(
      f.gravX * f.gravX + f.gravY * f.gravY + f.gravZ * f.gravZ,
    );
    if (g < 1) return;
    final omegaUp =
        (f.gyroX * f.gravX + f.gyroY * f.gravY + f.gyroZ * f.gravZ) / g;
    _alignment.add(hx: hx, hy: hy, speedMs: gnssSpeed, omegaUp: omegaUp);
    final fwd = _alignment.forward;
    if (fwd != null) _adapter.forward = fwd;
  }

  void _persist() {
    final prefs = ref.read(sharedPrefsProvider);
    final p = _pipeline;
    if (p != null) prefs.setDouble(_kIdle, p.idleBaseline);
    prefs.setDouble(_kGyroSseX, _gyroCheck.sseX);
    prefs.setDouble(_kGyroSseY, _gyroCheck.sseY);
    prefs.setInt(_kGyroN, _gyroCheck.samples);
    final fwd = _alignment.forward;
    if (fwd != null) {
      prefs.setDouble(_kMountX, fwd.$1);
      prefs.setDouble(_kMountY, fwd.$2);
    }
  }
}

final liveEngineProvider = NotifierProvider<LiveEngine, LiveEngineState>(
  LiveEngine.new,
);
