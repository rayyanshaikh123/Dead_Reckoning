import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/drives/drive_store.dart';
import 'live_engine.dart';
import 'settings.dart';

class RecorderState {
  const RecorderState({this.current});

  /// The drive being recorded (updated about once a second), or null.
  final DriveSummary? current;

  bool get recording => current != null;
}

/// All saved drives, newest first.
final drivesProvider = FutureProvider<List<DriveSummary>>(
  (ref) => ref.read(driveStoreProvider).list(),
);

/// Records drives automatically from the live engine: starts once the car
/// has been moving for a few seconds, ends after a few minutes parked.
class DriveRecorder extends Notifier<RecorderState> {
  /// GNSS speed that counts as driving, and for how many 10 Hz samples.
  static const startSpeedMs = 4.0;
  static const startSamples = 50;

  /// Stationary samples that end a drive (3 min).
  static const stopSamples = 1800;

  /// Shorter recordings (walking around, parking) are discarded.
  static const minDistanceM = 300.0;

  StreamSubscription<LiveFrame>? _sub;
  String? _id;
  DateTime? _start;
  double _distance = 0;
  double _maxSpeed = 0;
  final _outages = <Outage>[];
  final _track = <TrackPoint>[];
  int _moving = 0, _still = 0, _n = 0;
  IOSink? _csv;
  GyroCheck _gyroAtStart = const GyroCheck();

  // Outage in progress.
  bool _inOutage = false;
  DateTime? _outageStart;
  bool _outageSimulated = false, _outageOnRoad = false, _outageHandheld = false;
  double _outageDistance = 0, _outageSeconds = 0;

  @override
  RecorderState build() {
    ref.listen(liveEngineProvider, (_, _) {}); // keep the engine alive
    _sub = ref.read(liveEngineProvider.notifier).frames.listen(_onFrame);
    ref.onDispose(() {
      _sub?.cancel();
      finish();
    });
    return const RecorderState();
  }

  /// Starts recording now (normally automatic).
  void start() {
    if (_id == null) _begin();
  }

  /// Ends the current drive and saves it.
  Future<void> finish() async {
    final id = _id;
    if (id == null) return;
    _closeOutage(exitErrorM: null);
    final summary = _summary(inProgress: false);
    final csv = _csv;
    _reset();
    await csv?.close();
    final store = ref.read(driveStoreProvider);
    if (summary.distanceM < minDistanceM) {
      await store.delete(id);
    } else {
      await store.save(summary);
    }
    ref.invalidate(drivesProvider);
    state = const RecorderState();
  }

  void _onFrame(LiveFrame lf) {
    final out = lf.output;
    final gpsSpeed = lf.fix?.speedMs;
    final speed = gpsSpeed ?? ((out?.blackout ?? false) ? out!.speed : 0.0);

    if (_id == null) {
      if (!ref.read(appSettingsProvider).autoRecord) return;
      _moving = (gpsSpeed ?? 0) >= startSpeedMs ? _moving + 1 : 0;
      if (_moving >= startSamples) _begin();
      return;
    }

    _n++;
    _distance += speed * 0.1;
    _maxSpeed = math.max(_maxSpeed, speed);
    _still = speed < 1 ? _still + 1 : 0;

    if (out != null) {
      _trackOutage(
        out.blackout,
        out.distanceSinceEntry,
        out.blackoutSamples / 10,
        out.onRoad,
        out.exitErrorM,
        lf.simulated,
        lf.mounted,
      );
    }

    if (_n % 20 == 0) {
      final engine = ref.read(liveEngineProvider);
      final fix = lf.fix;
      if (fix != null) {
        _track.add(TrackPoint(fix.latitude, fix.longitude, true));
      } else if (engine.estimateLat != null) {
        _track.add(TrackPoint(engine.estimateLat!, engine.estimateLon!, false));
      }
    }
    _writeCsv(lf);

    if (_still >= stopSamples) {
      finish();
      return;
    }
    if (_n % 300 == 0) {
      ref.read(driveStoreProvider).save(_summary(inProgress: true));
    }
    if (_n % 10 == 0) {
      state = RecorderState(current: _summary(inProgress: true));
    }
  }

  void _trackOutage(
    bool blackout,
    double distance,
    double seconds,
    bool onRoad,
    double? exitError,
    bool simulated,
    bool mounted,
  ) {
    if (blackout) {
      if (!_inOutage) {
        _inOutage = true;
        _outageStart = DateTime.now();
        _outageSimulated = simulated;
        _outageOnRoad = false;
        _outageHandheld = false;
      }
      _outageHandheld |= !mounted;
      _outageDistance = distance;
      _outageSeconds = seconds;
      _outageOnRoad |= onRoad;
    } else if (_inOutage) {
      _closeOutage(exitErrorM: exitError);
    }
  }

  void _closeOutage({required double? exitErrorM}) {
    if (!_inOutage) return;
    _outages.add(
      Outage(
        start: _outageStart!,
        seconds: _outageSeconds,
        distanceM: _outageDistance,
        simulated: _outageSimulated,
        exitErrorM: exitErrorM,
        onRoad: _outageOnRoad,
        handheld: _outageHandheld,
      ),
    );
    _inOutage = false;
  }

  void _begin() {
    final now = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    _id =
        '${now.year}${two(now.month)}${two(now.day)}-${two(now.hour)}${two(now.minute)}${two(now.second)}';
    _start = now;
    _gyroAtStart = ref.read(liveEngineProvider).gyroCheck;
    if (ref.read(appSettingsProvider).saveSensorLogs) {
      final id = _id!;
      ref.read(driveStoreProvider).sensorLog(id).then((f) {
        if (_id != id) return;
        _csv = f.openWrite()..writeln(_csvHeader);
      }, onError: (Object e) => debugPrint('sensor log unavailable: $e'));
    }
    state = RecorderState(current: _summary(inProgress: true));
  }

  void _reset() {
    _id = null;
    _start = null;
    _distance = 0;
    _maxSpeed = 0;
    _outages.clear();
    _track.clear();
    _moving = _still = _n = 0;
    _csv = null;
    _inOutage = false;
  }

  DriveSummary _summary({required bool inProgress}) {
    final g = ref.read(liveEngineProvider).gyroCheck;
    final n = g.samples - _gyroAtStart.samples;
    double? rmse(double sse0, double sse1) =>
        n < 600 ? null : math.sqrt((sse1 - sse0) / n) * 3.6;
    return DriveSummary(
      id: _id!,
      start: _start!,
      end: DateTime.now(),
      vehicleName: ref.read(appSettingsProvider).vehicleName,
      distanceM: _distance,
      maxSpeedMs: _maxSpeed,
      outages: List.of(_outages),
      track: List.of(_track),
      inProgress: inProgress,
      hasSensorLog:
          _csv != null || ref.read(appSettingsProvider).saveSensorLogs,
      gyroRmseXKmh: rmse(_gyroAtStart.sseX, g.sseX),
      gyroRmseYKmh: rmse(_gyroAtStart.sseY, g.sseY),
    );
  }

  // IO-VNBD-style column names so the logs can join the training pipeline.
  // Gyro is the physical X/Y/Z (not the logger's Yaw/Pitch/Roll labels); the
  // IDR columns lag the sensors by 0.5 s (engine latency).
  static const _csvHeader =
      'TIME SINCE START (ms),GPS LATITUDE (degrees),GPS LONGITUDE (degrees),'
      'GPS SPEED (Kmh),GPS ACCURACY (m),GPS ORIENTATION (deg),'
      'ACCELEROMETER X (m/s2),ACCELEROMETER Y (m/s2),ACCELEROMETER Z (m/s2),'
      'GRAVITY X (m/s2),GRAVITY Y (m/s2),GRAVITY Z (m/s2),'
      'GYROSCOPE X (rad/s),GYROSCOPE Y (rad/s),GYROSCOPE Z (rad/s),'
      'IDR RAW SPEED (Kmh),IDR SPEED (Kmh),IDR MODE,MOUNTED';

  void _writeCsv(LiveFrame lf) {
    final sink = _csv;
    if (sink == null) return;
    final f = lf.frame, fix = lf.fix, o = lf.output;
    String n(double? v, [int d = 4]) => v == null ? '' : v.toStringAsFixed(d);
    final mode = lf.simulated
        ? 'test'
        : ((o?.blackout ?? false) ? 'dr' : 'gps');
    sink.writeln(
      [
        (_n * 100).toString(),
        n(fix?.latitude, 7),
        n(fix?.longitude, 7),
        n(fix?.speedMs == null ? null : fix!.speedMs! * 3.6, 2),
        n(fix?.accuracyM, 1),
        n(fix?.headingDeg, 1),
        n(f.ax),
        n(f.ay),
        n(f.az),
        n(f.gravX),
        n(f.gravY),
        n(f.gravZ),
        n(f.gyroX, 5),
        n(f.gyroY, 5),
        n(f.gyroZ, 5),
        n(o == null ? null : o.rawSpeed * 3.6, 2),
        n(o == null ? null : o.speed * 3.6, 2),
        mode,
        lf.mounted ? '1' : '0',
      ].join(','),
    );
  }
}

final driveRecorderProvider = NotifierProvider<DriveRecorder, RecorderState>(
  DriveRecorder.new,
);
