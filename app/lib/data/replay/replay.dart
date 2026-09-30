import 'dart:convert';
import 'dart:math' as math;

import '../../engine/engine.dart';

/// A recorded drive segment exported by tools/app_export/export_replay.py.
class ReplayData {
  ReplayData._(this._j)
    : id = _j['id'] as String,
      name = _j['name'] as String,
      outageStart = _j['outage_start'] as int,
      outageEnd = _j['outage_end'] as int,
      roadEnd = _j['road_end'] as int,
      idleBaseline = (_j['idle_baseline'] as num).toDouble(),
      t = _vec(_j['t']),
      gtSpeed = _vec(_j['gt']['speed_ms']),
      gtHeading = _vec(_j['gt']['heading_deg']),
      lat = _vec(_j['gt']['lat']),
      lon = _vec(_j['gt']['lon']),
      pythonEnhanced = _vec(_j['python']['enhanced']);

  factory ReplayData.parse(String json) =>
      ReplayData._(jsonDecode(json) as Map<String, dynamic>);

  final Map<String, dynamic> _j;
  final String id;
  final String name;

  /// First and last (inclusive) sample of the GNSS outage.
  final int outageStart;
  final int outageEnd;

  /// Exclusive end of the ground-truth road used for map-matching.
  final int roadEnd;
  final double idleBaseline;
  final List<double> t;
  final List<double> gtSpeed;
  final List<double> gtHeading;
  final List<double> lat;
  final List<double> lon;

  /// Filtered speed from the Python pipeline (for parity checks).
  final List<double> pythonEnhanced;

  int get length => t.length;

  static List<double> _vec(Object? v) => [
    for (final e in v as List) (e as num).toDouble(),
  ];

  /// Physical frames: accelerometer and gravity as logged (phone flat),
  /// gyro = (logger "Yaw", logger "Roll", vertical rate). The vertical rate
  /// (logger "Pitch") feeds heading propagation only; [toModelFrame] rebuilds
  /// the `[Yaw, Yaw, Roll]` inputs the model was trained on.
  List<SensorFrame> frames() {
    final s = _j['sensors'] as Map<String, dynamic>;
    final c = {for (final k in s.keys) k: _vec(s[k])};
    final up = _j['gyro_up'] == null ? null : _vec(_j['gyro_up']);
    return [
      for (var i = 0; i < length; i++)
        SensorFrame(
          t: t[i],
          ax: c['ax']![i],
          ay: c['ay']![i],
          az: c['az']![i],
          gravX: c['grav_x']![i],
          gravY: c['grav_y']![i],
          gravZ: c['grav_z']![i],
          gyroX: c['gx']![i], // logger "Yaw"
          gyroY: c['gz']![i], // logger "Roll"
          gyroZ: up?[i] ?? 0, // logger "Pitch" = rotation about up
        ),
    ];
  }

  /// Physical replay frame → the model's training inputs.
  static SensorFrame toModelFrame(SensorFrame f) => SensorFrame(
    t: f.t,
    ax: f.ax,
    ay: f.ay,
    az: f.az,
    gravX: f.gravX,
    gravY: f.gravY,
    gravZ: f.gravZ,
    gyroX: f.gyroX,
    gyroY: f.gyroX,
    gyroZ: f.gyroY,
  );
}

/// Outcome of replaying one outage.
class ScenarioResult {
  const ScenarioResult({
    required this.trueDistanceM,
    required this.predDistanceM,
    required this.exitErrorM,
    required this.bias,
  });

  final double trueDistanceM;
  final double predDistanceM;
  final double exitErrorM;
  final double bias;

  double get alongTrackDriftPct =>
      (predDistanceM - trueDistanceM).abs() / trueDistanceM * 100;
  double get exitDriftPct => exitErrorM / trueDistanceM * 100;

  /// SIH target: position error when GNSS returns under 10% of the
  /// distance driven without it.
  bool get passesSih => exitDriftPct < 10;
}

/// How positions are kept on the road during a replayed outage.
enum ReplayRoad {
  /// Snap along the ground-truth track (the benchmark protocol of
  /// src/simulate_blackout.py — assumes the road ahead is known exactly).
  truth,

  /// Follow the OpenStreetMap road graph, choosing branches by heading —
  /// what the app does live.
  osm,

  /// No road: integrate heading only.
  none,
}

/// Plays a [ReplayData] through [DrPipeline] one sample at a time, hiding
/// GNSS between [outageStart] and [outageEnd].
class ReplayRunner {
  ReplayRunner(
    IdnnV5 model,
    this.data, {
    int? outageStart,
    int? outageEnd,
    this.road = ReplayRoad.truth,
    RoadData? roads,
  }) : outageStart = outageStart ?? data.outageStart,
       outageEnd = outageEnd ?? data.outageEnd,
       plane = LocalTangentPlane(data.lat.first, data.lon.first),
       _frames = data.frames(),
       _pipeline = DrPipeline(
         model: model,
         idleBaseline: data.idleBaseline,
         learnIdleBaseline: false,
         toModelFrame: ReplayData.toModelFrame,
       ) {
    east = List.filled(data.length, 0);
    north = List.filled(data.length, 0);
    for (var i = 0; i < data.length; i++) {
      final (e, n) = plane.toLocal(data.lat[i], data.lon[i]);
      east[i] = e;
      north[i] = n;
    }
    switch (road) {
      case ReplayRoad.truth:
        final end = math.min(data.roadEnd, data.length);
        _pipeline.setRoadGuide(
          PolylineGuide(
            PolylineMatcher(
              east.sublist(this.outageStart, end),
              north.sublist(this.outageStart, end),
            ),
          ),
        );
      case ReplayRoad.osm:
        if (roads == null) throw ArgumentError('ReplayRoad.osm needs roads');
        _pipeline.setRoadGuide(GraphGuide(RoadGraph(roads, plane)));
      case ReplayRoad.none:
        break;
    }
  }

  final ReplayData data;
  final int outageStart;
  final int outageEnd;
  final ReplayRoad road;
  final LocalTangentPlane plane;

  /// Ground-truth track in the local plane.
  late final List<double> east;
  late final List<double> north;

  final List<SensorFrame> _frames;
  final DrPipeline _pipeline;
  final outputs = <DrOutput>[];
  int _next = 0;

  bool get done => _next >= data.length && outputs.length >= data.length;
  int get position => _next;

  bool isOutage(int i) => i >= outageStart && i <= outageEnd;

  /// Feeds up to [count] samples; returns outputs produced.
  List<DrOutput> step([int count = 1]) {
    final produced = <DrOutput>[];
    for (var k = 0; k < count && _next < data.length; k++, _next++) {
      final i = _next;
      final gnss = isOutage(i)
          ? null
          : GnssSample(
              speedMs: data.gtSpeed[i],
              headingDeg: data.gtHeading[i],
              east: east[i],
              north: north[i],
            );
      produced.addAll(_pipeline.push(_frames[i], gnss: gnss));
    }
    if (_next >= data.length) produced.addAll(_pipeline.flush());
    outputs.addAll(produced);
    return produced;
  }

  /// Runs to the end and scores the outage.
  ScenarioResult runAll() {
    while (!done) {
      step(256);
    }
    return score();
  }

  ScenarioResult score() {
    var trueDist = 0.0, predDist = 0.0;
    for (var i = outageStart; i <= outageEnd; i++) {
      final dt = i == 0 ? 0.1 : data.t[i] - data.t[i - 1];
      trueDist += data.gtSpeed[i] * dt;
      predDist += outputs[i].speed * dt;
    }
    final last = outputs[outageEnd];
    final de = (last.east ?? east[outageEnd]) - east[outageEnd];
    final dn = (last.north ?? north[outageEnd]) - north[outageEnd];
    return ScenarioResult(
      trueDistanceM: trueDist,
      predDistanceM: predDist,
      exitErrorM: math.sqrt(de * de + dn * dn),
      bias: outputs[outageStart].bias ?? 0,
    );
  }
}
