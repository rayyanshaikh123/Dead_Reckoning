import 'dart:math' as math;

import '../../engine/engine.dart';
import 'replay.dart';

/// How positions are kept on the road while in a simulated tunnel.
enum SimRoad {
  /// The route is known (as with turn-by-turn navigation): distance is laid
  /// along the road actually driven from the tunnel entrance.
  route,

  /// The OpenStreetMap road network, branches chosen by heading — what the
  /// app does live with no route set.
  map,

  /// No road at all: speed × heading integration only (raw dead reckoning).
  none,
}

/// One trip through a simulated tunnel, scored against the recorded truth.
class TunnelRun {
  TunnelRun(this.number, this.enteredAt);

  final int number;

  /// Input sample at which GPS was cut / restored.
  final int enteredAt;
  int? exitedAt;

  /// First and last engine output of the outage (outputs lag inputs by 0.5 s).
  int? firstOutput;
  int? lastOutput;

  /// Distance actually driven, and IDR's estimate of it (m).
  double trueDistanceM = 0;
  double idrDistanceM = 0;

  /// Straight-line gap between IDR's position and the truth (m).
  double liveErrorM = 0;
  double maxErrorM = 0;

  /// Gap when GPS came back — the SIH metric (m).
  double? exitErrorM;

  bool get open => exitErrorM == null;
  double get seconds => firstOutput == null || lastOutput == null
      ? 0
      : (lastOutput! - firstOutput! + 1) / 10;

  double get distanceDriftPct => trueDistanceM < 1
      ? 0
      : (idrDistanceM - trueDistanceM).abs() / trueDistanceM * 100;

  /// Position error as a share of the distance driven without GPS.
  double get errorPct {
    final e = exitErrorM ?? liveErrorM;
    return trueDistanceM < 1 ? 0 : e / trueDistanceM * 100;
  }

  /// Smart India Hackathon target: under 10 % when GPS returns.
  bool get passesSih => errorPct < 10;
}

/// A recorded drive played through the engine, with GPS cut whenever the
/// user drives "into a tunnel" and restored when they drive out — any time,
/// as often as they like. Everything shown is measured against the
/// recording's own GPS track (the truth).
class SimulationRunner {
  SimulationRunner(
    IdnnV5 model,
    this.data, {
    this.road = SimRoad.route,
    RoadData? roads,
  }) : plane = LocalTangentPlane(data.lat.first, data.lon.first),
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
    if (road == SimRoad.map) {
      if (roads == null) throw ArgumentError('SimRoad.map needs roads');
      _pipeline.setRoadGuide(GraphGuide(RoadGraph(roads, plane)));
    }
  }

  final ReplayData data;
  final SimRoad road;
  final LocalTangentPlane plane;

  /// The recorded (true) track in the local plane, metres.
  late final List<double> east;
  late final List<double> north;

  final List<SensorFrame> _frames;
  final DrPipeline _pipeline;
  final outputs = <DrOutput>[];
  final tunnels = <TunnelRun>[];
  int _next = 0;
  bool _tunnel = false;

  /// GPS is being withheld (the car is in the tunnel).
  bool get inTunnel => _tunnel;
  int get position => _next;
  bool get done => _next >= data.length;
  TunnelRun? get current =>
      tunnels.isNotEmpty && tunnels.last.open ? tunnels.last : null;

  /// Latest engine output.
  DrOutput? get latest => outputs.isEmpty ? null : outputs.last;

  /// Drive into a tunnel: GPS stops from the next sample.
  void enterTunnel() {
    if (_tunnel || done) return;
    _tunnel = true;
    if (road == SimRoad.route) {
      // The route ahead, starting at the last GPS fix (the sample before
      // this one): the engine starts dead reckoning from that fix, and the
      // guide lays distance along the route from its first point.
      final from = math.max(0, _next - 1);
      _pipeline.setRoadGuide(
        PolylineGuide(PolylineMatcher(east.sublist(from), north.sublist(from))),
      );
    }
    tunnels.add(TunnelRun(tunnels.length + 1, _next));
  }

  /// Drive out: GPS is back from the next sample.
  void exitTunnel() {
    if (!_tunnel) return;
    _tunnel = false;
    tunnels.last.exitedAt = _next;
  }

  /// Feeds up to [count] samples; returns the outputs produced.
  List<DrOutput> step([int count = 1]) {
    final produced = <DrOutput>[];
    for (var k = 0; k < count && _next < data.length; k++, _next++) {
      final i = _next;
      final gnss = _tunnel
          ? null
          : GnssSample(
              speedMs: data.gtSpeed[i],
              headingDeg: data.gtHeading[i],
              east: east[i],
              north: north[i],
            );
      produced.addAll(_pipeline.push(_frames[i], gnss: gnss));
    }
    if (done) {
      if (_tunnel) exitTunnel();
      produced.addAll(_pipeline.flush());
    }
    for (final o in produced) {
      _score(o);
    }
    outputs.addAll(produced);
    return produced;
  }

  void _score(DrOutput o) {
    final run = tunnels.isEmpty ? null : tunnels.last;
    if (run == null) return;
    if (o.blackout) {
      run.firstOutput ??= o.index;
      run.lastOutput = o.index;
      final dt = o.index == 0 ? 0.1 : data.t[o.index] - data.t[o.index - 1];
      run.trueDistanceM += data.gtSpeed[o.index] * dt;
      run.idrDistanceM = o.distanceSinceEntry;
      if (o.east != null && o.north != null) {
        final de = o.east! - east[o.index], dn = o.north! - north[o.index];
        run.liveErrorM = math.sqrt(de * de + dn * dn);
        run.maxErrorM = math.max(run.maxErrorM, run.liveErrorM);
      }
    } else if (o.exitErrorM != null && run.open) {
      run.exitErrorM = o.exitErrorM;
    } else if (run.open && run.firstOutput != null && !o.blackout) {
      // Came out before a position existed (no GPS fix ever): score the
      // last live error.
      run.exitErrorM = run.liveErrorM;
    }
  }

  /// IDR's position (while in a tunnel) or the GPS position, as lat/lon.
  (double, double) positionOf(int i) {
    final o = i < outputs.length ? outputs[i] : null;
    if (o != null && o.blackout && o.east != null && o.north != null) {
      return plane.toGeo(o.east!, o.north!);
    }
    return (data.lat[i], data.lon[i]);
  }
}
