import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/data/replay/replay.dart';
import 'package:idr_app/data/replay/simulation.dart';
import 'package:idr_app/engine/engine.dart';

void main() {
  late IdnnV5 model;
  late ReplayData data;
  late RoadData roads;

  setUpAll(() {
    model = IdnnV5.fromBytes(
      manifestJson: File('assets/models/idnn_v5.json').readAsStringSync(),
      weights: ByteData.sublistView(
        File('assets/models/idnn_v5.bin').readAsBytesSync(),
      ),
    );
    data = ReplayData.parse(
      File('assets/replays/medium_tunnel.json').readAsStringSync(),
    );
    roads = RoadData.fromJson(
      jsonDecode(
        File('assets/replays/medium_tunnel_roads.json').readAsStringSync(),
      ) as Map<String, dynamic>,
    );
  });

  /// Drives 60 s with GPS, then [tunnels] × (40 s in, 30 s out).
  SimulationRunner drive(SimRoad road, {int tunnels = 2}) {
    final sim = SimulationRunner(model, data, road: road, roads: roads);
    sim.step(600);
    for (var k = 0; k < tunnels; k++) {
      sim.enterTunnel();
      sim.step(400);
      sim.exitTunnel();
      sim.step(300);
    }
    return sim;
  }

  test('GPS is cut only while in the tunnel, and every run is scored', () {
    final sim = drive(SimRoad.route);
    expect(sim.tunnels, hasLength(2));
    for (final run in sim.tunnels) {
      expect(run.open, isFalse, reason: 'tunnel ${run.number} closed');
      expect(run.exitErrorM, isNotNull);
      expect(run.seconds, closeTo(40, 1));
      expect(run.trueDistanceM, greaterThan(50));
      expect(run.idrDistanceM, greaterThan(0));
      // ignore: avoid_print
      print(
        'route · tunnel ${run.number}: ${run.seconds.toStringAsFixed(0)} s, '
        '${run.trueDistanceM.round()} m true / ${run.idrDistanceM.round()} m IDR, '
        'exit ${run.exitErrorM!.toStringAsFixed(1)} m (${run.errorPct.toStringAsFixed(1)} %), '
        'max ${run.maxErrorM.toStringAsFixed(1)} m',
      );
    }
    // Outputs outside the tunnels are GPS; inside, dead reckoning.
    final blackout = sim.outputs.where((o) => o.blackout).length;
    expect(blackout, closeTo(800, 12));
  });

  test('a known route keeps the estimate on the road through the turns', () {
    // Not every stretch passes the SIH 10 % (the model misjudges speed on
    // some) — but following the route, the error comes only from distance,
    // never from drifting off the road, so it stays bounded.
    final sim = drive(SimRoad.route);
    for (final run in sim.tunnels) {
      final distanceGap = (run.idrDistanceM - run.trueDistanceM).abs();
      expect(
        run.exitErrorM!,
        lessThan(distanceGap + 15),
        reason: 'tunnel ${run.number}: error ≈ distance error along the road',
      );
      expect(run.errorPct, lessThan(30));
    }
  });

  test('raw dead reckoning (no road) still produces a position', () {
    final sim = drive(SimRoad.none, tunnels: 1);
    final run = sim.tunnels.single;
    expect(run.exitErrorM, isNotNull);
    // ignore: avoid_print
    print(
      'none  · tunnel 1: exit ${run.exitErrorM!.toStringAsFixed(1)} m '
      '(${run.errorPct.toStringAsFixed(1)} %)',
    );
  });

  test('the drive ending inside a tunnel closes and scores it', () {
    final sim = SimulationRunner(model, data, road: SimRoad.map, roads: roads);
    sim.step(data.length - 200);
    sim.enterTunnel();
    while (!sim.done) {
      sim.step(256);
    }
    expect(sim.inTunnel, isFalse);
    expect(sim.tunnels.single.lastOutput, isNotNull);
  });
}
