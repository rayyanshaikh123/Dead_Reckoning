// Reproduces the README blackout benchmark (IO-VNBD Drive M) with the Dart
// engine. Reference numbers come from tools/app_export/export_replay.py,
// which re-runs the Python pipeline on the same data.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/data/replay/replay.dart';
import 'package:idr_app/engine/engine.dart';

void main() {
  late IdnnV5 model;
  late Map<String, dynamic> index;

  setUpAll(() {
    model = IdnnV5.fromBytes(
      manifestJson: File('assets/models/idnn_v5.json').readAsStringSync(),
      weights: ByteData.sublistView(
        File('assets/models/idnn_v5.bin').readAsBytesSync(),
      ),
    );
    index = jsonDecode(
      File('assets/replays/index.json').readAsStringSync(),
    ) as Map<String, dynamic>;
  });

  for (final id in [
    'highway_tunnel',
    'short_underpass',
    'medium_tunnel',
    'mountain_tunnel',
    'city_canyon',
  ]) {
    test('Drive M · $id matches the Python benchmark', () {
      final data = ReplayData.parse(
        File('assets/replays/$id.json').readAsStringSync(),
      );
      final ref =
          ((index['scenarios'] as List).cast<Map<String, dynamic>>().firstWhere(
                (s) => s['id'] == id,
              ))['reference']
              as Map<String, dynamic>;
      double r(String k) => (ref[k] as num).toDouble();

      final runner = ReplayRunner(model, data);
      final res = runner.runAll();

      // ignore: avoid_print
      print(
        '${data.name.padRight(28)} dart: drift ${res.alongTrackDriftPct.toStringAsFixed(2)}% '
        'exit ${res.exitErrorM.toStringAsFixed(1)} m | python: drift ${r('along_track_drift_pct').toStringAsFixed(2)}% '
        'exit ${r('exit_error_m').toStringAsFixed(1)} m | bias ${res.bias.toStringAsFixed(3)} vs ${r('bias').toStringAsFixed(3)}',
      );

      // Filtered speed tracks Python once the filters have warmed up. The last
      // samples of the clip are excluded: ZUPT's centred window zero-pads past
      // the end of the clip, while Python still had the rest of the drive.
      var worst = 0.0;
      for (var i = 300; i < data.length - 5; i++) {
        final d = (runner.outputs[i].filteredSpeed - data.pythonEnhanced[i])
            .abs();
        if (d > worst) worst = d;
      }
      expect(
        worst,
        lessThan(0.05),
        reason: 'max |dart - python| filtered speed (m/s)',
      );

      expect(res.bias, closeTo(r('bias'), 0.01));
      expect(res.trueDistanceM, closeTo(r('true_distance_m'), 0.5));
      expect(res.alongTrackDriftPct, closeTo(r('along_track_drift_pct'), 1.0));
      expect(res.exitErrorM, closeTo(r('exit_error_m'), 1.0));
    });
  }

  // Realistic protocol: follow the OpenStreetMap road graph (what the app
  // does live) instead of snapping to the ground-truth track.
  for (final id in [
    'highway_tunnel',
    'short_underpass',
    'medium_tunnel',
    'mountain_tunnel',
    'city_canyon',
  ]) {
    test('Drive M · $id on OpenStreetMap roads passes the SIH target', () {
      final data = ReplayData.parse(
        File('assets/replays/$id.json').readAsStringSync(),
      );
      final roads = RoadData.fromJson(
        jsonDecode(File('assets/replays/${id}_roads.json').readAsStringSync())
            as Map<String, dynamic>,
      );
      final runner = ReplayRunner(
        model,
        data,
        road: ReplayRoad.osm,
        roads: roads,
      );
      final res = runner.runAll();
      final onRoad = runner.outputs.where((o) => o.blackout && o.onRoad);
      // ignore: avoid_print
      print(
        '${data.name.padRight(28)} osm: exit ${res.exitErrorM.toStringAsFixed(1)} m '
        '(${res.exitDriftPct.toStringAsFixed(1)}%)',
      );
      expect(onRoad.length, runner.outageEnd - runner.outageStart + 1);
      expect(res.exitDriftPct, lessThan(10));
    });
  }
}
