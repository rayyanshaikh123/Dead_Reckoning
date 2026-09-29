// Behaviour tests for the streaming pipeline (blackout handling, heading,
// idle baseline) that the Python goldens don't cover directly.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/engine/engine.dart';

late Map<String, dynamic> g;
late IdnnV5 model;

List<double> _vec(Object? v) =>
    (v as List).map((e) => (e as num).toDouble()).toList();
List<List<double>> _mat(Object? v) => (v as List).map(_vec).toList();

SensorFrame _frame(int i, List<double> a, List<double> gr, List<double> gy) =>
    SensorFrame(
      t: i * 0.1,
      ax: a[0],
      ay: a[1],
      az: a[2],
      gravX: gr[0],
      gravY: gr[1],
      gravZ: gr[2],
      gyroX: gy[0],
      gyroY: gy[1],
      gyroZ: gy[2],
    );

void main() {
  setUpAll(() {
    g = jsonDecode(
      File('test/golden/v5_pipeline.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    model = IdnnV5.fromBytes(
      manifestJson: File('assets/models/idnn_v5.json').readAsStringSync(),
      weights: ByteData.sublistView(
        File('assets/models/idnn_v5.bin').readAsBytesSync(),
      ),
    );
  });

  test(
    'blackout: bias frozen at entry, distance integrates, exit error reported',
    () {
      final acc = _mat(g['inputs']['acc']);
      final grav = _mat(g['inputs']['grav']);
      final gyro = _mat(g['inputs']['gyro']);
      final speed = _vec(g['inputs']['gnss_speed']);
      const entry = 400, exit = 550;

      // GNSS track heading due north so positions are easy to reason about.
      final north = <double>[0];
      for (var i = 1; i < speed.length; i++) {
        north.add(north.last + speed[i] * 0.1);
      }

      final p = DrPipeline(
        model: model,
        idleBaseline: (g['idle_baseline'] as num).toDouble(),
        learnIdleBaseline: false,
      );
      final out = <DrOutput>[];
      for (var i = 0; i < speed.length; i++) {
        final gnss = (i >= entry && i < exit)
            ? null
            : GnssSample(
                speedMs: speed[i],
                headingDeg: 0,
                east: 0,
                north: north[i],
              );
        out.addAll(p.push(_frame(i, acc[i], grav[i], gyro[i]), gnss: gnss));
      }
      out.addAll(p.flush());
      expect(out.length, speed.length);

      final inBlackout = out.where((o) => o.blackout).toList();
      expect(inBlackout.first.index, entry);
      expect(inBlackout.last.index, exit - 1);

      // Bias = mean(model - gnss) over the 50 samples before entry.
      final cal = GnssCalibrator();
      for (var i = entry - 50; i < entry; i++) {
        cal.track(out[i].filteredSpeed, speed[i]);
      }
      expect(inBlackout.first.bias, closeTo(cal.freeze(), 1e-12));
      expect(
        inBlackout.map((o) => o.bias).toSet().length,
        1,
        reason: 'bias stays frozen',
      );

      // Distance is the integral of the calibrated speed.
      final integrated = inBlackout.fold<double>(
        0,
        (s, o) => s + o.speed * 0.1,
      );
      expect(inBlackout.last.distanceSinceEntry, closeTo(integrated, 1e-9));
      expect(inBlackout.last.blackoutSamples, exit - entry);

      // Exit error is reported once, on the first sample back on GNSS.
      final back = out[exit];
      expect(back.blackout, isFalse);
      expect(back.exitErrorM, isNotNull);
      expect(out.where((o) => o.exitErrorM != null).length, 1);
    },
  );

  test('road snapping places the car along the polyline during a blackout', () {
    final acc = _mat(g['inputs']['acc']);
    final grav = _mat(g['inputs']['grav']);
    final gyro = _mat(g['inputs']['gyro']);
    final speed = _vec(g['inputs']['gnss_speed']);
    final p = DrPipeline(model: model, learnIdleBaseline: false)
      // A straight road heading east from the origin.
      ..setRoadGuide(PolylineGuide(PolylineMatcher([0, 5000], [0, 0])));
    DrOutput? last;
    for (var i = 0; i < 500; i++) {
      final gnss = i < 300
          ? GnssSample(speedMs: speed[i], headingDeg: 90, east: 0, north: 0)
          : null;
      for (final o in p.push(_frame(i, acc[i], grav[i], gyro[i]), gnss: gnss)) {
        last = o;
      }
    }
    expect(last!.blackout, isTrue);
    expect(last.north, closeTo(0, 1e-9));
    expect(last.east, closeTo(last.distanceSinceEntry, 1e-9));
  });

  test('heading tracker: positive yaw about "up" is a left turn', () {
    final h = HeadingTracker()..anchor(90);
    const f = SensorFrame(
      t: 0,
      ax: 0,
      ay: 0,
      az: 9.81,
      gravX: 0,
      gravY: 0,
      gravZ: 9.81,
      gyroX: 0,
      gyroY: 0,
      gyroZ: 0.1,
    );
    for (var i = 0; i < 100; i++) {
      h.propagate(f);
    }
    // 0.1 rad/s for 10 s = 57.30° to the left.
    expect(h.headingDeg, closeTo(90 - 57.29578, 1e-3));
  });

  test('idle baseline is learned from stationary samples', () {
    final acc = _mat(g['inputs']['acc']);
    final grav = _mat(g['inputs']['grav']);
    final gyro = _mat(g['inputs']['gyro']);
    final speed = _vec(g['inputs']['gnss_speed']);
    final p = DrPipeline(model: model, idleBaseline: 5.0);
    expect(p.idleBaseline, 5.0);
    for (var i = 0; i < 80; i++) {
      p.push(
        _frame(i, acc[i], grav[i], gyro[i]),
        gnss: GnssSample(speedMs: speed[i]),
      );
    }
    // Car is stationary for the first 10 s; the learned σ_idle replaces 5.0.
    expect(p.idleBaseline, lessThan(1.0));
    expect(
      p.idleBaseline,
      greaterThanOrEqualTo(FeatureExtractorV5.minIdleBaseline),
    );
  });
}
