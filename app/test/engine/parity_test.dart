// Parity tests: the Dart engine must reproduce the Python v5 pipeline.
// Goldens come from tools/app_export/make_goldens.py.

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

void expectSeries(
  String name,
  List<double> actual,
  List<double> expected, {
  double abs = 1e-3,
  double rel = 1e-4,
}) {
  expect(actual.length, expected.length, reason: '$name length');
  var worst = 0.0;
  var worstAt = -1;
  for (var i = 0; i < expected.length; i++) {
    final err = (actual[i] - expected[i]).abs();
    final tol = abs + rel * expected[i].abs();
    if (err - tol > worst) {
      worst = err - tol;
      worstAt = i;
    }
  }
  expect(
    worstAt,
    -1,
    reason: worstAt < 0
        ? ''
        : '$name[$worstAt]: dart=${actual[worstAt]} python=${expected[worstAt]}',
  );
}

List<SensorFrame> _frames() {
  final acc = _mat(g['inputs']['acc']);
  final grav = _mat(g['inputs']['grav']);
  final gyro = _mat(g['inputs']['gyro']);
  return [
    for (var i = 0; i < acc.length; i++)
      SensorFrame(
        t: i * 0.1,
        ax: acc[i][0],
        ay: acc[i][1],
        az: acc[i][2],
        gravX: grav[i][0],
        gravY: grav[i][1],
        gravZ: grav[i][2],
        gyroX: gyro[i][0],
        gyroY: gyro[i][1],
        gyroZ: gyro[i][2],
      ),
  ];
}

List<FeatureFrame> _features() {
  final fx = FeatureExtractorV5(
    idleBaseline: (g['idle_baseline'] as num).toDouble(),
  );
  final out = <FeatureFrame>[];
  for (final f in _frames()) {
    final ff = fx.push(f);
    if (ff != null) out.add(ff);
  }
  out.add(fx.flush()!);
  return out;
}

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

  test('erf / gelu match reference values', () {
    expect(erf(0), closeTo(0, 2e-7));
    expect(erf(0.5), closeTo(0.5204998778, 2e-7));
    expect(erf(-1.5), closeTo(-0.9661051465, 2e-7));
    expect(gelu(1.0), closeTo(0.8413447461, 2e-7));
  });

  test('model loads with the expected shape', () {
    expect(model.inputSize, 378);
    expect(model.windowLength, 21);
    expect(model.parameterCount, 139137);
  });

  test('features match extract_features_v5 (all 18 × 900)', () {
    final dart = _features();
    final py = _mat(g['features']);
    expect(dart.length, py.length);
    for (var c = 0; c < FeatureFrame.count; c++) {
      expectSeries(
        'feature $c',
        [for (final f in dart) f.values[c]],
        [for (final r in py) r[c]],
        abs: 1e-5,
        rel: 1e-5,
      );
    }
  });

  test('model matches torch on probe windows', () {
    final xs = _mat(g['model_probe']['x']);
    final ys = _vec(g['model_probe']['y']);
    for (var i = 0; i < xs.length; i++) {
      expect(
        model.predict(Float64List.fromList(xs[i])),
        closeTo(ys[i], 1e-4),
        reason: 'probe $i',
      );
    }
  });

  test('raw speed series matches predict_v5', () {
    final est = SpeedEstimator(model);
    final raw = [for (final f in _features()) est.push(f)];
    expectSeries('raw', raw, _vec(g['raw']));
  });

  group('physics filters (each fed the Python output of the stage before)', () {
    late List<FeatureFrame> feats;
    setUpAll(() => feats = _features());

    test('kinematic gate', () {
      final gate = KinematicGate();
      final raw = _vec(g['raw']);
      expectSeries('gated', [
        for (var i = 0; i < raw.length; i++) gate.push(raw[i], feats[i].linAx),
      ], _vec(g['gated']));
    });

    test('feed-forward EKF', () {
      final ekf = FeedforwardEkf();
      final gated = _vec(g['gated']);
      expectSeries('ekf', [
        for (var i = 0; i < gated.length; i++)
          ekf.push(gated[i], feats[i].linAx),
      ], _vec(g['ekf']));
    });

    test('cruise lock', () {
      final cruise = CruiseLock();
      expect(cruise.windowPts, 15, reason: 'int(1.5 / 0.1) in Python');
      final ekf = _vec(g['ekf']);
      expectSeries('cruise', [
        for (var i = 0; i < ekf.length; i++)
          cruise.push(ekf[i], feats[i].linAx, feats[i].gyroZ),
      ], _vec(g['cruise']));
    });

    test('enhanced ZUPT (centred window) and AC variance', () {
      final zupt = EnhancedZupt();
      final cruise = _vec(g['cruise']);
      final out = <ZuptSample>[];
      for (var i = 0; i < cruise.length; i++) {
        final z = zupt.push(cruise[i], feats[i]);
        if (z != null) out.add(z);
      }
      expect(out.length, cruise.length - zupt.lookahead);
      out.addAll(zupt.flush());
      expect([
        for (final z in out) z.index,
      ], List.generate(cruise.length, (i) => i));
      expectSeries('enhanced', [
        for (final z in out) z.speed,
      ], _vec(g['enhanced']));
      expectSeries(
        'ac_var',
        [for (final z in out) z.acVariance],
        _vec(g['ac_var']),
        abs: 1e-6,
      );
    });
  });

  test('end-to-end DrPipeline reproduces raw and enhanced speed', () {
    final gnss = _vec(g['inputs']['gnss_speed']);
    final p = DrPipeline(
      model: model,
      idleBaseline: (g['idle_baseline'] as num).toDouble(),
      learnIdleBaseline: false,
    );
    final frames = _frames();
    final out = <DrOutput>[];
    for (var i = 0; i < frames.length; i++) {
      out.addAll(p.push(frames[i], gnss: GnssSample(speedMs: gnss[i])));
    }
    out.addAll(p.flush());
    expect([
      for (final o in out) o.index,
    ], List.generate(frames.length, (i) => i));
    expectSeries('pipeline raw', [
      for (final o in out) o.rawSpeed,
    ], _vec(g['raw']));
    expectSeries('pipeline enhanced', [
      for (final o in out) o.filteredSpeed,
    ], _vec(g['enhanced']));
    expect(out.every((o) => !o.blackout), isTrue);
  });

  test('GNSS calibrator: bias, braking guard and standstill', () {
    final enhanced = _vec(g['enhanced']);
    final gnss = _vec(g['inputs']['gnss_speed']);
    final acVar = _vec(g['ac_var']);
    for (final c in (g['calibration'] as List).cast<Map<String, dynamic>>()) {
      final entry = c['entry'] as int;
      final length = c['length'] as int;
      final cal = GnssCalibrator();
      for (var i = entry - 60; i < entry; i++) {
        cal.track(enhanced[i], gnss[i]);
      }
      final bias = cal.freeze();
      expect(
        bias,
        closeTo((c['bias'] as num).toDouble(), 1e-9),
        reason: 'entry $entry',
      );
      expectSeries('calibrated@$entry', [
        for (var i = entry; i < entry + length; i++)
          GnssCalibrator.apply(enhanced[i], bias, acVibration: acVar[i]),
      ], _vec(c['speed']));
    }
  });

  test('dead_reckon_2d and map_match_to_road', () {
    final speed = _vec(g['inputs']['gnss_speed']);
    final heading = _vec(g['inputs']['heading']);
    final dr = DeadReckoner(0, 0);
    final xs = <double>[0], ys = <double>[0];
    for (var i = 1; i < speed.length; i++) {
      dr.step(speed[i], heading[i], 0.1);
      xs.add(dr.east);
      ys.add(dr.north);
    }
    expectSeries('dr x', xs, _vec(g['dead_reckon']['x']), abs: 1e-6);
    expectSeries('dr y', ys, _vec(g['dead_reckon']['y']), abs: 1e-6);

    final mm = g['map_match'] as Map<String, dynamic>;
    final road = PolylineMatcher(_vec(mm['road_x']), _vec(mm['road_y']));
    final pts = [for (final d in _vec(mm['cum'])) road.at(d)];
    expectSeries('mm x', [for (final p in pts) p.$1], _vec(mm['x']), abs: 1e-6);
    expectSeries('mm y', [for (final p in pts) p.$2], _vec(mm['y']), abs: 1e-6);
  });

  test('percentile / median match numpy', () {
    expect(percentile([1, 2, 3, 4], 10), closeTo(1.3, 1e-12));
    expect(median([5, 1, 3]), 3);
    expect(median([4, 1, 3, 2]), 2.5);
  });

  test('model runs fast enough for 10 Hz on device', () {
    final w = Float64List(model.inputSize);
    final sw = Stopwatch()..start();
    for (var i = 0; i < 200; i++) {
      model.predict(w);
    }
    final perCallMs = sw.elapsedMicroseconds / 200 / 1000;
    // 10 Hz leaves 100 ms per sample; the model must use a tiny fraction.
    expect(
      perCallMs,
      lessThan(5),
      reason: '${perCallMs.toStringAsFixed(3)} ms/inference',
    );
    // ignore: avoid_print
    print('IDNN v5 inference: ${perCallMs.toStringAsFixed(3)} ms (host VM)');
  });
}
