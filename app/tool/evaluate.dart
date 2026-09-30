// Full-drive evaluation of the IDR engine.
//
// Runs the app's own engine (lib/engine) over an entire IO-VNBD drive packed
// by tools/app_export/export_eval_pack.py, cutting GNSS at many random places
// (30/60/90/120 s outages, ≥ 100 m driven), and scores every outage:
//
//   osm    IDR following OpenStreetMap roads (what the app does live)
//   truth  IDR snapped to the ground-truth track (the README protocol)
//   none   IDR integrating heading only (no map)
//   hold   naive baseline: keep the last GPS speed and heading (what a
//          phone without dead reckoning does)
//
// Usage (from app/):
//   dart run tool/evaluate.dart                    # Drive M, 4 passes
//   dart run tool/evaluate.dart --passes 8 --seed 3
//
// Writes a Markdown + JSON report next to the pack (data/eval/).

import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:idr_app/data/replay/replay.dart';
import 'package:idr_app/engine/engine.dart';

const _durations = [30, 60, 90, 120]; // seconds
const _warmUp = 3000; // samples of GNSS before the first outage (5 min)
const _minGap = 900; // samples of GNSS between outages (90 s) to recalibrate
const _minDistanceM = 100.0;

// ---------------------------------------------------------------- data ----

class Pack {
  Pack._(this.header, this.cols);

  final Map<String, dynamic> header;
  final Map<String, Float64List> cols;

  int get n => header['n'] as int;
  String get drive => header['drive'] as String;
  double get idleBaseline => (header['idle_baseline'] as num).toDouble();
  Float64List operator [](String c) => cols[c]!;

  static Pack load(String path) {
    final bytes = File(path).readAsBytesSync();
    final nl = bytes.indexOf(10);
    final header =
        jsonDecode(utf8.decode(bytes.sublist(0, nl))) as Map<String, dynamic>;
    final n = header['n'] as int;
    final data = ByteData.sublistView(bytes, nl + 1);
    var off = 0;
    final cols = <String, Float64List>{};
    for (final c in (header['columns'] as List).cast<Map<String, dynamic>>()) {
      final out = Float64List(n);
      if (c['dtype'] == 'f8') {
        for (var i = 0; i < n; i++, off += 8) {
          out[i] = data.getFloat64(off, Endian.little);
        }
      } else {
        for (var i = 0; i < n; i++, off += 4) {
          out[i] = data.getFloat32(off, Endian.little);
        }
      }
      cols[c['name'] as String] = out;
    }
    return Pack._(header, cols);
  }

  SensorFrame frame(int i) => SensorFrame(
    t: this['t'][i],
    ax: this['ax'][i],
    ay: this['ay'][i],
    az: this['az'][i],
    gravX: this['grav_x'][i],
    gravY: this['grav_y'][i],
    gravZ: this['grav_z'][i],
    gyroX: this['gyro_x'][i],
    gyroY: this['gyro_y'][i],
    gyroZ: this['gyro_up'][i],
  );
}

IdnnV5 loadModel() => IdnnV5.fromBytes(
  manifestJson: File('assets/models/idnn_v5.json').readAsStringSync(),
  weights: ByteData.sublistView(
    File('assets/models/idnn_v5.bin').readAsBytesSync(),
  ),
);

RoadData loadRoads(String dir) {
  var all = const RoadData([]);
  for (final f in Directory(
    dir,
  ).listSync().whereType<File>().where((f) => f.path.endsWith('.json'))) {
    all = all.merge(
      RoadData.fromJson(
        jsonDecode(f.readAsStringSync()) as Map<String, dynamic>,
      ),
    );
  }
  return all;
}

// ------------------------------------------------------------- outages ----

class Outage {
  const Outage(this.start, this.end, this.seconds);

  final int start; // first sample without GNSS
  final int end; // last sample without GNSS (inclusive)
  final int seconds;

  Map<String, dynamic> toJson() => {
    'start': start,
    'end': end,
    'seconds': seconds,
  };
}

/// The five README tunnels (src/simulate_blackout.py) — `--fixed` runs these
/// instead of random outages, to check the evaluator against the benchmark.
const readmeOutages = [
  Outage(92100, 92400, 30),
  Outage(15000, 15300, 30),
  Outage(35000, 35600, 60),
  Outage(50000, 51200, 120),
  Outage(75000, 75900, 90),
];

/// Non-overlapping random outages over the whole drive, reproducible from
/// [seed]. Each needs ≥ 100 m of true driving and ≥ 90 s of GNSS before it.
/// Seed 0 means the fixed [readmeOutages].
List<Outage> scheduleOutages(Pack p, int seed) {
  if (seed == 0) return readmeOutages;
  final rnd = math.Random(seed);
  final speed = p['gt_speed'];
  final out = <Outage>[];
  var i = _warmUp + rnd.nextInt(1200);
  while (true) {
    final secs = _durations[rnd.nextInt(_durations.length)];
    final end = i + secs * 10 - 1;
    if (end + 300 >= p.n) break;
    var dist = 0.0;
    for (var k = i; k <= end; k++) {
      dist += speed[k] * 0.1;
    }
    if (dist >= _minDistanceM) {
      out.add(Outage(i, end, secs));
      i = end + _minGap + rnd.nextInt(2400);
    } else {
      i += 300; // stopped here; try a bit later
    }
  }
  return out;
}

// ---------------------------------------------------------------- runs ----

class PassTask {
  const PassTask(
    this.packPath,
    this.roadsDir,
    this.mode,
    this.seed, [
    this.tune = const {},
  ]);

  final String packPath;
  final String roadsDir;
  final String mode;
  final int seed;

  /// RoadTracker parameter overrides (`--tune driftLearnM=90,pruneMargin=15`).
  final Map<String, double> tune;
}

RoadTracker _tracker(
  RoadGraph g,
  List<(EdgePosition, double)> starts,
  Map<String, double> t,
) {
  // Defaults match RoadTracker's.
  final lo = t['scaleMin'] ?? 0.85, hi = t['scaleMax'] ?? 1.15;
  final steps = (t['scaleSteps'] ?? 7).round();
  return RoadTracker(
    g,
    starts,
    scales: [
      for (var i = 0; i < steps; i++)
        steps == 1 ? 1.0 : lo + (hi - lo) * i / (steps - 1),
    ],
    scalePriorCost: t['scalePriorCost'] ?? 0.6,
    scaleCostPer10m: t['scaleCostPer10m'] ?? 0,
    maxHypotheses: (t['maxHypotheses'] ?? 40).round(),
    pruneMargin: t['pruneMargin'] ?? 10,
    headingScale: t['headingScale'] ?? 30,
    maxMismatchDeg: t['maxMismatchDeg'] ?? 60,
    wayChangeCost: t['wayChangeCost'] ?? 0.8,
    rankDropCost: t['rankDropCost'] ?? 0.4,
    driftLearnM: t['driftLearnM'] ?? 60,
    maxDriftDeg: t['maxDriftDeg'] ?? 45,
  );
}

class OutageResult {
  const OutageResult({
    required this.outage,
    required this.trueDistanceM,
    required this.predDistanceM,
    required this.exitErrorM,
    required this.onRoadShare,
    this.crossTrackM = double.nan,
    this.headingErrDeg = double.nan,
  });

  final Outage outage;
  final double trueDistanceM;
  final double predDistanceM;
  final double exitErrorM;
  final double onRoadShare;

  /// Distance from the exit estimate to the nearest point of the true route
  /// (how far off the road it ended up, regardless of distance error). Large
  /// values mean IDR followed the wrong road.
  final double crossTrackM;

  bool get wrongRoad => crossTrackM > 25;

  /// Mean |gyro heading − true heading| while moving during the outage.
  final double headingErrDeg;

  double get alongDriftPct =>
      (predDistanceM - trueDistanceM).abs() / trueDistanceM * 100;
  double get exitPct => exitErrorM / trueDistanceM * 100;

  Map<String, dynamic> toJson() => {
    ...outage.toJson(),
    'true_distance_m': trueDistanceM,
    'pred_distance_m': predDistanceM,
    'exit_error_m': exitErrorM,
    'on_road_share': onRoadShare,
    'cross_track_m': _finite(crossTrackM),
    'heading_err_deg': _finite(headingErrDeg),
  };
}

class PassResult {
  const PassResult(this.mode, this.seed, this.outages, this.speed);

  final String mode;
  final int seed;
  final List<OutageResult> outages;

  /// Whole-drive speed accuracy (only filled by one pass).
  final Map<String, double>? speed;
}

/// Local plane centred on the drive, so distortion stays small everywhere.
LocalTangentPlane _plane(Pack p) {
  final lat = p['lat'], lon = p['lon'];
  var la = 0.0, lo = 0.0;
  for (var i = 0; i < p.n; i += 100) {
    la += lat[i];
    lo += lon[i];
  }
  final k = (p.n + 99) ~/ 100;
  return LocalTangentPlane(la / k, lo / k);
}

PassResult runPass(PassTask task) {
  final p = Pack.load(task.packPath);
  final model = loadModel();
  final plane = _plane(p);
  final n = p.n;
  final east = Float64List(n), north = Float64List(n);
  for (var i = 0; i < n; i++) {
    final (e, no) = plane.toLocal(p['lat'][i], p['lon'][i]);
    east[i] = e;
    north[i] = no;
  }
  final outages = scheduleOutages(p, task.seed);
  final inOutage = Uint8List(n);
  final outageAt = <int, Outage>{};
  for (final o in outages) {
    inOutage.fillRange(o.start, o.end + 1, 1);
    outageAt[o.start] = o;
  }

  final pipeline = DrPipeline(
    model: model,
    idleBaseline: p.idleBaseline,
    learnIdleBaseline: false,
    toModelFrame: ReplayData.toModelFrame,
  );
  if (task.mode == 'osm') {
    pipeline.setRoadGuide(
      GraphGuide(
        RoadGraph(loadRoads(task.roadsDir), plane),
        newTracker: (g, s) => _tracker(g, s, task.tune),
      ),
    );
  }

  final speed = Float64List(n), filtered = Float64List(n);
  final oe = Float64List(n), on = Float64List(n);
  final onRoad = Uint8List(n);
  final hd = Float64List(n);
  void keep(DrOutput o) {
    speed[o.index] = o.speed;
    filtered[o.index] = o.filteredSpeed;
    oe[o.index] = o.east ?? double.nan;
    on[o.index] = o.north ?? double.nan;
    onRoad[o.index] = o.onRoad ? 1 : 0;
    hd[o.index] = o.headingDeg ?? double.nan;
  }

  final gt = p['gt_speed'], heading = p['gt_heading'];
  for (var i = 0; i < n; i++) {
    final o = outageAt[i];
    if (task.mode == 'truth' && o != null) {
      final end = math.min(o.end + 250, n);
      pipeline.setRoadGuide(
        PolylineGuide(
          PolylineMatcher(
            east.sublist(o.start, end),
            north.sublist(o.start, end),
          ),
        ),
      );
    }
    final gnss = inOutage[i] == 1
        ? null
        : GnssSample(
            speedMs: gt[i],
            headingDeg: heading[i],
            east: east[i],
            north: north[i],
          );
    pipeline.push(p.frame(i), gnss: gnss).forEach(keep);
  }
  pipeline.flush().forEach(keep);

  final results = <OutageResult>[];
  final t = p['t'];
  for (final o in outages) {
    var trueD = 0.0, predD = 0.0, road = 0;
    for (var k = o.start; k <= o.end; k++) {
      final dt = t[k] - t[k - 1];
      trueD += gt[k] * dt;
      predD += speed[k] * dt;
      road += onRoad[k];
    }
    final de = oe[o.end] - east[o.end], dn = on[o.end] - north[o.end];
    var herr = 0.0, hn = 0;
    for (var k = o.start; k <= o.end; k++) {
      if (gt[k] < 3 || hd[k].isNaN) continue;
      herr += ((hd[k] - heading[k] + 540) % 360 - 180).abs();
      hn++;
    }
    var cross = double.infinity;
    for (var k = o.start; k < math.min(o.end + 600, n); k++) {
      final ce = oe[o.end] - east[k], cn = on[o.end] - north[k];
      cross = math.min(cross, math.sqrt(ce * ce + cn * cn));
    }
    results.add(
      OutageResult(
        outage: o,
        trueDistanceM: trueD,
        predDistanceM: predD,
        exitErrorM: math.sqrt(de * de + dn * dn),
        onRoadShare: road / (o.end - o.start + 1),
        crossTrackM: cross,
        headingErrDeg: hn == 0 ? double.nan : herr / hn,
      ),
    );
  }

  // Whole-drive speed accuracy and Dart-vs-Python parity (mode-independent).
  Map<String, double>? speedStats;
  if (task.mode == 'none') {
    final py = p['py_enhanced'];
    var se = 0.0, ae = 0.0, cnt = 0, parity = 0.0;
    for (var i = 300; i < n - 5; i++) {
      parity = math.max(parity, (filtered[i] - py[i]).abs());
      if (gt[i] < 1) continue;
      final e = filtered[i] - gt[i];
      se += e * e;
      ae += e.abs();
      cnt++;
    }
    speedStats = {
      'rmse_kmh': math.sqrt(se / cnt) * 3.6,
      'mae_kmh': ae / cnt * 3.6,
      'parity_max_diff_ms': parity,
    };
  }
  return PassResult(task.mode, task.seed, results, speedStats);
}

/// "Hold last GPS speed and heading" baseline for the same outages.
List<OutageResult> holdBaseline(Pack p, int seed) {
  final plane = _plane(p);
  final gt = p['gt_speed'], heading = p['gt_heading'], t = p['t'];
  final out = <OutageResult>[];
  for (final o in scheduleOutages(p, seed)) {
    final v0 = gt[o.start - 1];
    final h = heading[o.start - 1] * math.pi / 180;
    final (e0, n0) = plane.toLocal(
      p['lat'][o.start - 1],
      p['lon'][o.start - 1],
    );
    final (e1, n1) = plane.toLocal(p['lat'][o.end], p['lon'][o.end]);
    var trueD = 0.0, secs = 0.0;
    for (var k = o.start; k <= o.end; k++) {
      trueD += gt[k] * (t[k] - t[k - 1]);
      secs += t[k] - t[k - 1];
    }
    final predD = v0 * secs;
    final de = e0 + predD * math.sin(h) - e1,
        dn = n0 + predD * math.cos(h) - n1;
    out.add(
      OutageResult(
        outage: o,
        trueDistanceM: trueD,
        predDistanceM: predD,
        exitErrorM: math.sqrt(de * de + dn * dn),
        onRoadShare: 0,
      ),
    );
  }
  return out;
}

// -------------------------------------------------------------- report ----

double _pct(List<double> v, double q) {
  if (v.isEmpty) return double.nan;
  final s = [...v]..sort();
  final pos = (s.length - 1) * q;
  final lo = pos.floor(), hi = pos.ceil();
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

class Summary {
  Summary(this.rs);

  final List<OutageResult> rs;

  int get count => rs.length;
  List<double> get _exitPct => [for (final r in rs) r.exitPct];
  List<double> get _exitM => [for (final r in rs) r.exitErrorM];
  List<double> get _along => [for (final r in rs) r.alongDriftPct];
  double get medianExitPct => _pct(_exitPct, 0.5);
  double get p90ExitPct => _pct(_exitPct, 0.9);
  double get maxExitPct => _pct(_exitPct, 1);
  double get medianExitM => _pct(_exitM, 0.5);
  double get medianAlongPct => _pct(_along, 0.5);
  double get medianCrossM => _pct([for (final r in rs) r.crossTrackM], 0.5);
  double get wrongRoadRate => rs.isEmpty
      ? double.nan
      : rs.where((r) => r.wrongRoad).length / rs.length * 100;
  double get passRate => rs.isEmpty
      ? double.nan
      : rs.where((r) => r.exitPct < 10).length / rs.length * 100;

  Map<String, dynamic> toJson() => {
    'outages': count,
    'median_exit_pct': medianExitPct,
    'p90_exit_pct': p90ExitPct,
    'max_exit_pct': maxExitPct,
    'median_exit_m': medianExitM,
    'median_along_track_pct': medianAlongPct,
    'sih_pass_rate_pct': passRate,
    'median_cross_track_m': _finite(medianCrossM),
    'wrong_road_rate_pct': _finite(wrongRoadRate),
  };
}

/// JSON has no NaN/Infinity: write them as null.
double? _finite(double v) => v.isFinite ? v : null;

String _f(double v, [int d = 1]) => v.isNaN ? '–' : v.toStringAsFixed(d);

const _labels = {
  'osm': 'IDR · OSM roads (live app)',
  'truth': 'IDR · true route (README method)',
  'none': 'IDR · heading only (no map)',
  'hold': 'Hold last GPS speed (no IDR)',
};

String markdown(
  Pack p,
  Map<String, List<OutageResult>> byMode,
  Map<String, double>? speed,
  int passes,
  int seed,
) {
  final km = p['gt_speed'].fold<double>(0, (s, v) => s + v * 0.1) / 1000;
  final b = StringBuffer()
    ..writeln('# IDR full-drive evaluation — Drive ${p.drive}')
    ..writeln()
    ..writeln(
      '${(p['t'][p.n - 1] / 3600).toStringAsFixed(2)} h, ${km.toStringAsFixed(1)} km, '
      '${byMode.values.first.length} random GPS outages ($passes passes, seed $seed; '
      '30/60/90/120 s, ≥ 100 m driven, ≥ 90 s of GPS between outages).',
    )
    ..writeln()
    ..writeln(
      '**Exit error** = distance between the estimate and the true position when GPS returns; '
      '**%** = share of the distance driven without GPS. SIH target: < 10%.',
    )
    ..writeln()
    ..writeln(
      '| Method | Outages | Median exit | Median exit % | 90th pct exit % | Worst exit % | SIH pass rate | Median distance drift % | Ended on wrong road |',
    )
    ..writeln('|---|---|---|---|---|---|---|---|---|');
  for (final m in ['osm', 'truth', 'none', 'hold']) {
    final rs = byMode[m];
    if (rs == null) continue;
    final s = Summary(rs);
    b.writeln(
      '| ${_labels[m]} | ${s.count} | ${_f(s.medianExitM, 0)} m | ${_f(s.medianExitPct)}% | '
      '${_f(s.p90ExitPct)}% | ${_f(s.maxExitPct)}% | **${_f(s.passRate, 0)}%** | ${_f(s.medianAlongPct)}% | '
      '${m == 'hold' ? '–' : '${_f(s.wrongRoadRate, 0)}%'} |',
    );
  }
  b
    ..writeln()
    ..writeln('## By outage length (SIH pass rate / median exit %)')
    ..writeln()
    ..writeln('| Method | ${_durations.map((d) => '$d s').join(' | ')} |')
    ..writeln('|---|${_durations.map((_) => '---').join('|')}|');
  for (final m in ['osm', 'truth', 'none', 'hold']) {
    final rs = byMode[m];
    if (rs == null) continue;
    final cells = [
      for (final d in _durations)
        () {
          final s = Summary(rs.where((r) => r.outage.seconds == d).toList());
          return '${_f(s.passRate, 0)}% / ${_f(s.medianExitPct)}% (n=${s.count})';
        }(),
    ];
    b.writeln('| ${_labels[m]} | ${cells.join(' | ')} |');
  }
  final osm = byMode['osm'];
  if (osm != null && osm.isNotEmpty) {
    final worst = [...osm]..sort((a, c) => c.exitPct.compareTo(a.exitPct));
    b
      ..writeln()
      ..writeln('## Worst OSM outages (to inspect)')
      ..writeln()
      ..writeln('| Starts at | Length | Distance | Exit error | On road |')
      ..writeln('|---|---|---|---|---|');
    for (final r in worst.take(8)) {
      final t0 = p['t'][r.outage.start];
      b.writeln(
        '| ${(t0 ~/ 60)}:${(t0 % 60).round().toString().padLeft(2, '0')} (sample ${r.outage.start}) | '
        '${r.outage.seconds} s | ${r.trueDistanceM.round()} m | ${r.exitErrorM.round()} m (${_f(r.exitPct)}%) | '
        '${(r.onRoadShare * 100).round()}% |',
      );
    }
  }
  if (speed != null) {
    b
      ..writeln()
      ..writeln(
        '## Whole-drive speed (model + filters vs true speed, while moving)',
      )
      ..writeln()
      ..writeln(
        '- RMSE ${_f(speed['rmse_kmh']!)} km/h, mean absolute error ${_f(speed['mae_kmh']!)} km/h',
      )
      ..writeln(
        '- Dart vs Python filtered speed over the whole drive: max difference '
        '${speed['parity_max_diff_ms']!.toStringAsExponential(1)} m/s',
      );
  }
  return b.toString();
}

// ---------------------------------------------------------------- main ----

Future<void> main(List<String> argv) async {
  String arg(String name, String def) {
    final i = argv.indexOf('--$name');
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
  }

  final packPath = arg('pack', '../data/eval/drive_m.idreval');
  final roadsDir = arg('roads', '../data/eval/roads');
  final passes = int.parse(arg('passes', '4'));
  final seed = int.parse(arg('seed', '1'));
  final modes = arg('modes', 'osm,truth,none').split(',');
  final tune = <String, double>{
    for (final kv in arg('tune', '').split(',').where((e) => e.contains('=')))
      kv.split('=')[0]: double.parse(kv.split('=')[1]),
  };

  if (!File(packPath).existsSync()) {
    stderr.writeln(
      'No pack at $packPath — run tools/app_export/export_eval_pack.py first.',
    );
    exit(1);
  }
  final pack = Pack.load(packPath);
  final fixed = argv.contains('--fixed');
  final seeds = fixed
      ? [0]
      : [for (var k = 0; k < passes; k++) seed * 1000 + k];
  final tasks = [
    for (final m in modes)
      for (final s in seeds) PassTask(packPath, roadsDir, m, s, tune),
  ];
  final workers = math.max(
    1,
    math.min(Platform.numberOfProcessors - 1, tasks.length),
  );
  stdout.writeln(
    'Drive ${pack.drive}: ${pack.n} samples · ${tasks.length} runs on $workers cores',
  );

  final sw = Stopwatch()..start();
  final results = <PassResult>[];
  var next = 0;
  Future<void> worker() async {
    while (next < tasks.length) {
      final t = tasks[next++];
      final r = await Isolate.run(() => runPass(t));
      results.add(r);
      stdout.writeln(
        '  ${r.mode.padRight(5)} seed ${r.seed}: ${r.outages.length} outages  [${sw.elapsed.inSeconds} s]',
      );
    }
  }

  await Future.wait([for (var w = 0; w < workers; w++) worker()]);

  final byMode = <String, List<OutageResult>>{
    for (final m in modes)
      m: [for (final r in results.where((r) => r.mode == m)) ...r.outages],
    'hold': [for (final s in seeds) ...holdBaseline(pack, s)],
  };
  final speed = results
      .map((r) => r.speed)
      .whereType<Map<String, double>>()
      .firstOrNull;

  final report = markdown(pack, byMode, speed, passes, seed);
  stdout
    ..writeln()
    ..writeln(report);

  final stamp = DateTime.now()
      .toIso8601String()
      .substring(0, 16)
      .replaceAll(':', '');
  final base = '${File(packPath).parent.path}/report_${pack.drive}_$stamp';
  File('$base.md').writeAsStringSync(report);
  File('$base.json').writeAsStringSync(
    const JsonEncoder.withIndent(' ').convert({
      'drive': pack.drive,
      'passes': passes,
      'seed': seed,
      'summary': {
        for (final e in byMode.entries) e.key: Summary(e.value).toJson(),
      },
      'speed': speed,
      'outages': {
        for (final e in byMode.entries)
          e.key: [for (final r in e.value) r.toJson()],
      },
    }),
  );
  stdout.writeln('Report: $base.md');
}
