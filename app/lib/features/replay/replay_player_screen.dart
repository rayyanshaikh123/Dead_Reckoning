import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/benchmark.dart';
import '../../data/model_loader.dart';
import '../../data/replay/replay.dart';
import '../../data/replay/replay_repository.dart';
import '../../engine/engine.dart';

/// Plays a real Drive M outage through the on-device engine: the true route
/// in white, IDR's GPS-free estimate in orange, then a scorecard.
class ReplayPlayerScreen extends ConsumerWidget {
  const ReplayPlayerScreen({super.key, required this.id});

  final String id;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final model = ref.watch(speedModelProvider);
    final data = ref.watch(replayDataProvider(id));
    final roads = ref.watch(replayRoadsProvider(id));
    return Scaffold(
      body: switch ((model, data, roads)) {
        (
          AsyncData(value: final m),
          AsyncData(value: final d),
          AsyncData(value: final r),
        ) =>
          _Player(model: m.model, data: d, roads: r),
        (AsyncError(:final error), _, _) ||
        (_, AsyncError(:final error), _) ||
        (_, _, AsyncError(:final error)) => Center(
          child: Text(
            'Could not load replay\n$error',
            textAlign: TextAlign.center,
            style: IdrText.small,
          ),
        ),
        _ => const Center(
          child: CircularProgressIndicator(color: IdrColors.accent),
        ),
      },
    );
  }
}

class _Player extends StatefulWidget {
  const _Player({required this.model, required this.data, required this.roads});

  final IdnnV5 model;
  final ReplayData data;
  final RoadData roads;

  @override
  State<_Player> createState() => _PlayerState();
}

class _PlayerState extends State<_Player> with SingleTickerProviderStateMixin {
  static const _speeds = [1.0, 4.0, 16.0];
  static const _preRoll = 150; // start 15 s before the outage

  final _map = MapController();
  late final Ticker _ticker = createTicker(_onTick);
  late ReplayRunner _runner;
  late List<LatLng> _truth;
  bool _mapReady = false;
  bool _playing = true;
  int _speedIndex = 1;
  double _carry = 0;
  Duration _last = Duration.zero;
  ScenarioResult? _result;
  ReplayRoad _roadMode = ReplayRoad.osm;
  int _warmTarget = 0;

  ReplayData get _d => widget.data;
  BenchmarkScenario? get _bench =>
      benchmarkScenarios.where((b) => b.id == _d.id).firstOrNull;

  @override
  void initState() {
    super.initState();
    _truth = [for (var i = 0; i < _d.length; i++) LatLng(_d.lat[i], _d.lon[i])];
    _restart();
    _ticker.start();
  }

  void _restart() {
    _runner = ReplayRunner(
      widget.model,
      _d,
      road: _roadMode,
      roads: widget.roads,
    );
    // The lead-in (GNSS healthy) is run in chunks from the ticker so the UI
    // stays responsive while the filters and calibrator warm up.
    _warmTarget = math.max(_d.outageStart - _preRoll, 0);
    _result = null;
    _carry = 0;
    _playing = true;
  }

  void _onTick(Duration now) {
    final dt = (now - _last).inMicroseconds / 1e6;
    _last = now;
    if (_warming) {
      _runner.step(math.min(150, _warmTarget - _runner.position));
      if (!_warming && _mapReady) _map.move(_currentPoint(), _map.camera.zoom);
      setState(() {});
      return;
    }
    if (!_playing || _runner.done || dt <= 0 || dt > 0.5) return;
    _carry += dt * 10 * _speeds[_speedIndex];
    final n = _carry.floor();
    if (n == 0) return;
    _carry -= n;
    _runner.step(n);
    if (_result == null && _runner.outputs.length > _d.outageEnd + 1) {
      _result = _runner.score();
    }
    // Stop 30 s after GPS returns.
    if (_runner.outputs.length > math.min(_d.outageEnd + 300, _d.length - 1)) {
      _playing = false;
    }
    if (_mapReady) _map.move(_currentPoint(), _map.camera.zoom);
    setState(() {});
  }

  bool get _warming => _runner.position < _warmTarget;

  /// Timeline position of sample [i]; the bar starts where playback does.
  double _span(int i) =>
      ((i - _warmTarget) / math.max(_d.length - 1 - _warmTarget, 1)).clamp(
        0.0,
        1.0,
      );
  int get _k => math.max(_runner.outputs.length - 1, 0);
  bool get _inOutage => _runner.isOutage(_k);

  LatLng? _estimate(int i) {
    if (i >= _runner.outputs.length) return null;
    final o = _runner.outputs[i];
    if (!o.blackout || o.east == null) return null;
    final (lat, lon) = _runner.plane.toGeo(o.east!, o.north!);
    return LatLng(lat, lon);
  }

  LatLng _currentPoint() => (_inOutage ? _estimate(_k) : null) ?? _truth[_k];

  double? get _liveErrorM {
    final o = _runner.outputs.isEmpty ? null : _runner.outputs[_k];
    if (o == null || !o.blackout || o.east == null) return null;
    final de = o.east! - _runner.east[_k], dn = o.north! - _runner.north[_k];
    return math.sqrt(de * de + dn * dn);
  }

  @override
  void dispose() {
    _ticker.dispose();
    _map.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final k = _k;
    final s = _d.outageStart, e = _d.outageEnd;
    final estimate = [for (var i = s; i <= math.min(k, e); i++) ?_estimate(i)];

    return Stack(
      children: [
        Positioned.fill(
          child: IdrMap(
            controller: _map,
            initialCenter: _truth[k],
            initialZoom: 16,
            onMapReady: () => _mapReady = true,
            layers: [
              PolylineLayer(
                polylines: [
                  Polyline(
                    points: _truth,
                    color: IdrColors.outline,
                    strokeWidth: 3,
                  ),
                  Polyline(
                    points: _truth.sublist(s, e + 1),
                    color: IdrColors.surfaceHigh.withValues(alpha: 0.9),
                    strokeWidth: 22,
                  ),
                  Polyline(
                    points: _truth.sublist(0, math.min(k, s) + 1),
                    color: IdrColors.textPrimary,
                    strokeWidth: 4,
                  ),
                  if (k > e)
                    Polyline(
                      points: _truth.sublist(e, k + 1),
                      color: IdrColors.textPrimary,
                      strokeWidth: 4,
                    ),
                  if (_inOutage || k > e)
                    Polyline(
                      points: _truth.sublist(s, math.min(k, e) + 1),
                      color: IdrColors.textPrimary.withValues(alpha: 0.5),
                      strokeWidth: 2,
                      pattern: StrokePattern.dashed(segments: const [6, 6]),
                    ),
                  if (estimate.length > 1)
                    Polyline(
                      points: estimate,
                      color: IdrColors.accent,
                      strokeWidth: 5,
                    ),
                ],
              ),
              MarkerLayer(
                markers: [
                  Marker(
                    point: _truth[k],
                    width: 20,
                    height: 20,
                    child: MapDot(
                      color: IdrColors.textPrimary,
                      hollow: _inOutage,
                    ),
                  ),
                  if (_inOutage && estimate.isNotEmpty)
                    Marker(
                      point: estimate.last,
                      width: 20,
                      height: 20,
                      child: const MapDot(color: IdrColors.accent),
                    ),
                ],
              ),
            ],
          ),
        ),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(IdrSpace.md),
            child: _Pill(
              child: BracketButton(
                'close',
                onTap: () => Navigator.of(context).maybePop(),
              ),
            ),
          ),
        ),
        Align(alignment: Alignment.bottomCenter, child: _sheet()),
      ],
    );
  }

  Widget _sheet() {
    final k = _k;
    final o = _runner.outputs.isEmpty ? null : _runner.outputs[k];
    final result = _result;
    final secondsIn = _inOutage ? ((k - _d.outageStart) / 10).floor() : 0;
    final toGo = ((_d.outageStart - k) / 10).ceil();

    final String title;
    final String caption;
    if (_warming) {
      title = 'Calibrating';
      caption =
          'Learning from the 5 minutes before the tunnel '
          '${(100 * _runner.position / math.max(_warmTarget, 1)).round()}%';
    } else if (_inOutage) {
      title =
          'No GPS · ${(secondsIn ~/ 60).toString().padLeft(2, '0')}:${(secondsIn % 60).toString().padLeft(2, '0')}';
      caption = 'IDR is estimating from motion sensors';
    } else if (k < _d.outageStart) {
      title = 'GPS';
      caption = 'Tunnel in $toGo s — GPS will be cut';
    } else {
      title = 'GPS back';
      caption = 'Compare where IDR thought you were';
    }

    return Container(
      decoration: const BoxDecoration(
        color: IdrColors.surface,
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(IdrRadius.sheet),
        ),
        boxShadow: [
          BoxShadow(
            color: Color(0x66000000),
            blurRadius: 30,
            offset: Offset(0, -6),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(
        IdrSpace.gutter,
        IdrSpace.xl,
        IdrSpace.gutter,
        0,
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    title,
                    style: IdrText.title.copyWith(
                      fontSize: 26,
                      color: _inOutage ? IdrColors.accent : null,
                    ),
                  ),
                ),
                BracketButton(
                  _roadMode == ReplayRoad.osm ? 'osm roads' : 'true road',
                  style: IdrText.small.copyWith(color: IdrColors.textPrimary),
                  onTap: () => setState(() {
                    _roadMode = _roadMode == ReplayRoad.osm
                        ? ReplayRoad.truth
                        : ReplayRoad.osm;
                    _restart();
                  }),
                ),
                const SizedBox(width: IdrSpace.md),
                BracketButton(
                  '${_speeds[_speedIndex].toStringAsFixed(0)}×',
                  onTap: () => setState(
                    () => _speedIndex = (_speedIndex + 1) % _speeds.length,
                  ),
                ),
              ],
            ),
            const SizedBox(height: IdrSpace.xs),
            Text('${_d.name} · $caption', style: IdrText.small),
            const SizedBox(height: IdrSpace.lg),
            _Timeline(
              progress: _span(k),
              outageStart: _span(_d.outageStart),
              outageEnd: _span(_d.outageEnd),
            ),
            const SizedBox(height: IdrSpace.lg),
            if (result != null)
              _Scorecard(result: result, bench: _bench)
            else
              _LiveStats(
                output: o,
                trueSpeed: _d.gtSpeed[k],
                errorM: _liveErrorM,
              ),
            const SizedBox(height: IdrSpace.lg),
            Row(
              children: [
                Expanded(
                  child: PrimaryButton(
                    _runner.done || (!_playing && result != null)
                        ? 'Replay'
                        : (_playing ? 'Pause' : 'Play'),
                    outlined: _playing,
                    onPressed: () => setState(() {
                      if (!_playing && result != null) {
                        _restart();
                      } else {
                        _playing = !_playing;
                      }
                    }),
                  ),
                ),
              ],
            ),
            const SizedBox(height: IdrSpace.lg),
          ],
        ),
      ),
    );
  }
}

class _LiveStats extends StatelessWidget {
  const _LiveStats({
    required this.output,
    required this.trueSpeed,
    required this.errorM,
  });

  final DrOutput? output;
  final double trueSpeed;
  final double? errorM;

  @override
  Widget build(BuildContext context) {
    final o = output;
    return Row(
      children: [
        Expanded(
          child: _Stat(
            label: 'IDR speed',
            value: o == null ? '--' : '${(o.speed * 3.6).round()} km/h',
            accent: o?.blackout ?? false,
          ),
        ),
        Expanded(
          child: _Stat(
            label: 'true speed',
            value: '${(trueSpeed * 3.6).round()} km/h',
          ),
        ),
        Expanded(
          child: _Stat(
            label: 'IDR off by',
            value: errorM == null ? '--' : '${errorM!.round()} m',
            accent: errorM != null,
          ),
        ),
      ],
    );
  }
}

class _Scorecard extends StatelessWidget {
  const _Scorecard({required this.result, this.bench});

  final ScenarioResult result;
  final BenchmarkScenario? bench;

  @override
  Widget build(BuildContext context) {
    final pass = result.passesSih;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: _Stat(
                label: 'exit error',
                value:
                    '${result.exitErrorM.round()} m · '
                    '${result.exitDriftPct.toStringAsFixed(1)}%',
                color: pass ? IdrColors.positive : IdrColors.negative,
              ),
            ),
            Expanded(
              child: _Stat(
                label: 'distance drift',
                value: '${result.alongTrackDriftPct.toStringAsFixed(1)}%',
              ),
            ),
            Expanded(
              child: _Stat(
                label: 'without AI',
                value: bench == null ? '--' : '${bench!.insDriftPct.round()}%',
                color: IdrColors.negative,
              ),
            ),
          ],
        ),
        const SizedBox(height: IdrSpace.sm),
        Text(
          pass
              ? 'Within 10% after ${result.trueDistanceM.round()} m without GPS — passes the SIH target.'
              : 'Over the 10% SIH target on this ${result.trueDistanceM.round()} m stretch.',
          style: IdrText.small.copyWith(
            color: pass ? IdrColors.positive : IdrColors.textSecondary,
          ),
        ),
      ],
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({
    required this.label,
    required this.value,
    this.accent = false,
    this.color,
  });

  final String label;
  final String value;
  final bool accent;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: IdrText.micro),
        const SizedBox(height: 2),
        Text(
          value,
          style: IdrText.body.copyWith(
            color: color ?? (accent ? IdrColors.accent : null),
          ),
        ),
      ],
    );
  }
}

/// Playback bar with the outage window marked in orange.
class _Timeline extends StatelessWidget {
  const _Timeline({
    required this.progress,
    required this.outageStart,
    required this.outageEnd,
  });

  final double progress;
  final double outageStart;
  final double outageEnd;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 14,
      child: CustomPaint(
        painter: _TimelinePainter(progress, outageStart, outageEnd),
        size: Size.infinite,
      ),
    );
  }
}

class _TimelinePainter extends CustomPainter {
  _TimelinePainter(this.p, this.s, this.e);

  final double p, s, e;

  @override
  void paint(Canvas canvas, Size size) {
    final y = size.height / 2;
    final track = Paint()
      ..color = IdrColors.surfaceHigh
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(Offset(0, y), Offset(size.width, y), track);
    canvas.drawLine(
      Offset(size.width * s, y),
      Offset(size.width * e, y),
      Paint()
        ..color = IdrColors.accent.withValues(alpha: 0.35)
        ..strokeWidth = 6,
    );
    canvas.drawLine(
      Offset(0, y),
      Offset(size.width * p, y),
      Paint()
        ..color = IdrColors.textPrimary
        ..strokeWidth = 2,
    );
    canvas.drawCircle(
      Offset(size.width * p, y),
      6,
      Paint()..color = IdrColors.textPrimary,
    );
  }

  @override
  bool shouldRepaint(_TimelinePainter old) => old.p != p;
}

class _Pill extends StatelessWidget {
  const _Pill({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
      decoration: BoxDecoration(
        color: IdrColors.background.withValues(alpha: 0.85),
        borderRadius: BorderRadius.circular(20),
      ),
      child: child,
    );
  }
}
