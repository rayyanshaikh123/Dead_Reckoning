import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart' show LatLng;

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/model_loader.dart';
import '../../data/replay/replay.dart';
import '../../data/replay/replay_repository.dart';
import '../../data/replay/simulation.dart';
import '../../engine/engine.dart';

/// Routes to simulate: real recorded drives (IO-VNBD Drive M), so the AI
/// sees real sensor data. Ordered by how much they turn.
class SimRoute {
  const SimRoute(this.id, this.name, this.caption);

  final String id;
  final String name;
  final String caption;
}

const simRoutes = [
  SimRoute('short_underpass', 'Tight city loops', '2.5 km · 16 turns · 24 km/h'),
  SimRoute('medium_tunnel', 'Winding town road', '3.4 km · 15 turns · 31 km/h'),
  SimRoute('mountain_tunnel', 'Mountain road', '6.6 km · 9 bends · 53 km/h'),
  SimRoute('highway_tunnel', 'Highway', '5.7 km · 3 curves · 56 km/h'),
  SimRoute('city_canyon', 'Downtown, stop & go', '3.0 km · 10 stops · 26 km/h'),
];

/// Test simulation: a real drive plays on the map; the user opens and closes
/// "tunnels" (GPS off / on) whenever they like and watches IDR navigate
/// without GPS, measured live against where the car really was.
class SimulationScreen extends ConsumerStatefulWidget {
  const SimulationScreen({super.key});

  @override
  ConsumerState<SimulationScreen> createState() => _SimulationScreenState();
}

class _SimulationScreenState extends ConsumerState<SimulationScreen> {
  SimRoute _route = simRoutes.first;
  // OSM map: what the app does live with no route set (and, measured over
  // many simulated tunnels, the most accurate on winding roads).
  SimRoad _road = SimRoad.map;
  int _generation = 0;

  @override
  Widget build(BuildContext context) {
    final model = ref.watch(speedModelProvider);
    final data = ref.watch(replayDataProvider(_route.id));
    final roads = ref.watch(replayRoadsProvider(_route.id));
    return Scaffold(
      backgroundColor: IdrColors.background,
      body: switch ((model, data, roads)) {
        (
          AsyncData(value: final m),
          AsyncData(value: final d),
          AsyncData(value: final r),
        ) =>
          _Sim(
            key: ValueKey('${_route.id}/${_road.name}/$_generation'),
            model: m.model,
            data: d,
            roads: r,
            route: _route,
            road: _road,
            onRoute: (r) => setState(() => _route = r),
            onRoad: (r) => setState(() => _road = r),
            onRestart: () => setState(() => _generation++),
          ),
        (AsyncError(:final error), _, _) ||
        (_, AsyncError(:final error), _) ||
        (_, _, AsyncError(:final error)) => Center(
          child: Text(
            'Could not load the simulation\n$error',
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

class _Sim extends StatefulWidget {
  const _Sim({
    super.key,
    required this.model,
    required this.data,
    required this.roads,
    required this.route,
    required this.road,
    required this.onRoute,
    required this.onRoad,
    required this.onRestart,
  });

  final IdnnV5 model;
  final ReplayData data;
  final RoadData roads;
  final SimRoute route;
  final SimRoad road;
  final ValueChanged<SimRoute> onRoute;
  final ValueChanged<SimRoad> onRoad;
  final VoidCallback onRestart;

  @override
  State<_Sim> createState() => _SimState();
}

class _SimState extends State<_Sim> with SingleTickerProviderStateMixin {
  static const _speeds = [1.0, 2.0, 4.0, 8.0];

  /// GPS-healthy lead-in run instantly, so the filters and the GPS
  /// calibration have something to learn from before the first tunnel.
  static const _warmSamples = 300;

  final _map = MapController();
  late final Ticker _ticker = createTicker(_onTick);
  late final SimulationRunner _sim = SimulationRunner(
    widget.model,
    widget.data,
    road: widget.road,
    roads: widget.roads,
  );
  late final List<LatLng> _truth = [
    for (var i = 0; i < widget.data.length; i++)
      LatLng(widget.data.lat[i], widget.data.lon[i]),
  ];
  bool _mapReady = false;
  bool _follow = true;
  bool _playing = true;
  int _speedIndex = 1;
  double _carry = 0;
  Duration _last = Duration.zero;

  ReplayData get _d => widget.data;
  bool get _warming => _sim.position < _warmSamples;
  int get _k => math.max(_sim.outputs.length - 1, 0);
  DrOutput? get _o => _sim.latest;
  bool get _blackout => _o?.blackout ?? false;

  @override
  void initState() {
    super.initState();
    _ticker.start();
  }

  @override
  void dispose() {
    _ticker.dispose();
    _map.dispose();
    super.dispose();
  }

  void _onTick(Duration now) {
    final dt = (now - _last).inMicroseconds / 1e6;
    _last = now;
    if (_warming) {
      _sim.step(math.min(100, _warmSamples - _sim.position));
      setState(() {});
      return;
    }
    if (!_playing || _sim.done || dt <= 0 || dt > 0.25) {
      if (_sim.done && _playing) setState(() => _playing = false);
      return;
    }
    _carry += dt * 10 * _speeds[_speedIndex];
    final n = _carry.floor();
    if (n > 0) {
      _carry -= n;
      _sim.step(n);
    }
    // Redraw every frame: the car glides between 10 Hz samples.
    if (_mapReady && _follow) _map.move(_carPoint, _map.camera.zoom);
    setState(() {});
  }

  /// Where to draw the car: between the last two samples, so it moves
  /// smoothly at any playback speed.
  LatLng get _carPoint {
    final k = _k;
    if (k < 1) return _truth[0];
    final (la0, lo0) = _sim.positionOf(k - 1);
    final (la1, lo1) = _sim.positionOf(k);
    final f = _carry.clamp(0.0, 1.0);
    return LatLng(la0 + (la1 - la0) * f, lo0 + (lo1 - lo0) * f);
  }

  LatLng get _truePoint {
    final k = _k;
    if (k < 1) return _truth[0];
    final a = _truth[k - 1], b = _truth[math.min(k, _truth.length - 1)];
    final f = _carry.clamp(0.0, 1.0);
    return LatLng(
      a.latitude + (b.latitude - a.latitude) * f,
      a.longitude + (b.longitude - a.longitude) * f,
    );
  }

  void _toggleTunnel() {
    if (_warming || _sim.done) return;
    setState(() => _sim.inTunnel ? _sim.exitTunnel() : _sim.enterTunnel());
  }

  LatLng _geo(double e, double n) {
    final (lat, lon) = _sim.plane.toGeo(e, n);
    return LatLng(lat, lon);
  }

  // ---------------------------------------------------------------- map

  List<Widget> _layers() {
    final k = _k;
    final lines = <Polyline>[
      // the whole route, faint; the part driven, bright
      Polyline(
        points: _truth,
        color: IdrColors.textMuted.withValues(alpha: 0.5),
        strokeWidth: 3,
      ),
      Polyline(
        points: _truth.sublist(0, k + 1),
        color: IdrColors.textPrimary.withValues(alpha: 0.85),
        strokeWidth: 4,
      ),
    ];
    final markers = <Marker>[];
    for (final run in _sim.tunnels) {
      final a = run.firstOutput, b = run.lastOutput;
      if (a == null || b == null) continue;
      final inside = _truth.sublist(a, b + 1);
      // the tunnel itself: a dark band over the road
      lines.add(
        Polyline(
          points: inside,
          color: const Color(0xFF0B0C0C).withValues(alpha: 0.85),
          strokeWidth: 22,
        ),
      );
      lines.add(
        Polyline(
          points: inside,
          color: IdrColors.textPrimary.withValues(alpha: 0.55),
          strokeWidth: 2,
          pattern: StrokePattern.dashed(segments: const [6, 6]),
        ),
      );
      final idr = [
        for (var i = a; i <= b && i < _sim.outputs.length; i++)
          if (_sim.outputs[i].east != null)
            _geo(_sim.outputs[i].east!, _sim.outputs[i].north!),
      ];
      if (idr.length > 1) {
        lines.add(
          Polyline(points: idr, color: IdrColors.accent, strokeWidth: 5),
        );
      }
      markers.add(_portal(_truth[a], 'in'));
      if (!run.open) {
        markers.add(_portal(_truth[math.min(b + 1, _truth.length - 1)], 'out'));
        // where IDR thought it was vs where the car came out
        final exitTrue = _truth[math.min(b + 1, _truth.length - 1)];
        if (idr.isNotEmpty) {
          final ok = run.passesSih;
          lines.add(
            Polyline(
              points: [idr.last, exitTrue],
              color: ok ? IdrColors.positive : IdrColors.accent,
              strokeWidth: 2,
              pattern: StrokePattern.dashed(segments: const [3, 4]),
            ),
          );
          markers.add(
            Marker(
              point: idr.last,
              width: 120,
              height: 26,
              alignment: Alignment.topCenter,
              child: _ErrorTag(run: run),
            ),
          );
        }
      }
    }

    final o = _o;
    final car = _carPoint;
    final heading = _blackout ? o?.headingDeg : _d.gtHeading[_k];
    return [
      PolylineLayer(polylines: lines),
      MarkerLayer(
        markers: [
          ...markers,
          if (_blackout)
            // the real car, hidden from IDR: a hollow ghost
            Marker(
              point: _truePoint,
              width: 16,
              height: 16,
              child: const MapDot(
                color: IdrColors.textPrimary,
                hollow: true,
                size: 14,
              ),
            ),
          Marker(
            point: car,
            width: 64,
            height: 64,
            child: _CarPuck(
              color: _blackout ? IdrColors.accent : IdrColors.location,
              headingDeg: heading,
            ),
          ),
        ],
      ),
    ];
  }

  Marker _portal(LatLng p, String kind) => Marker(
    point: p,
    width: 26,
    height: 20,
    child: _PortalIcon(entry: kind == 'in'),
  );

  // ---------------------------------------------------------------- UI

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: IdrMap(
            controller: _map,
            initialCenter: _truth.first,
            initialZoom: 16.5,
            onMapReady: () => _mapReady = true,
            onUserGesture: () => setState(() => _follow = false),
            layers: _layers(),
          ),
        ),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(
              IdrSpace.md,
              IdrSpace.sm,
              IdrSpace.md,
              0,
            ),
            child: Column(
              children: [
                Row(
                  children: [
                    _Pill(
                      child: BracketButton(
                        'close',
                        onTap: () => Navigator.of(context).maybePop(),
                      ),
                    ),
                    const SizedBox(width: IdrSpace.sm),
                    // route name shrinks (…) rather than pushing others off
                    Expanded(
                      child: Align(
                        alignment: Alignment.centerRight,
                        child: _Pill(
                          child: GestureDetector(
                            onTap: _pickRoute,
                            child: Text(
                              '[${widget.route.name.toLowerCase()} ▾]',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: IdrText.body,
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: IdrSpace.sm),
                    _Pill(
                      child: BracketButton(
                        '${_speeds[_speedIndex].toStringAsFixed(0)}×',
                        onTap: () => setState(
                          () =>
                              _speedIndex = (_speedIndex + 1) % _speeds.length,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: IdrSpace.md),
                _TunnelBanner(
                  warming: _warming,
                  warmProgress: _sim.position / _warmSamples,
                  inTunnel: _sim.inTunnel,
                  seconds: _sim.current?.seconds ?? 0,
                ),
                if (!_follow)
                  Padding(
                    padding: const EdgeInsets.only(top: IdrSpace.sm),
                    child: _Pill(
                      child: BracketButton(
                        'follow car',
                        onTap: () => setState(() => _follow = true),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
        DraggableScrollableSheet(
          initialChildSize: 0.43,
          minChildSize: 0.25,
          maxChildSize: 0.86,
          snap: true,
          snapSizes: const [0.25, 0.43, 0.86],
          builder: (context, scroll) => _sheet(scroll),
        ),
      ],
    );
  }

  Widget _sheet(ScrollController scroll) {
    final o = _o;
    final k = _k;
    final run =
        _sim.current ?? (_sim.tunnels.isEmpty ? null : _sim.tunnels.last);
    final trueKmh = _d.gtSpeed[k] * 3.6;
    final idrKmh = (o?.speed ?? 0) * 3.6;
    final heading = o?.headingDeg;
    final trueHeading = _d.gtHeading[k];
    final dHeading = heading == null
        ? null
        : ((heading - trueHeading + 540) % 360) - 180;

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
      child: ListView(
        controller: scroll,
        padding: const EdgeInsets.fromLTRB(
          IdrSpace.gutter,
          IdrSpace.md,
          IdrSpace.gutter,
          IdrSpace.xxl,
        ),
        children: [
          Center(
            child: Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: IdrColors.outline,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          const SizedBox(height: IdrSpace.lg),
          if (_sim.done)
            PrimaryButton('Drive again', onPressed: widget.onRestart)
          else
            PrimaryButton(
              _warming
                  ? 'Calibrating on GPS…'
                  : _sim.inTunnel
                  ? 'Exit tunnel · GPS back'
                  : 'Enter tunnel · cut GPS',
              outlined: _sim.inTunnel || _warming,
              onPressed: _warming ? null : _toggleTunnel,
            ),
          const SizedBox(height: IdrSpace.xl),

          // live read-out
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: _Big(
                  label: _blackout ? 'IDR (no GPS)' : 'IDR',
                  value: idrKmh.round().toString(),
                  unit: 'km/h',
                  accent: _blackout,
                ),
              ),
              Expanded(
                child: _Big(
                  label: _blackout ? 'true, hidden' : 'GPS',
                  value: trueKmh.round().toString(),
                  unit: 'km/h',
                  muted: _blackout,
                ),
              ),
              Expanded(
                child: _Big(
                  label: 'heading',
                  value: heading == null ? '--' : '${heading.round()}°',
                  unit: _blackout && dHeading != null
                      ? '${dHeading >= 0 ? '+' : ''}${dHeading.round()}°'
                      : _compass(trueHeading),
                ),
              ),
            ],
          ),
          const SizedBox(height: IdrSpace.lg),
          _SpeedChart(sim: _sim),
          const SizedBox(height: IdrSpace.lg),
          Wrap(
            spacing: IdrSpace.sm,
            runSpacing: IdrSpace.sm,
            children: [
              StageChip('AI', active: _blackout),
              StageChip('GATE', active: o?.stages.gateClamped ?? false),
              StageChip('EKF', active: o?.stages.ekfBraking ?? false),
              StageChip('CRUISE', active: o?.stages.cruiseLocked ?? false),
              StageChip('ZUPT', active: o?.stages.zuptStopped ?? false),
              StageChip('ROAD', active: o?.onRoad ?? false),
            ],
          ),

          if (run != null) ...[
            const SizedBox(height: IdrSpace.xl),
            SectionLabel(run.open ? 'In the tunnel' : 'Last tunnel'),
            const SizedBox(height: IdrSpace.md),
            _RunStats(run: run),
          ],

          if (_sim.tunnels.where((r) => !r.open).isNotEmpty) ...[
            const SizedBox(height: IdrSpace.xl),
            const SectionLabel('Tunnels driven'),
            const SizedBox(height: IdrSpace.sm),
            for (final r in _sim.tunnels.reversed.where((r) => !r.open))
              _RunRow(run: r),
          ],

          const SizedBox(height: IdrSpace.xl),
          const SectionLabel('Keep IDR on the road with'),
          const SizedBox(height: IdrSpace.sm),
          Row(
            children: [
              for (final (mode, label) in const [
                (SimRoad.map, 'OSM map'),
                (SimRoad.route, 'known route'),
                (SimRoad.none, 'nothing'),
              ]) ...[
                Expanded(
                  child: _Choice(
                    label: label,
                    selected: widget.road == mode,
                    onTap: widget.road == mode
                        ? null
                        : () => widget.onRoad(mode),
                  ),
                ),
                if (mode != SimRoad.none) const SizedBox(width: IdrSpace.sm),
              ],
            ],
          ),
          const SizedBox(height: IdrSpace.sm),
          Text(switch (widget.road) {
            SimRoad.route => 'Like navigation: the route is known, IDR works out how far along it you are.',
            SimRoad.map => 'Like free driving: IDR follows OpenStreetMap roads and picks turns by heading.',
            SimRoad.none =>
              'Pure dead reckoning: speed × heading, no map at all.',
          }, style: IdrText.small),
          const SizedBox(height: IdrSpace.lg),
          Row(
            children: [
              BracketButton(
                _playing ? 'pause' : 'play',
                onTap: _sim.done
                    ? null
                    : () => setState(() => _playing = !_playing),
              ),
              const SizedBox(width: IdrSpace.lg),
              BracketButton('restart', onTap: widget.onRestart),
              const Spacer(),
              Text(
                '${_clock(k / 10)} / ${_clock(_d.length / 10)}',
                style: IdrText.micro,
              ),
            ],
          ),
          const SizedBox(height: IdrSpace.sm),
          Text(
            'Real recorded drive (IO-VNBD). The AI sees the phone’s real sensor data; '
            'the recording’s own GPS is the truth everything is measured against.',
            style: IdrText.micro,
          ),
        ],
      ),
    );
  }

  Future<void> _pickRoute() async {
    final picked = await showModalBottomSheet<SimRoute>(
      context: context,
      backgroundColor: IdrColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(IdrRadius.sheet),
        ),
      ),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
            IdrSpace.gutter,
            IdrSpace.xl,
            IdrSpace.gutter,
            IdrSpace.lg,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Choose a road', style: IdrText.title),
              const SizedBox(height: IdrSpace.lg),
              for (final r in simRoutes)
                LinkTile(
                  label: r.name,
                  caption: r.caption,
                  onTap: () => Navigator.of(context).pop(r),
                ),
            ],
          ),
        ),
      ),
    );
    if (picked != null && picked.id != widget.route.id) widget.onRoute(picked);
  }
}

String _clock(double s) {
  final m = s ~/ 60, r = (s % 60).floor();
  return '${m.toString().padLeft(2, '0')}:${r.toString().padLeft(2, '0')}';
}

String _compass(double deg) => const [
  'N',
  'NE',
  'E',
  'SE',
  'S',
  'SW',
  'W',
  'NW',
][((deg % 360) / 45).round() % 8];

// ------------------------------------------------------------------ pieces

/// Top indicator: open road (GPS) or inside a tunnel (no GPS, IDR driving).
class _TunnelBanner extends StatelessWidget {
  const _TunnelBanner({
    required this.warming,
    required this.warmProgress,
    required this.inTunnel,
    required this.seconds,
  });

  final bool warming;
  final double warmProgress;
  final bool inTunnel;
  final double seconds;

  @override
  Widget build(BuildContext context) {
    final Widget content;
    if (warming) {
      content = Row(
        key: const ValueKey('warm'),
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: IdrColors.positive,
            ),
          ),
          const SizedBox(width: IdrSpace.md),
          Text(
            'Calibrating on GPS · ${(warmProgress * 100).clamp(0, 100).round()}%',
            style: IdrText.small.copyWith(color: IdrColors.textPrimary),
          ),
        ],
      );
    } else if (inTunnel) {
      content = Row(
        key: const ValueKey('tunnel'),
        mainAxisSize: MainAxisSize.min,
        children: [
          const _PortalIcon(entry: true, size: 22),
          const SizedBox(width: IdrSpace.md),
          Text(
            'In tunnel · ${_clock(seconds)}',
            style: IdrText.body.copyWith(color: IdrColors.accent),
          ),
          const SizedBox(width: IdrSpace.md),
          const DotIndicator(filled: 0, size: 8),
          const SizedBox(width: IdrSpace.sm),
          Text(
            'no GPS',
            style: IdrText.micro.copyWith(color: IdrColors.accent),
          ),
        ],
      );
    } else {
      content = Row(
        key: const ValueKey('open'),
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(
            Icons.wb_sunny_outlined,
            size: 18,
            color: IdrColors.textPrimary,
          ),
          const SizedBox(width: IdrSpace.md),
          Text('Open road', style: IdrText.body),
          const SizedBox(width: IdrSpace.md),
          const DotIndicator(filled: 5, size: 8),
          const SizedBox(width: IdrSpace.sm),
          Text('GPS', style: IdrText.micro.copyWith(color: IdrColors.positive)),
        ],
      );
    }
    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
      decoration: BoxDecoration(
        color: IdrColors.background.withValues(alpha: 0.92),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: inTunnel ? IdrColors.accent : IdrColors.outline,
          width: inTunnel ? 1.5 : 1,
        ),
      ),
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 250),
        child: content,
      ),
    );
  }
}

/// Tunnel mouth glyph: an arch, orange for the way in.
class _PortalIcon extends StatelessWidget {
  const _PortalIcon({required this.entry, this.size = 20});

  final bool entry;
  final double size;

  @override
  Widget build(BuildContext context) => CustomPaint(
    size: Size(size * 1.3, size),
    painter: _PortalPainter(entry ? IdrColors.accent : IdrColors.positive),
  );
}

class _PortalPainter extends CustomPainter {
  _PortalPainter(this.color);

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width, h = size.height;
    canvas.drawRRect(
      RRect.fromLTRBAndCorners(
        0,
        0,
        w,
        h,
        topLeft: Radius.circular(w / 2),
        topRight: Radius.circular(w / 2),
      ),
      Paint()..color = IdrColors.background,
    );
    final arch = Path()
      ..moveTo(w * 0.18, h)
      ..lineTo(w * 0.18, h * 0.55)
      ..arcToPoint(
        Offset(w * 0.82, h * 0.55),
        radius: Radius.circular(w * 0.32),
      )
      ..lineTo(w * 0.82, h);
    canvas.drawPath(
      arch,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.2,
    );
  }

  @override
  bool shouldRepaint(_PortalPainter old) => old.color != color;
}

/// The car on the map: a dot with a heading beam (blue on GPS, orange on IDR).
class _CarPuck extends StatelessWidget {
  const _CarPuck({required this.color, this.headingDeg});

  final Color color;
  final double? headingDeg;

  @override
  Widget build(BuildContext context) =>
      CustomPaint(painter: _PuckPainter(color, headingDeg));
}

class _PuckPainter extends CustomPainter {
  _PuckPainter(this.color, this.heading);

  final Color color;
  final double? heading;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final h = heading;
    if (h != null) {
      final a = (h - 90) * math.pi / 180;
      canvas.drawPath(
        Path()
          ..moveTo(c.dx, c.dy)
          ..arcTo(Rect.fromCircle(center: c, radius: 30), a - 0.45, 0.9, false)
          ..close(),
        Paint()
          ..shader = RadialGradient(
            colors: [color.withValues(alpha: 0.55), color.withValues(alpha: 0)],
          ).createShader(Rect.fromCircle(center: c, radius: 30)),
      );
    }
    canvas.drawCircle(c, 13, Paint()..color = color.withValues(alpha: 0.18));
    canvas.drawCircle(c, 8.5, Paint()..color = Colors.white);
    canvas.drawCircle(c, 6.5, Paint()..color = color);
  }

  @override
  bool shouldRepaint(_PuckPainter old) =>
      old.color != color || old.heading != heading;
}

/// "off by 7 m" tag at IDR's position when GPS came back.
class _ErrorTag extends StatelessWidget {
  const _ErrorTag({required this.run});

  final TunnelRun run;

  @override
  Widget build(BuildContext context) {
    final ok = run.passesSih;
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(
          color: IdrColors.background.withValues(alpha: 0.92),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: ok ? IdrColors.positive : IdrColors.accent),
        ),
        child: Text(
          '#${run.number} · ${run.exitErrorM!.round()} m',
          style: IdrText.micro.copyWith(
            color: ok ? IdrColors.positive : IdrColors.accent,
          ),
        ),
      ),
    );
  }
}

class _Big extends StatelessWidget {
  const _Big({
    required this.label,
    required this.value,
    required this.unit,
    this.accent = false,
    this.muted = false,
  });

  final String label;
  final String value;
  final String unit;
  final bool accent;
  final bool muted;

  @override
  Widget build(BuildContext context) {
    final color = accent
        ? IdrColors.accent
        : (muted ? IdrColors.textSecondary : IdrColors.textPrimary);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: IdrText.micro),
        const SizedBox(height: 4),
        Row(
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            Text(
              value,
              style: IdrText.stat.copyWith(color: color, fontSize: 30),
            ),
            const SizedBox(width: 4),
            Flexible(
              child: Text(
                unit,
                style: IdrText.micro,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

/// The current (or last) tunnel, measured against the truth.
class _RunStats extends StatelessWidget {
  const _RunStats({required this.run});

  final TunnelRun run;

  @override
  Widget build(BuildContext context) {
    final off = run.exitErrorM ?? run.liveErrorM;
    final ok = run.passesSih;
    Widget cell(String label, String value, {Color? color}) => Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: IdrText.micro),
          const SizedBox(height: 2),
          Text(value, style: IdrText.body.copyWith(color: color)),
        ],
      ),
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            cell('time without GPS', _clock(run.seconds)),
            cell('driven', '${run.trueDistanceM.round()} m'),
            cell('IDR measured', '${run.idrDistanceM.round()} m'),
          ],
        ),
        const SizedBox(height: IdrSpace.md),
        Row(
          children: [
            cell(
              run.open ? 'IDR is off by' : 'off when GPS back',
              '${off.round()} m',
              color: ok ? IdrColors.positive : IdrColors.accent,
            ),
            cell(
              'error vs distance',
              '${run.errorPct.toStringAsFixed(1)} %',
              color: ok ? IdrColors.positive : IdrColors.accent,
            ),
            cell(
              'distance error',
              '${run.distanceDriftPct.toStringAsFixed(1)} %',
            ),
          ],
        ),
        const SizedBox(height: IdrSpace.md),
        Row(
          children: [
            Icon(
              ok ? Icons.check_circle_outline : Icons.error_outline,
              size: 16,
              color: ok ? IdrColors.positive : IdrColors.accent,
            ),
            const SizedBox(width: IdrSpace.sm),
            Expanded(
              child: Text(
                ok
                    ? 'Within the SIH target (under 10 % of the distance driven)'
                    : 'Over the SIH 10 % target on this stretch',
                style: IdrText.small.copyWith(
                  color: ok ? IdrColors.positive : IdrColors.textSecondary,
                ),
              ),
            ),
          ],
        ),
        if (run.maxErrorM > 0) ...[
          const SizedBox(height: IdrSpace.xs),
          Text(
            'Largest gap along the way: ${run.maxErrorM.round()} m',
            style: IdrText.micro,
          ),
        ],
      ],
    );
  }
}

class _RunRow extends StatelessWidget {
  const _RunRow({required this.run});

  final TunnelRun run;

  @override
  Widget build(BuildContext context) {
    final ok = run.passesSih;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 34,
            child: Text(
              '#${run.number}',
              style: IdrText.small.copyWith(color: IdrColors.textPrimary),
            ),
          ),
          Expanded(
            child: Text(
              '${_clock(run.seconds)} · ${run.trueDistanceM.round()} m',
              style: IdrText.small,
            ),
          ),
          Text(
            '${run.exitErrorM!.round()} m · ${run.errorPct.toStringAsFixed(1)} %',
            style: IdrText.small.copyWith(
              color: ok ? IdrColors.positive : IdrColors.accent,
            ),
          ),
          const SizedBox(width: IdrSpace.sm),
          Icon(
            ok ? Icons.check : Icons.close,
            size: 16,
            color: ok ? IdrColors.positive : IdrColors.accent,
          ),
        ],
      ),
    );
  }
}

class _Choice extends StatelessWidget {
  const _Choice({required this.label, required this.selected, this.onTap});

  final String label;
  final bool selected;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      height: 40,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: selected ? IdrColors.textPrimary : Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: selected ? IdrColors.textPrimary : IdrColors.outline,
        ),
      ),
      child: Text(
        label,
        style: IdrText.small.copyWith(
          color: selected ? IdrColors.onTileActive : IdrColors.textSecondary,
        ),
      ),
    ),
  );
}

/// Last 60 s of speed: GPS/truth in white, IDR in orange, tunnels shaded.
class _SpeedChart extends StatelessWidget {
  const _SpeedChart({required this.sim});

  final SimulationRunner sim;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text('speed · last minute', style: IdrText.micro),
            const Spacer(),
            Container(width: 10, height: 2, color: IdrColors.textPrimary),
            const SizedBox(width: 4),
            Text('true', style: IdrText.micro),
            const SizedBox(width: IdrSpace.md),
            Container(width: 10, height: 2, color: IdrColors.accent),
            const SizedBox(width: 4),
            Text('IDR', style: IdrText.micro),
          ],
        ),
        const SizedBox(height: IdrSpace.sm),
        SizedBox(
          height: 72,
          width: double.infinity,
          child: CustomPaint(painter: _ChartPainter(sim, sim.outputs.length)),
        ),
      ],
    );
  }
}

class _ChartPainter extends CustomPainter {
  _ChartPainter(this.sim, this.count);

  final SimulationRunner sim;
  final int count;
  static const _window = 600;

  @override
  void paint(Canvas canvas, Size size) {
    final outs = sim.outputs;
    if (outs.length < 2) return;
    final from = math.max(0, outs.length - _window);
    var top = 5.0;
    for (var i = from; i < outs.length; i++) {
      top = math.max(top, math.max(outs[i].speed, sim.data.gtSpeed[i]));
    }
    top *= 1.15;
    double x(int i) => (i - from) / (_window - 1) * size.width;
    double y(double v) => size.height - (v / top).clamp(0.0, 1.0) * size.height;

    // tunnels
    final shade = Paint()..color = IdrColors.accent.withValues(alpha: 0.12);
    var start = -1;
    for (var i = from; i <= outs.length; i++) {
      final b = i < outs.length && outs[i].blackout;
      if (b && start < 0) start = i;
      if (!b && start >= 0) {
        canvas.drawRect(
          Rect.fromLTRB(x(start), 0, x(i - 1), size.height),
          shade,
        );
        start = -1;
      }
    }
    // baseline
    canvas.drawLine(
      Offset(0, size.height),
      Offset(size.width, size.height),
      Paint()
        ..color = IdrColors.outline
        ..strokeWidth = 1,
    );
    Path line(double Function(int) v) {
      final p = Path()..moveTo(x(from), y(v(from)));
      for (var i = from + 1; i < outs.length; i++) {
        p.lineTo(x(i), y(v(i)));
      }
      return p;
    }

    canvas.drawPath(
      line((i) => sim.data.gtSpeed[i]),
      Paint()
        ..color = IdrColors.textPrimary.withValues(alpha: 0.8)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );
    canvas.drawPath(
      line((i) => outs[i].speed),
      Paint()
        ..color = IdrColors.accent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(_ChartPainter old) => old.count != count;
}

class _Pill extends StatelessWidget {
  const _Pill({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
    decoration: BoxDecoration(
      color: IdrColors.background.withValues(alpha: 0.88),
      borderRadius: BorderRadius.circular(20),
    ),
    child: child,
  );
}
