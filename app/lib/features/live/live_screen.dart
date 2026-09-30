import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../state/live_engine.dart';
import '../../state/nav_state.dart';
import '../shell/page_scaffold.dart';

/// Live tab: speed over the last minute (GPS dots vs the AI's line), what
/// IDR is doing right now, and engine internals behind "[details]".
class LiveScreen extends ConsumerStatefulWidget {
  const LiveScreen({super.key});

  @override
  ConsumerState<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends ConsumerState<LiveScreen> {
  bool _details = false;

  @override
  Widget build(BuildContext context) {
    final v = ref.watch(navViewProvider);
    final engine = ref.watch(liveEngineProvider);
    final history = ref.watch(speedHistoryProvider);
    final chart = _chartColumns(history);

    return IdrScrollPage(
      children: [
        IdrHeader(
          title: '${v.speedLabel} km/h',
          titleStyle: IdrText.display.copyWith(
            color: v.inTunnel ? IdrColors.accent : null,
          ),
          action: BracketButton(
            _details ? 'less' : 'details',
            onTap: () => setState(() => _details = !_details),
          ),
        ),
        const SizedBox(height: IdrSpace.sm),
        Text(switch (v.mode) {
          NavMode.sensors => 'estimated from sensors',
          NavMode.waiting => 'waiting for GPS',
          _ => 'from GPS',
        }, style: IdrText.label),
        const SizedBox(height: IdrSpace.xl),
        DotMatrixChart(
          values: chart.shown,
          overlay: chart.hasAi ? chart.ai : null,
          labels: const ['60', '50', '40', '30', '20', '10', '0'],
        ),
        const SizedBox(height: IdrSpace.sm),
        Row(
          children: [
            _Legend(
              color: IdrColors.accent,
              label: v.inTunnel ? 'speed shown' : 'GPS',
            ),
            const SizedBox(width: IdrSpace.lg),
            if (chart.hasAi)
              const _Legend(color: Color(0xFFE8C9BC), label: 'AI estimate'),
            const Spacer(),
            Text(
              's ago',
              style: IdrText.small.copyWith(color: IdrColors.textMuted),
            ),
          ],
        ),
        const SizedBox(height: IdrSpace.xl),
        IdrTile(
          padding: const EdgeInsets.all(IdrSpace.xl),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 10,
                height: 10,
                margin: const EdgeInsets.only(top: 6, right: IdrSpace.md),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: switch (v.mode) {
                    NavMode.sensors => IdrColors.accent,
                    NavMode.waiting => IdrColors.textMuted,
                    _ => IdrColors.positive,
                  },
                ),
              ),
              Expanded(
                child: Text(
                  _explain(v, engine),
                  style: IdrText.body.copyWith(fontSize: 15, height: 1.5),
                ),
              ),
            ],
          ),
        ),
        if (_details) ..._engineRows(engine),
        if (_details) ..._gpsRows(v),
      ],
    );
  }

  /// Seven columns (60, 50 … 0 s ago) scaled to at least 60 km/h.
  ({List<double> shown, List<double> ai, bool hasAi}) _chartColumns(
    List<SpeedSample> history,
  ) {
    final shown = <double>[], ai = <double>[];
    var hasAi = false;
    for (var ago = 60; ago >= 0; ago -= 10) {
      final i = history.length - 1 - ago;
      final s = i >= 0 ? history[i] : null;
      shown.add(s?.shown ?? 0);
      ai.add(s?.ai ?? 0);
      hasAi |= s?.ai != null;
    }
    final top = math.max(60.0, [...shown, ...ai].reduce(math.max));
    return (
      shown: [for (final s in shown) s / top],
      ai: [for (final s in ai) s / top],
      hasAi: hasAi,
    );
  }

  String _explain(NavView v, LiveEngineState e) => switch (v.mode) {
    NavMode.waiting when v.locationOff =>
      'Location is off, so IDR can’t see GPS. Turn it on to start tracking.',
    NavMode.waiting =>
      'Looking for satellites. Outdoors with a clear sky works best.',
    NavMode.parked => 'GPS is locked. While you’re stopped, IDR measures your car’s idle vibration.',
    NavMode.gps when !e.aligned => 'GPS is healthy. IDR is learning how your phone is mounted — a few turns will do it.',
    NavMode.gps => 'GPS is healthy. IDR is calibrating against it so it’s ready for the next tunnel.',
    NavMode.sensors when v.simulated => 'Tunnel test: GPS is hidden from IDR. Speed now comes from the AI reading your phone’s motion sensors.',
    NavMode.sensors => 'GPS is gone. Speed and position now come from the AI reading your phone’s motion sensors.',
  };

  List<Widget> _engineRows(LiveEngineState e) {
    final o = e.output;
    String kmh(double? ms) =>
        ms == null ? '—' : '${(ms * 3.6).toStringAsFixed(1)} km/h';
    final st = o?.stages;
    return [
      const SectionLabel('ai engine'),
      if (e.error != null)
        Text(
          'Motion sensors unavailable: ${e.error}',
          style: IdrText.small.copyWith(color: IdrColors.accent),
        )
      else if (!e.running)
        const Text('Starting…', style: IdrText.small)
      else ...[
        KeyValueRow(label: 'Raw AI', value: kmh(o?.rawSpeed)),
        KeyValueRow(label: 'Filtered', value: kmh(o?.filteredSpeed)),
        if (o?.bias != null)
          KeyValueRow(
            label: 'Entry bias',
            value: '${o!.bias!.toStringAsFixed(2)} m/s',
          ),
        const SizedBox(height: IdrSpace.sm),
        Wrap(
          spacing: IdrSpace.sm,
          runSpacing: IdrSpace.sm,
          children: [
            StageChip('GATE', active: st?.gateClamped ?? false),
            StageChip('EKF', active: st?.ekfBraking ?? false),
            StageChip('CRUISE', active: st?.cruiseLocked ?? false),
            StageChip('ZUPT', active: st?.zuptStopped ?? false),
            StageChip('CALIB', active: st?.calibrated ?? false),
          ],
        ),
        const SizedBox(height: IdrSpace.sm),
        KeyValueRow(
          label: 'Sensors',
          value: '${e.frameRateHz.toStringAsFixed(1)} Hz',
        ),
        KeyValueRow(
          label: 'Gravity',
          value: '${e.gravityMagnitude.toStringAsFixed(2)} m/s²',
        ),
        KeyValueRow(
          label: 'Held by',
          value: e.mounted
              ? 'mount (wobble ${e.wobbleDeg.toStringAsFixed(1)}°)'
              : 'hand? (wobble ${e.wobbleDeg.toStringAsFixed(1)}°)',
        ),
        KeyValueRow(
          label: 'Mount',
          value: e.aligned
              ? 'learned'
              : 'learning ${(e.alignmentProgress * 100).round()}%',
        ),
        KeyValueRow(
          label: 'Idle vib.',
          value: e.idleBaseline.toStringAsFixed(4),
        ),
        KeyValueRow(label: 'GPS fed', value: e.gnssUsed ? 'yes' : 'no'),
      ],
    ];
  }

  List<Widget> _gpsRows(NavView v) {
    final f = v.fix;
    final age = f == null ? null : DateTime.now().difference(f.time).inSeconds;
    return [
      const SectionLabel('gps readings'),
      KeyValueRow(
        label: 'Latitude',
        value: f == null ? '—' : f.latitude.toStringAsFixed(5),
      ),
      KeyValueRow(
        label: 'Longitude',
        value: f == null ? '—' : f.longitude.toStringAsFixed(5),
      ),
      KeyValueRow(
        label: 'Accuracy',
        value: f == null ? '—' : '±${f.accuracyM.toStringAsFixed(1)} m',
      ),
      KeyValueRow(label: 'Fix age', value: age == null ? '—' : '$age s'),
    ];
  }
}

class _Legend extends StatelessWidget {
  const _Legend({required this.color, required this.label});

  final Color color;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(label, style: IdrText.small),
      ],
    );
  }
}
