import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../state/nav_state.dart';

/// Map tab: live position on the dark map, with the tunnel-test controls
/// in a bottom sheet (layout from the "Wien Energie" mockup).
class MapScreen extends ConsumerStatefulWidget {
  const MapScreen({super.key});

  @override
  ConsumerState<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends ConsumerState<MapScreen> {
  final _map = MapController();
  bool _ready = false;
  bool _follow = true;

  @override
  void dispose() {
    _map.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final v = ref.watch(navViewProvider);
    final toggle = ref.read(tunnelTestProvider.notifier).toggle;
    final fix = v.fix;
    final gps = fix == null ? null : LatLng(fix.latitude, fix.longitude);
    final estimate = v.estimateLat == null
        ? null
        : LatLng(v.estimateLat!, v.estimateLon!);
    // In sensor mode the best position is IDR's estimate.
    final here = (v.inTunnel ? estimate : null) ?? gps;

    return Stack(
      children: [
        Positioned.fill(
          child: IdrMap(
            controller: _map,
            initialCenter: here ?? const LatLng(52.4068, -1.5197),
            initialZoom: here == null ? 12 : 16,
            onMapReady: () => _ready = true,
            onUserGesture: () {
              if (_follow) setState(() => _follow = false);
            },
            layers: [
              if (here != null)
                LocationPuckLayer(
                  position: here,
                  headingDeg: v.headingDeg,
                  // GPS: its reported accuracy. Sensors: uncertainty grows
                  // with distance (≈5 % in the Drive M benchmark).
                  accuracyM: v.inTunnel
                      ? (5 + 0.05 * v.tunnelDistanceM) * (v.mounted ? 1 : 3)
                      : v.accuracyM,
                  color: v.inTunnel ? IdrColors.accent : IdrColors.location,
                  stale: v.mode == NavMode.waiting,
                  onMoved: (p) {
                    if (_ready && _follow) _map.move(p, _map.camera.zoom);
                  },
                ),
            ],
          ),
        ),
        if (here != null)
          SafeArea(
            child: Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(0, 28, IdrSpace.md, 0),
                child: RecenterButton(
                  following: _follow,
                  onTap: () {
                    setState(() => _follow = true);
                    if (_ready) _map.move(here, math.max(_map.camera.zoom, 16));
                  },
                ),
              ),
            ),
          ),
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
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
              IdrSpace.xl,
            ),
            child: v.inTunnel
                ? _SensorSheet(view: v, onEnd: toggle)
                : _GpsSheet(view: v, onTest: toggle),
          ),
        ),
      ],
    );
  }
}

class _GpsSheet extends StatelessWidget {
  const _GpsSheet({required this.view, required this.onTest});

  final NavView view;
  final VoidCallback onTest;

  @override
  Widget build(BuildContext context) {
    final hasFix = view.accuracyM != null;
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _SheetTitle(
          title: hasFix ? 'GPS locked' : view.statusLabel,
          trailing: hasFix ? '[±${view.accuracyM!.round()}m]' : null,
        ),
        const SizedBox(height: IdrSpace.sm),
        Text(
          'Try a tunnel test to see how IDR tracks you without GPS.',
          style: IdrText.label.copyWith(fontSize: 15),
        ),
        const SizedBox(height: IdrSpace.xl),
        PrimaryButton('Start tunnel test', onPressed: onTest),
      ],
    );
  }
}

class _SensorSheet extends StatelessWidget {
  const _SensorSheet({required this.view, required this.onEnd});

  final NavView view;
  final VoidCallback onEnd;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _SheetTitle(
          title: 'No GPS · ${view.tunnelClock}',
          trailing: view.simulated ? '[test]' : '[live]',
          accent: true,
        ),
        const SizedBox(height: IdrSpace.sm),
        Text(
          !view.mounted
              ? 'Phone isn’t mounted — this estimate is unreliable.'
              : view.onRoad
              ? 'Following the road map with motion sensors.'
              : 'Tracking with motion sensors.',
          style: IdrText.label.copyWith(fontSize: 15),
        ),
        const SizedBox(height: IdrSpace.xl),
        Row(
          children: [
            Expanded(
              child: RoundInfo(
                icon: Icons.speed,
                line1: '${view.speedLabel} km/h',
                line2: 'estimated',
              ),
            ),
            Expanded(
              child: RoundInfo(
                icon: Icons.straighten,
                line1: '${view.tunnelDistanceM.round()} m',
                line2: 'since GPS lost',
              ),
            ),
          ],
        ),
        const SizedBox(height: IdrSpace.xl),
        if (view.simulated)
          PrimaryButton('End test', onPressed: onEnd, outlined: true),
      ],
    );
  }
}

class _SheetTitle extends StatelessWidget {
  const _SheetTitle({required this.title, this.trailing, this.accent = false});

  final String title;
  final String? trailing;
  final bool accent;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            title,
            style: IdrText.title.copyWith(
              fontSize: 26,
              color: accent ? IdrColors.accent : null,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ),
        if (trailing != null) Text(trailing!, style: IdrText.body),
      ],
    );
  }
}
