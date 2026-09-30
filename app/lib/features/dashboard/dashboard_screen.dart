import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/routes.dart';
import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/sources/gnss_source.dart';
import '../../data/model_loader.dart';
import '../../state/drive_recorder.dart';
import '../../state/live_engine.dart';
import '../../state/nav_state.dart';
import '../shell/page_scaffold.dart';

/// Home tab: speed, heading, GPS signal, and two actions.
class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final v = ref.watch(navViewProvider);
    final recording = ref.watch(
      driveRecorderProvider.select((r) => r.recording),
    );

    return SafeArea(
      bottom: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          IdrSpace.gutter,
          IdrSpace.xl,
          0,
          IdrSpace.lg,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.only(right: IdrSpace.gutter),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  IdrHeader(title: v.vehicleName, subtitle: v.statusLabel),
                  const SizedBox(height: 28),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.baseline,
                    textBaseline: TextBaseline.alphabetic,
                    children: [
                      Text(
                        v.speedLabel,
                        style: IdrText.hero.copyWith(
                          color: v.inTunnel ? IdrColors.accent : null,
                        ),
                      ),
                      const SizedBox(width: IdrSpace.sm),
                      Text('km/h', style: IdrText.label),
                      const Spacer(),
                      if (v.inTunnel)
                        Text(
                          'estimated',
                          style: IdrText.small.copyWith(
                            color: IdrColors.accent,
                          ),
                        )
                      else if (recording)
                        GestureDetector(
                          onTap: () => context.push(Routes.history),
                          child: Text(
                            '● rec',
                            style: IdrText.small.copyWith(
                              color: IdrColors.accent,
                            ),
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: IdrSpace.lg),
                  KeyValueRow(label: 'Heading', value: v.headingLabel),
                  KeyValueRow(
                    label: 'GPS',
                    value: switch (v.mode) {
                      NavMode.sensors => 'off',
                      _ when v.accuracyM == null => '—',
                      _ => '±${v.accuracyM!.round()} m',
                    },
                    valueColor: v.inTunnel ? IdrColors.accent : null,
                    trailing: DotIndicator(filled: v.gpsDots, size: 14),
                  ),
                  if (v.locationOff) ...[
                    const SizedBox(height: IdrSpace.md),
                    _LocationBanner(access: v.access),
                  ] else if (_problem(ref) case final msg?) ...[
                    const SizedBox(height: IdrSpace.md),
                    _ProblemBanner(message: msg),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 28),
            Expanded(child: _ControlDeck(view: v)),
          ],
        ),
      ),
    );
  }
}

/// Anything that stops IDR from working, in plain words (null if all good).
String? _problem(WidgetRef ref) {
  if (ref.watch(speedModelProvider).hasError) {
    return 'The AI model failed to load. Reinstall the app.';
  }
  if (ref.watch(liveEngineProvider.select((e) => e.error)) != null) {
    return 'Motion sensors are unavailable, so IDR can’t navigate without GPS.';
  }
  final engine = ref.watch(liveEngineProvider);
  if (engine.running && engine.frames > 50 && !engine.mounted) {
    return 'Phone isn’t steady. Put it in a mount — tracking without GPS needs it.';
  }
  return null;
}

class _ProblemBanner extends StatelessWidget {
  const _ProblemBanner({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return IdrTile(
      padding: const EdgeInsets.fromLTRB(
        IdrSpace.lg,
        IdrSpace.md,
        IdrSpace.lg,
        IdrSpace.md,
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: IdrColors.accent, size: 22),
          const SizedBox(width: IdrSpace.md),
          Expanded(
            child: Text(
              message,
              style: IdrText.small.copyWith(color: IdrColors.textPrimary),
            ),
          ),
        ],
      ),
    );
  }
}

class _LocationBanner extends ConsumerWidget {
  const _LocationBanner({required this.access});

  final LocationAccess access;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ctl = ref.read(locationAccessProvider.notifier);
    final canPrompt = access == LocationAccess.denied;
    return IdrTile(
      padding: const EdgeInsets.fromLTRB(
        IdrSpace.lg,
        IdrSpace.md,
        IdrSpace.lg,
        IdrSpace.md,
      ),
      child: Row(
        children: [
          const Icon(
            Icons.location_off_outlined,
            color: IdrColors.accent,
            size: 22,
          ),
          const SizedBox(width: IdrSpace.md),
          Expanded(
            child: Text(
              'Location is needed.',
              style: IdrText.small.copyWith(color: IdrColors.textPrimary),
            ),
          ),
          BracketButton(
            canPrompt ? 'allow' : 'open',
            onTap: canPrompt ? ctl.request : ctl.openSettings,
            style: IdrText.body.copyWith(color: IdrColors.accent, fontSize: 15),
          ),
        ],
      ),
    );
  }
}

class _ControlDeck extends ConsumerWidget {
  const _ControlDeck({required this.view});

  final NavView view;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return LayoutBuilder(
      builder: (context, box) {
        const gap = 8.0;
        final tile = ((box.maxHeight - gap) / 2).clamp(
          64.0,
          box.maxWidth * 0.3,
        );
        final deckHeight = tile * 2 + gap;
        return Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            SizedBox(
              width: tile,
              height: deckHeight,
              child: VerticalGauge(value: view.confidence, caption: 'accuracy'),
            ),
            const SizedBox(width: gap),
            SizedBox(
              width: tile,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  IconTile(
                    icon: Icons.satellite_alt_outlined,
                    label: view.inTunnel ? 'End test' : 'Tunnel test',
                    semanticLabel: 'Tunnel test',
                    active: view.inTunnel,
                    onTap: ref.read(tunnelTestProvider.notifier).toggle,
                  ),
                  const SizedBox(height: gap),
                  IconTile(
                    icon: Icons.alt_route,
                    label: 'Simulate',
                    semanticLabel: 'Start simulation',
                    onTap: () => context.push(Routes.simulation),
                  ),
                ],
              ),
            ),
            const SizedBox(width: IdrSpace.xl),
            Expanded(
              child: SizedBox(
                height: deckHeight,
                child: CarLineArt(headlightsOn: view.inTunnel),
              ),
            ),
          ],
        );
      },
    );
  }
}
