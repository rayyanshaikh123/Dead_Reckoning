import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/routes.dart';
import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/sources/gnss_source.dart';
import '../../state/settings.dart';

/// Animated splash: a trail drives through a tunnel, then the wordmark fades in.
/// Tap anywhere to skip.
class SplashScreen extends ConsumerStatefulWidget {
  const SplashScreen({super.key});

  @override
  ConsumerState<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends ConsumerState<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2400),
  );
  late final Animation<double> _route = CurvedAnimation(
    parent: _c,
    curve: const Interval(0.0, 0.7, curve: Curves.easeInOut),
  );
  late final Animation<double> _brand = CurvedAnimation(
    parent: _c,
    curve: const Interval(0.45, 0.85, curve: Curves.easeOut),
  );
  bool _left = false;

  @override
  void initState() {
    super.initState();
    // Read permission state early so Home knows whether GPS can start.
    ref.read(locationAccessProvider.notifier).refresh();
    _c.forward().whenComplete(_continue);
  }

  void _continue() {
    if (_left || !mounted) return;
    _left = true;
    final onboarded = ref.read(appSettingsProvider).onboarded;
    context.go(onboarded ? Routes.dashboard : Routes.onboarding);
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: _continue,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: IdrSpace.gutter),
            child: AnimatedBuilder(
              animation: _c,
              builder: (context, _) => Column(
                children: [
                  const Spacer(flex: 3),
                  SizedBox(
                    height: 150,
                    child: TunnelRouteIllustration(
                      progress: _route.value,
                      sensorNav: true,
                    ),
                  ),
                  const SizedBox(height: IdrSpace.xxl),
                  Opacity(
                    opacity: _brand.value,
                    child: Transform.translate(
                      offset: Offset(0, 12 * (1 - _brand.value)),
                      child: Column(
                        children: [
                          const IdrWordmark(),
                          const SizedBox(height: IdrSpace.md),
                          Text(
                            'Keeps navigating\nwhen GPS drops.',
                            textAlign: TextAlign.center,
                            style: IdrText.label.copyWith(fontSize: 16),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const Spacer(flex: 4),
                  Opacity(
                    opacity: _brand.value,
                    child: Text(
                      'on-device · offline',
                      style: IdrText.small.copyWith(color: IdrColors.textMuted),
                    ),
                  ),
                  const SizedBox(height: IdrSpace.lg),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
