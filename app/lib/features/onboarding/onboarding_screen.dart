import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/routes.dart';
import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/sources/gnss_source.dart';
import '../../state/settings.dart';

/// First-run intro: three explainer pages, then a one-screen setup
/// (car name + location permission).
class OnboardingScreen extends ConsumerStatefulWidget {
  const OnboardingScreen({super.key});

  @override
  ConsumerState<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends ConsumerState<OnboardingScreen>
    with SingleTickerProviderStateMixin {
  static const _pageCount = 4;
  static const _setupPage = _pageCount - 1;

  final _pages = PageController();
  late final AnimationController _loop = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 3200),
  )..repeat();
  late final TextEditingController _name = TextEditingController(
    text: _initialName(),
  );
  int _page = 0;
  bool _finishing = false;

  String _initialName() {
    final n = ref.read(appSettingsProvider).vehicleName;
    return n == AppSettingsController.defaultVehicleName ? '' : n;
  }

  @override
  void dispose() {
    _pages.dispose();
    _loop.dispose();
    _name.dispose();
    super.dispose();
  }

  void _goTo(int page) {
    _pages.animateToPage(
      page,
      duration: const Duration(milliseconds: 380),
      curve: Curves.easeOutCubic,
    );
  }

  Future<void> _finish() async {
    if (_finishing) return;
    setState(() => _finishing = true);
    await ref
        .read(appSettingsProvider.notifier)
        .completeOnboarding(vehicleName: _name.text);
    if (mounted) context.go(Routes.dashboard);
  }

  @override
  Widget build(BuildContext context) {
    final onSetup = _page == _setupPage;
    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(
                IdrSpace.gutter,
                IdrSpace.lg,
                IdrSpace.gutter,
                0,
              ),
              child: Row(
                children: [
                  for (var i = 0; i < _pageCount; i++)
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      width: i == _page ? 28 : 10,
                      height: 6,
                      margin: const EdgeInsets.only(right: 6),
                      decoration: BoxDecoration(
                        color: i <= _page
                            ? IdrColors.accent
                            : IdrColors.surfaceHigh,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  const Spacer(),
                  AnimatedOpacity(
                    duration: const Duration(milliseconds: 200),
                    opacity: onSetup ? 0 : 1,
                    child: BracketButton(
                      'skip',
                      onTap: onSetup ? null : () => _goTo(_setupPage),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: PageView(
                controller: _pages,
                onPageChanged: (p) {
                  FocusScope.of(context).unfocus();
                  setState(() => _page = p);
                },
                children: [
                  _IntroPage(
                    art: AnimatedBuilder(
                      animation: _loop,
                      builder: (_, _) => TunnelRouteIllustration(
                        progress: _loop.value,
                        sensorNav: false,
                      ),
                    ),
                    title: 'GPS drops in tunnels.',
                    body:
                        'Tunnels, underpasses and tall buildings block satellite signals. '
                        'Most map apps freeze or guess where you are.',
                  ),
                  _IntroPage(
                    art: AnimatedBuilder(
                      animation: _loop,
                      builder: (_, _) => TunnelRouteIllustration(
                        progress: _loop.value,
                        sensorNav: true,
                      ),
                    ),
                    title: 'IDR keeps going.',
                    body:
                        "An AI model reads your phone's motion sensors to estimate speed "
                        'and position until GPS comes back. It runs on your phone, with no internet.',
                  ),
                  _IntroPage(
                    art: AnimatedBuilder(
                      animation: _loop,
                      builder: (_, _) =>
                          PhoneMountIllustration(pulse: _loop.value),
                    ),
                    title: 'Mount it. Forget it.',
                    body:
                        'Put your phone in a fixed car mount. IDR learns your car and '
                        'mount by itself during the first few minutes of driving.',
                  ),
                  _SetupPage(nameController: _name),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(
                IdrSpace.gutter,
                IdrSpace.md,
                IdrSpace.gutter,
                IdrSpace.lg,
              ),
              child: PrimaryButton(
                onSetup ? 'Get started' : 'Next',
                onPressed: onSetup ? _finish : () => _goTo(_page + 1),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _IntroPage extends StatelessWidget {
  const _IntroPage({
    required this.art,
    required this.title,
    required this.body,
  });

  final Widget art;
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: IdrSpace.gutter),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 5,
            child: Center(child: SizedBox(height: 220, child: art)),
          ),
          Text(title, style: IdrText.title),
          const SizedBox(height: IdrSpace.lg),
          Text(body, style: IdrText.label.copyWith(height: 1.5)),
          const Spacer(flex: 2),
        ],
      ),
    );
  }
}

class _SetupPage extends ConsumerWidget {
  const _SetupPage({required this.nameController});

  final TextEditingController nameController;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final access = ref.watch(locationAccessProvider);
    final ctl = ref.read(locationAccessProvider.notifier);

    final (String status, Widget action) = switch (access) {
      LocationAccess.granted => (
        'Allowed',
        Text('[on]', style: IdrText.body.copyWith(color: IdrColors.positive)),
      ),
      LocationAccess.deniedForever => (
        'Blocked — enable it in Settings',
        BracketButton(
          'open',
          onTap: ctl.openSettings,
          style: IdrText.body.copyWith(color: IdrColors.accent),
        ),
      ),
      LocationAccess.serviceOff => (
        'Location services are off',
        BracketButton(
          'open',
          onTap: ctl.openSettings,
          style: IdrText.body.copyWith(color: IdrColors.accent),
        ),
      ),
      _ => (
        'Needed to find you and calibrate',
        BracketButton(
          'allow',
          onTap: ctl.request,
          style: IdrText.body.copyWith(color: IdrColors.accent),
        ),
      ),
    };

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(
        IdrSpace.gutter,
        IdrSpace.xxl,
        IdrSpace.gutter,
        IdrSpace.lg,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("Let's set up.", style: IdrText.title),
          const SizedBox(height: IdrSpace.lg),
          Text(
            "Two quick things and you're ready.",
            style: IdrText.label.copyWith(height: 1.5),
          ),
          const SizedBox(height: 40),
          Text(
            'What do you drive?',
            style: IdrText.small.copyWith(color: IdrColors.textSecondary),
          ),
          const SizedBox(height: IdrSpace.sm),
          TextField(
            controller: nameController,
            style: IdrText.body,
            cursorColor: IdrColors.accent,
            textCapitalization: TextCapitalization.words,
            textInputAction: TextInputAction.done,
            decoration: const InputDecoration(hintText: 'e.g. Hyundai Creta'),
          ),
          const SizedBox(height: IdrSpace.xl),
          Text(
            'Permissions',
            style: IdrText.small.copyWith(color: IdrColors.textSecondary),
          ),
          const SizedBox(height: IdrSpace.sm),
          IdrTile(
            padding: const EdgeInsets.fromLTRB(
              IdrSpace.lg,
              IdrSpace.lg,
              IdrSpace.lg,
              IdrSpace.lg,
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.location_on_outlined,
                  color: IdrColors.textPrimary,
                ),
                const SizedBox(width: IdrSpace.md),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Location', style: IdrText.body),
                      const SizedBox(height: 2),
                      Text(status, style: IdrText.small),
                    ],
                  ),
                ),
                action,
              ],
            ),
          ),
          const SizedBox(height: IdrSpace.xl),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.lock_outline,
                size: 16,
                color: IdrColors.textMuted,
              ),
              const SizedBox(width: IdrSpace.sm),
              Expanded(
                child: Text(
                  'Everything runs on your phone. Nothing is uploaded.',
                  style: IdrText.small.copyWith(color: IdrColors.textMuted),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
