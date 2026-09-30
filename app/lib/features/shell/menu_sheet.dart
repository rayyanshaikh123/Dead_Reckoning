import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../app/routes.dart';
import '../../core/theme/theme.dart';

/// The "[menu]" sheet: secondary destinations that don't live in the tab bar.
Future<void> showIdrMenu(BuildContext context) {
  return showModalBottomSheet<void>(
    context: context,
    builder: (sheetContext) {
      void go(String location) {
        Navigator.of(sheetContext).pop();
        context.push(location);
      }

      return SafeArea(
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
              Text(
                'menu',
                style: IdrText.small.copyWith(color: IdrColors.textMuted),
              ),
              const SizedBox(height: IdrSpace.md),
              _MenuItem(
                'Simulation',
                'drive a real route, open and close tunnels yourself',
                () => go(Routes.simulation),
              ),
              _MenuItem(
                'Replay',
                'watch IDR handle real tunnels',
                () => go(Routes.replay),
              ),
              _MenuItem(
                'History',
                'your past drives',
                () => go(Routes.history),
              ),
              if (kDebugMode)
                _MenuItem(
                  'Widget gallery',
                  'design system (debug only)',
                  () => go(Routes.gallery),
                ),
            ],
          ),
        ),
      );
    },
  );
}

class _MenuItem extends StatelessWidget {
  const _MenuItem(this.title, this.caption, this.onTap);

  final String title;
  final String caption;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: IdrSpace.md),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('[$title]', style: IdrText.body),
                  const SizedBox(height: 2),
                  Text(caption, style: IdrText.small),
                ],
              ),
            ),
            const Icon(
              Icons.arrow_forward,
              size: 20,
              color: IdrColors.textSecondary,
            ),
          ],
        ),
      ),
    );
  }
}
