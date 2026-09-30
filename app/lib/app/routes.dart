import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../core/theme/theme.dart';
import '../core/widgets/widgets.dart';
import '../features/dashboard/dashboard_screen.dart';
import '../features/gallery/gallery_screen.dart';
import '../features/history/drive_detail_screen.dart';
import '../features/history/history_screen.dart';
import '../features/live/live_screen.dart';
import '../features/map/map_screen.dart';
import '../features/onboarding/onboarding_screen.dart';
import '../features/profile/profile_screen.dart';
import '../features/replay/replay_player_screen.dart';
import '../features/replay/replay_screen.dart';
import '../features/simulation/simulation_screen.dart';
import '../features/splash/splash_screen.dart';

abstract final class Routes {
  static const splash = '/splash';
  static const onboarding = '/onboarding';
  static const live = '/live';
  static const dashboard = '/dashboard';
  static const map = '/map';
  static const profile = '/profile';
  static const history = '/history';
  static const replay = '/replay';
  static const simulation = '/simulation';
  static const gallery = '/gallery';
}

const _navItems = [
  IdrNavItem(icon: Icons.bolt_outlined, activeIcon: Icons.bolt, label: 'Live'),
  IdrNavItem(
    icon: Icons.grid_view_outlined,
    activeIcon: Icons.grid_view_rounded,
    label: 'Home',
  ),
  IdrNavItem(icon: Icons.map_outlined, activeIcon: Icons.map, label: 'Map'),
  IdrNavItem(
    icon: Icons.settings_outlined,
    activeIcon: Icons.settings,
    label: 'Settings',
  ),
];

GoRouter buildRouter() {
  return GoRouter(
    initialLocation: Routes.splash,
    routes: [
      GoRoute(path: Routes.splash, builder: (_, _) => const SplashScreen()),
      GoRoute(
        path: Routes.onboarding,
        pageBuilder: (_, state) => NoTransitionPage(
          key: state.pageKey,
          child: const OnboardingScreen(),
        ),
      ),
      StatefulShellRoute.indexedStack(
        builder: (context, state, shell) => Scaffold(
          backgroundColor: IdrColors.background,
          body: shell,
          bottomNavigationBar: IdrBottomNav(
            items: _navItems,
            index: shell.currentIndex,
            onSelect: (i) =>
                shell.goBranch(i, initialLocation: i == shell.currentIndex),
          ),
        ),
        branches: [
          StatefulShellBranch(
            routes: [
              GoRoute(path: Routes.live, builder: (_, _) => const LiveScreen()),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: Routes.dashboard,
                builder: (_, _) => const DashboardScreen(),
              ),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(path: Routes.map, builder: (_, _) => const MapScreen()),
            ],
          ),
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: Routes.profile,
                builder: (_, _) => const ProfileScreen(),
              ),
            ],
          ),
        ],
      ),
      GoRoute(
        path: Routes.history,
        builder: (_, _) => const HistoryScreen(),
        routes: [
          GoRoute(
            path: ':id',
            builder: (_, state) =>
                DriveDetailScreen(id: state.pathParameters['id']!),
          ),
        ],
      ),
      GoRoute(
        path: Routes.replay,
        builder: (_, _) => const ReplayScreen(),
        routes: [
          GoRoute(
            path: ':id',
            builder: (_, state) =>
                ReplayPlayerScreen(id: state.pathParameters['id']!),
          ),
        ],
      ),
      GoRoute(
        path: Routes.simulation,
        builder: (_, _) => const SimulationScreen(),
      ),
      GoRoute(path: Routes.gallery, builder: (_, _) => const GalleryScreen()),
    ],
  );
}
