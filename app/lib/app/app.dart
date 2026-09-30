import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../core/theme/theme.dart';
import '../state/drive_recorder.dart';
import '../state/nav_state.dart';
import 'routes.dart';

class IdrApp extends ConsumerStatefulWidget {
  const IdrApp({super.key});

  @override
  ConsumerState<IdrApp> createState() => _IdrAppState();
}

class _IdrAppState extends ConsumerState<IdrApp> {
  late final GoRouter _router = buildRouter();

  @override
  void initState() {
    super.initState();
    // Keep the speed history sampling from launch, not just while Live is open.
    ref.listenManual(speedHistoryProvider, (_, _) {});
    // Drives are recorded in the background from launch.
    ref.listenManual(driveRecorderProvider, (_, _) {});
  }

  @override
  void dispose() {
    _router.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'IDR',
      debugShowCheckedModeBanner: false,
      theme: buildIdrTheme(),
      darkTheme: buildIdrTheme(),
      themeMode: ThemeMode.dark,
      routerConfig: _router,
    );
  }
}
