import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/fakes.dart';

import 'package:shared_preferences/shared_preferences.dart';

import 'package:idr_app/app/app.dart';
import 'package:idr_app/core/widgets/idr_map.dart';
import 'package:idr_app/data/sources/gnss_source.dart';
import 'package:idr_app/data/sources/motion_source.dart';
import 'package:idr_app/engine/frame.dart';
import 'package:idr_app/state/settings.dart';

class _GrantedAccess extends LocationAccessController {
  @override
  LocationAccess build() => LocationAccess.granted;

  @override
  Future<LocationAccess> refresh() async => state;
}

GnssFix _fix({double speedMs = 0}) => GnssFix(
  latitude: 18.94,
  longitude: 72.82,
  accuracyM: 4,
  speedMs: speedMs,
  headingDeg: 40,
  time: DateTime.now(),
);

void main() {
  setUpAll(() => IdrMap.showTiles = false);

  Future<void> pumpApp(
    WidgetTester tester, {
    required Map<String, Object> prefs,
    GnssFix? fix,
  }) async {
    tester.view.physicalSize = const Size(1179, 2556);
    tester.view.devicePixelRatio = 3;
    addTearDown(tester.view.reset);
    SharedPreferences.setMockInitialValues(prefs);
    final sp = await SharedPreferences.getInstance();
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          ...roadOverrides(),
          sharedPrefsProvider.overrideWithValue(sp),
          locationAccessProvider.overrideWith(_GrantedAccess.new),
          gnssProvider.overrideWith((ref) => Stream.value(fix)),
          motionSamplesProvider.overrideWithValue(
            const Stream<SensorFrame>.empty(),
          ),
        ],
        child: const IdrApp(),
      ),
    );
    // Let the splash animation finish and route onward.
    await tester.pump(const Duration(seconds: 3));
    await tester.pump(const Duration(milliseconds: 500));
  }

  testWidgets('first run: splash → intro → setup → home', (tester) async {
    await pumpApp(tester, prefs: {});
    expect(find.text('GPS drops in tunnels.'), findsOneWidget);

    for (var i = 0; i < 3; i++) {
      await tester.tap(find.text('Next'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 600));
    }
    expect(find.text("Let's set up."), findsOneWidget);
    expect(find.text('Allowed'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'Hyundai Creta');
    await tester.tap(find.text('Get started'));
    await tester.pump(const Duration(milliseconds: 600));

    expect(find.text('Hyundai Creta'), findsOneWidget);
  });

  testWidgets('returning user lands on home with live GPS', (tester) async {
    await pumpApp(
      tester,
      prefs: {'onboarded': true, 'vehicle_name': 'Creta'},
      fix: _fix(speedMs: 10),
    );
    expect(find.text('Creta'), findsOneWidget);
    expect(find.text('Driving · GPS'), findsOneWidget);
    expect(find.text('36'), findsOneWidget); // 10 m/s
    expect(find.text('±4 m'), findsOneWidget);
  });

  testWidgets('tunnel test switches to sensor mode and back', (tester) async {
    await pumpApp(tester, prefs: {'onboarded': true}, fix: _fix(speedMs: 10));
    await tester.tap(find.bySemanticsLabel('Tunnel test'));
    await tester.pump(const Duration(seconds: 2));
    expect(find.text('No GPS · sensors'), findsOneWidget);
    expect(find.text('estimated'), findsOneWidget);

    await tester.tap(find.bySemanticsLabel('Tunnel test'));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('Driving · GPS'), findsOneWidget);
  });

  testWidgets('every tab renders', (tester) async {
    await pumpApp(tester, prefs: {'onboarded': true}, fix: _fix());
    for (final label in ['Live', 'Map', 'Settings', 'Home']) {
      await tester.tap(find.bySemanticsLabel(label));
      await tester.pump(const Duration(milliseconds: 400));
      expect(tester.takeException(), isNull, reason: '$label tab threw');
    }
  });
}
