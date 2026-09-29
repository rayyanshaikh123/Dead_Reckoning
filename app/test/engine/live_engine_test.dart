// Wiring test for the live engine: fake 50 Hz motion + fake GNSS through the
// real providers, then a tunnel test.

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:idr_app/data/model_loader.dart';
import 'package:idr_app/data/sources/gnss_source.dart';
import 'package:idr_app/data/sources/motion_source.dart';
import 'package:idr_app/engine/engine.dart';
import 'package:idr_app/state/live_engine.dart';
import 'package:idr_app/state/nav_state.dart';
import 'package:idr_app/state/settings.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../support/fakes.dart';

class _Granted extends LocationAccessController {
  @override
  LocationAccess build() => LocationAccess.granted;
}

void main() {
  late IdnnV5 model;

  setUpAll(() {
    model = IdnnV5.fromBytes(
      manifestJson: File('assets/models/idnn_v5.json').readAsStringSync(),
      weights: ByteData.sublistView(
        File('assets/models/idnn_v5.bin').readAsBytesSync(),
      ),
    );
  });

  test('decimator turns ~50 Hz into 10 Hz and re-anchors after a pause', () {
    final d = Decimator();
    SensorFrame at(double t) => SensorFrame(
      t: t,
      ax: 0,
      ay: 0,
      az: 9.8,
      gravX: 0,
      gravY: 0,
      gravZ: 9.8,
      gyroX: 0,
      gyroY: 0,
      gyroZ: 0,
    );
    var n = 0;
    for (var i = 0; i < 500; i++) {
      if (d.push(at(i * 0.02 + 0.003 * math.sin(i.toDouble()))) != null) n++;
    }
    expect(n, inInclusiveRange(98, 101)); // 10 s → ~100 frames
    // A 5 s gap produces one frame, not a burst of 50.
    var burst = 0;
    for (var i = 0; i < 5; i++) {
      if (d.push(at(15.0 + i * 0.02)) != null) burst++;
    }
    expect(burst, 1);
  });

  test('gravity fallback estimates gravity when the sensor is missing', () {
    final gf = GravityFallback();
    late SensorFrame out;
    for (var i = 0; i < 500; i++) {
      out = gf.apply(
        SensorFrame(
          t: i * 0.02,
          ax: 0.5 * math.sin(i / 3),
          ay: 9.81,
          az: 0.3 * math.cos(i / 5),
          gravX: double.nan,
          gravY: double.nan,
          gravZ: double.nan,
          gyroX: 0,
          gyroY: 0,
          gyroZ: 0,
        ),
      );
    }
    expect(out.gravY, closeTo(9.81, 0.05));
    expect(out.gravX.abs(), lessThan(0.2));
  });

  test(
    'live engine: GNSS feeds it, tunnel test switches to sensor mode',
    () async {
      SharedPreferences.setMockInitialValues({});
      final prefs = await SharedPreferences.getInstance();
      final motion = StreamController<SensorFrame>.broadcast();
      final gnss = StreamController<GnssFix?>.broadcast();
      final container = ProviderContainer(
        overrides: [
          ...roadOverrides(
            data: RoadData([
              // A road running north along the test drive.
              RoadWay(
                id: 1,
                nodes: [1, 2],
                latLon: [52.39, -1.5, 52.42, -1.5],
                highway: 'primary',
              ),
            ]),
          ),
          sharedPrefsProvider.overrideWithValue(prefs),
          motionSamplesProvider.overrideWithValue(motion.stream),
          gnssProvider.overrideWith((ref) => gnss.stream),
          locationAccessProvider.overrideWith(_Granted.new),
          speedModelProvider.overrideWith(
            (ref) async => ModelStatus(model: model, msPerStep: 0),
          ),
        ],
      );
      addTearDown(container.dispose);
      addTearDown(motion.close);
      addTearDown(gnss.close);

      // Keep the providers alive and let the model future resolve.
      container.listen(navViewProvider, (_, _) {});
      await container.read(speedModelProvider.future);
      await Future<void>.delayed(Duration.zero);
      expect(container.read(liveEngineProvider).running, isTrue);

      // Portrait phone in a dash mount, car cruising at ~15 m/s.
      final rnd = math.Random(5);
      var t = 0.0;
      Future<void> drive(double seconds) async {
        final end = t + seconds;
        while (t < end) {
          gnss.add(
            GnssFix(
              latitude: 52.4 + t * 1e-5,
              longitude: -1.5,
              accuracyM: 4,
              speedMs: 15,
              headingDeg: 0,
              time: DateTime.now(),
            ),
          );
          for (var k = 0; k < 5; k++) {
            motion.add(
              SensorFrame(
                t: t,
                ax: rnd.nextDouble() - 0.5,
                ay: 9.81 + rnd.nextDouble() - 0.5,
                az: rnd.nextDouble() - 0.5,
                gravX: 0,
                gravY: 9.81,
                gravZ: 0,
                gyroX: 0.01 * (rnd.nextDouble() - 0.5),
                gyroY: 0.01 * (rnd.nextDouble() - 0.5),
                gyroZ: 0,
              ),
            );
            t += 0.02;
          }
          await Future<void>.delayed(Duration.zero);
        }
      }

      await drive(10);
      var e = container.read(liveEngineProvider);
      expect(e.frames, inInclusiveRange(95, 101));
      expect(e.hadGnss, isTrue);
      expect(e.gnssUsed, isTrue);
      expect(e.gravityMagnitude, closeTo(9.81, 1e-6));
      expect(e.output, isNotNull);
      expect(e.output!.blackout, isFalse);
      expect(container.read(navViewProvider).mode, NavMode.gps);

      // Road data loads asynchronously (file cache); under load it can lag
      // the simulated drive, so wait for it before starting the outage.
      for (var i = 0; i < 200; i++) {
        if (container.read(liveEngineProvider).roadsLoaded) break;
        await Future<void>.delayed(const Duration(milliseconds: 10));
        await drive(0.1);
      }
      expect(container.read(liveEngineProvider).roadsLoaded, isTrue);

      container.read(tunnelTestProvider.notifier).toggle();
      await drive(3);
      e = container.read(liveEngineProvider);
      expect(e.gnssUsed, isFalse);
      expect(e.output!.blackout, isTrue);
      expect(e.roadsLoaded, isTrue);
      expect(e.output!.onRoad, isTrue, reason: 'snapped to the OSM road');
      final v = container.read(navViewProvider);
      expect(v.mode, NavMode.sensors);
      expect(v.simulated, isTrue);
      expect(v.tunnelSeconds, inInclusiveRange(2, 3));

      container.read(tunnelTestProvider.notifier).toggle();
      await drive(1);
      expect(container.read(liveEngineProvider).output!.blackout, isFalse);
      expect(container.read(navViewProvider).mode, NavMode.gps);
    },
  );
}
