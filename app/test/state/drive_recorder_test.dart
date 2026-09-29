import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/data/drives/drive_store.dart';
import 'package:idr_app/data/model_loader.dart';
import 'package:idr_app/data/sources/gnss_source.dart';
import 'package:idr_app/data/sources/motion_source.dart';
import 'package:idr_app/engine/engine.dart';
import 'package:idr_app/state/drive_recorder.dart';
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

  Future<
    ({
      ProviderContainer c,
      Future<void> Function(double seconds, double speed) drive,
      Directory dir,
    })
  >
  setUpDrive() async {
    SharedPreferences.setMockInitialValues({});
    final prefs = await SharedPreferences.getInstance();
    final dir = Directory.systemTemp.createTempSync('idr_drives_');
    final motion = StreamController<SensorFrame>.broadcast();
    final gnss = StreamController<GnssFix?>.broadcast();
    final c = ProviderContainer(
      overrides: [
        ...roadOverrides(),
        sharedPrefsProvider.overrideWithValue(prefs),
        motionSamplesProvider.overrideWithValue(motion.stream),
        gnssProvider.overrideWith((ref) => gnss.stream),
        locationAccessProvider.overrideWith(_Granted.new),
        speedModelProvider.overrideWith(
          (ref) async => ModelStatus(model: model, msPerStep: 0),
        ),
        driveStoreProvider.overrideWithValue(DriveStore(Future.value(dir))),
      ],
    );
    addTearDown(c.dispose);
    addTearDown(motion.close);
    addTearDown(gnss.close);
    c.listen(driveRecorderProvider, (_, _) {});
    c.listen(navViewProvider, (_, _) {});
    await c.read(speedModelProvider.future);
    await Future<void>.delayed(Duration.zero);

    final rnd = math.Random(2);
    var t = 0.0, lat = 52.4;
    Future<void> drive(double seconds, double speed) async {
      final end = t + seconds;
      while (t < end) {
        lat += speed * 0.1 / 111320;
        gnss.add(
          GnssFix(
            latitude: lat,
            longitude: -1.5,
            accuracyM: 4,
            speedMs: speed,
            headingDeg: 0,
            time: DateTime.now(),
          ),
        );
        for (var k = 0; k < 5; k++) {
          motion.add(
            SensorFrame(
              t: t,
              ax: rnd.nextDouble() - 0.5,
              ay: 9.81,
              az: rnd.nextDouble() - 0.5,
              gravX: 0,
              gravY: 9.81,
              gravZ: 0,
              gyroX: 0,
              gyroY: 0,
              gyroZ: 0,
            ),
          );
          t += 0.02;
        }
        await Future<void>.delayed(Duration.zero);
      }
    }

    return (c: c, drive: drive, dir: dir);
  }

  test('a drive is recorded with its outage and sensor log', () async {
    final s = await setUpDrive();
    await s.drive(3, 15); // below the 5 s start threshold
    expect(s.c.read(driveRecorderProvider).recording, isFalse);
    await s.drive(10, 15);
    expect(s.c.read(driveRecorderProvider).recording, isTrue);

    s.c.read(tunnelTestProvider.notifier).toggle();
    await s.drive(4, 15);
    s.c.read(tunnelTestProvider.notifier).toggle();
    await s.drive(20, 15);
    await s.c.read(driveRecorderProvider.notifier).finish();
    expect(s.c.read(driveRecorderProvider).recording, isFalse);

    final drives = await s.c.read(drivesProvider.future);
    expect(drives, hasLength(1));
    final d = drives.single;
    expect(d.inProgress, isFalse);
    expect(d.distanceM, greaterThan(300));
    expect(d.outages, hasLength(1));
    expect(d.outages.single.simulated, isTrue);
    expect(d.outages.single.seconds, closeTo(4, 0.6));
    expect(d.track, isNotEmpty);

    final csv = File('${s.dir.path}/${d.id}.csv');
    await Future<void>.delayed(const Duration(milliseconds: 50));
    final lines = csv.readAsLinesSync();
    expect(lines.first, startsWith('TIME SINCE START (ms),GPS LATITUDE'));
    expect(lines.length, greaterThan(200));
    expect(lines.where((l) => l.endsWith(',test')), isNotEmpty);
  });

  test('a very short recording is discarded', () async {
    final s = await setUpDrive();
    await s.drive(8, 15); // starts after 5 s, ~45 m recorded
    expect(s.c.read(driveRecorderProvider).recording, isTrue);
    await s.c.read(driveRecorderProvider.notifier).finish();
    expect(await s.c.read(drivesProvider.future), isEmpty);
  });
}
