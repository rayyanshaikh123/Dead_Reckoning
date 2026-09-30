import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/data/replay/replay.dart';
import 'package:idr_app/engine/engine.dart';

/// Gravity pointing along (pitch, roll) degrees from the phone's +z.
SensorFrame tilted(double pitchDeg, double rollDeg) {
  final p = pitchDeg * math.pi / 180, r = rollDeg * math.pi / 180;
  final gx = 9.81 * math.sin(r),
      gy = 9.81 * math.sin(p) * math.cos(r),
      gz = 9.81 * math.cos(p) * math.cos(r);
  return SensorFrame(
    t: 0,
    ax: gx,
    ay: gy,
    az: gz,
    gravX: gx,
    gravY: gy,
    gravZ: gz,
    gyroX: 0,
    gyroY: 0,
    gyroZ: 0,
  );
}

void main() {
  test('mounted phone on real roads (Drive M, all 5 clips) reads as steady', () {
    for (final id in [
      'highway_tunnel',
      'short_underpass',
      'medium_tunnel',
      'mountain_tunnel',
      'city_canyon',
    ]) {
      final frames = ReplayData.parse(
        File('assets/replays/$id.json').readAsStringSync(),
      ).frames();
      final m = MountStability();
      var unsteady = 0;
      for (final f in frames) {
        m.add(f);
        if (!m.steady) unsteady++;
      }
      expect(
        unsteady / frames.length,
        lessThan(0.005),
        reason:
            '$id: ${(100 * unsteady / frames.length).toStringAsFixed(2)}% flagged',
      );
    }
  });

  test('car pitching on a hill and braking stays steady', () {
    final m = MountStability();
    for (var i = 0; i < 600; i++) {
      // 6° hill building up over 60 s, ±1° braking dips.
      m.add(tilted(70 + 6 * i / 600 + math.sin(i / 15), 0.5 * math.sin(i / 7)));
      expect(m.steady, isTrue, reason: 'sample $i');
    }
  });

  test('phone in a hand is flagged within a few seconds, and recovers when mounted', () {
    final m = MountStability();
    final rnd = math.Random(4);
    for (var i = 0; i < 100; i++) {
      m.add(tilted(70, 0));
    }
    expect(m.steady, isTrue);

    // Hand-held: slow drifts of ±10° plus tremor.
    var flaggedAt = -1;
    for (var i = 0; i < 100; i++) {
      m.add(
        tilted(
          60 + 10 * math.sin(i / 12) + rnd.nextDouble() * 2,
          8 * math.cos(i / 17) + rnd.nextDouble() * 2,
        ),
      );
      if (!m.steady && flaggedAt < 0) flaggedAt = i;
    }
    expect(flaggedAt, inInclusiveRange(0, 40), reason: 'flagged within 4 s');

    // Back in the mount at the same angle: steady again after ~10 s.
    var steadyAt = -1;
    for (var i = 0; i < 250; i++) {
      m.add(tilted(70, 0));
      if (m.steady && steadyAt < 0) steadyAt = i;
      expect(m.remounted, isFalse);
    }
    expect(steadyAt, inInclusiveRange(100, 170));
  });

  test('settling at a new angle is reported as a remount', () {
    final m = MountStability();
    for (var i = 0; i < 100; i++) {
      m.add(tilted(70, 0));
    }
    for (var i = 0; i < 60; i++) {
      m.add(tilted(70 - i * 0.75, i * 0.6)); // moved by hand
    }
    var remounts = 0;
    for (var i = 0; i < 250; i++) {
      m.add(tilted(25, 36)); // new mount position
      if (m.remounted) remounts++;
    }
    expect(m.steady, isTrue);
    expect(remounts, 1);
  });
}
