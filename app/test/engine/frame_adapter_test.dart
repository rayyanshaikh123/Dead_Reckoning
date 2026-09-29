import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/engine/engine.dart';

const g = 9.80665;
final k = TrainingFrameAdapter.trainingMountDeg * math.pi / 180;

SensorFrame frame({
  required (double, double, double) acc,
  required (double, double, double) grav,
  (double, double, double) gyro = (0, 0, 0),
}) => SensorFrame(
  t: 0,
  ax: acc.$1,
  ay: acc.$2,
  az: acc.$3,
  gravX: grav.$1,
  gravY: grav.$2,
  gravZ: grav.$3,
  gyroX: gyro.$1,
  gyroY: gyro.$2,
  gyroZ: gyro.$3,
);

void main() {
  test('rotationToUp maps gravity onto +z for any orientation', () {
    final rnd = math.Random(3);
    for (var i = 0; i < 200; i++) {
      final v = (
        rnd.nextDouble() * 2 - 1,
        rnd.nextDouble() * 2 - 1,
        rnd.nextDouble() * 2 - 1,
      );
      final n = math.sqrt(v.$1 * v.$1 + v.$2 * v.$2 + v.$3 * v.$3);
      final (x, y, z) = rotationToUp(v.$1, v.$2, v.$3).apply(v.$1, v.$2, v.$3);
      expect(x, closeTo(0, 1e-9));
      expect(y, closeTo(0, 1e-9));
      expect(z, closeTo(n, 1e-9));
    }
    final (x, y, z) = rotationToUp(0, 0, -g).apply(0, 0, -g);
    expect(
      [x, y, z],
      [closeTo(0, 1e-12), closeTo(0, 1e-12), closeTo(g, 1e-12)],
    );
  });

  test('portrait dash mount: braking shows up on the training axes', () {
    // Phone upright, screen facing the driver: gravity along +y, car forward
    // out of the phone's back (−z). Car accelerates forward at 1 m/s².
    final a = TrainingFrameAdapter(); // default forward = +y after flattening
    final out = a.apply(frame(acc: (0, g, -1), grav: (0, g, 0)));
    expect(out.gravX, closeTo(0, 1e-9));
    expect(out.gravY, closeTo(0, 1e-9));
    expect(out.gravZ, closeTo(g, 1e-9));
    // Forward accel projected on the training axes (x is 28.5° left of forward).
    expect(out.ax - out.gravX, closeTo(math.cos(k), 1e-9));
    expect(out.ay - out.gravY, closeTo(-math.sin(k), 1e-9));
    expect(out.az - out.gravZ, closeTo(0, 1e-9));
  });

  test('gyro columns are rebuilt as [Yaw, Yaw, Roll]', () {
    final a = TrainingFrameAdapter()..forward = (1, 0);
    // Flat phone, rotation purely about the training x-axis.
    final wx = (math.cos(k), math.sin(k), 0.0);
    final out = a.apply(frame(acc: (0, 0, g), grav: (0, 0, g), gyro: wx));
    expect(out.gyroX, closeTo(1, 1e-9));
    expect(out.gyroY, closeTo(1, 1e-9));
    expect(out.gyroZ, closeTo(0, 1e-9));

    final b = TrainingFrameAdapter(gyroMapping: GyroMapping.yawIsY)
      ..forward = (1, 0);
    final out2 = b.apply(frame(acc: (0, 0, g), grav: (0, 0, g), gyro: wx));
    expect(out2.gyroX, closeTo(0, 1e-9));
    expect(out2.gyroZ, closeTo(1, 1e-9));
  });

  test('mount alignment learns the forward axis from cornering', () {
    // True forward in the flattened frame: 130°.
    const fwd = 130 * math.pi / 180;
    final f = (math.cos(fwd), math.sin(fwd));
    final left = (-f.$2, f.$1);
    final m = MountAlignment();
    final rnd = math.Random(1);
    for (var i = 0; i < 3000; i++) {
      const speed = 12.0;
      final omega = 0.15 * math.sin(i / 40); // alternating left/right bends
      final cent = speed * omega;
      final along = rnd.nextDouble() - 0.5; // braking/accelerating noise
      m.add(
        hx: cent * left.$1 + along * f.$1 + (rnd.nextDouble() - 0.5),
        hy: cent * left.$2 + along * f.$2 + (rnd.nextDouble() - 0.5),
        speedMs: speed,
        omegaUp: omega,
      );
    }
    expect(m.learned, isTrue);
    final (fx, fy) = m.forward!;
    final errDeg = (math.atan2(fy, fx) - fwd).abs() * 180 / math.pi;
    expect(errDeg, lessThan(3));
  });

  test('mount alignment ignores samples when parked or driving straight', () {
    final m = MountAlignment();
    for (var i = 0; i < 1000; i++) {
      m.add(hx: 1, hy: 0, speedMs: 0, omegaUp: 0.3);
      m.add(hx: 1, hy: 0, speedMs: 20, omegaUp: 0);
    }
    expect(m.learned, isFalse);
    expect(m.forward, isNull);
  });
}
