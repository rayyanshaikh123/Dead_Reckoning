import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/engine/engine.dart';

// A tiny synthetic map around (52.4, -1.5). Coordinates are built from
// local metres so the geometry is easy to reason about.
final plane = LocalTangentPlane(52.4, -1.5);

List<double> pts(List<(double, double)> en) => [
  for (final (e, n) in en) ...[plane.toGeo(e, n).$1, plane.toGeo(e, n).$2],
];

void main() {
  test('snap picks the nearby road that matches the heading', () {
    // Main road running north, side road running east from (0, 100).
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2, 3],
          latLon: pts([(0, 0), (0, 100), (0, 200)]),
          highway: 'primary',
        ),
        RoadWay(
          id: 2,
          nodes: [2, 4],
          latLon: pts([(0, 100), (100, 100)]),
          highway: 'residential',
        ),
      ]),
      plane,
    );
    final north = g.snap(3, 50, headingDeg: 0)!;
    expect(g.edges[north.edge].bearing, closeTo(0, 1e-6));
    expect(north.offset, closeTo(50, 0.01));
    final south = g.snap(3, 50, headingDeg: 180)!;
    expect(g.edges[south.edge].bearing, closeTo(180, 1e-6));
    expect(g.snap(80, 50, headingDeg: 0), isNull); // too far from any road
  });

  test('one-way roads only snap in their direction', () {
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2],
          latLon: pts([(0, 0), (0, 100)]),
          oneway: 1,
          highway: 'primary',
        ),
      ]),
      plane,
    );
    expect(g.edges.length, 1);
    expect(g.snap(0, 50, headingDeg: 180), isNull);
  });

  // Helper: a tracker started where snapping would start it.
  RoadTracker start(RoadGraph g, double e, double n, double heading) =>
      RoadTracker(g, g.snapCandidates(e, n, headingDeg: heading));

  test('stays on the main road at a junction when heading is straight', () {
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2, 3],
          latLon: pts([(0, 0), (0, 100), (0, 200)]),
          highway: 'primary',
        ),
        RoadWay(
          id: 2,
          nodes: [2, 4],
          latLon: pts([(0, 100), (100, 110)]),
          highway: 'residential',
        ),
      ]),
      plane,
    );
    final t = start(g, 0, 10, 0);
    late (double, double) p;
    for (var i = 0; i < 100; i++) {
      p = t.advance(1.5, 0);
    }
    expect(p.$1, closeTo(0, 0.01));
    expect(p.$2, closeTo(160, 0.01));
  });

  test('turns when the heading turns, even a little after the junction', () {
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2, 3],
          latLon: pts([(0, 0), (0, 100), (0, 200)]),
          highway: 'primary',
        ),
        RoadWay(
          id: 2,
          nodes: [2, 4],
          latLon: pts([(0, 100), (200, 100)]),
          highway: 'primary',
        ),
      ]),
      plane,
    );
    final t = start(g, 0, 10, 0);
    late (double, double) p;
    for (var i = 0; i < 70; i++) {
      p = t.advance(1.5, 0); // 105 m: 15 m past the junction, still "north"
    }
    for (var i = 0; i < 20; i++) {
      p = t.advance(1.5, 90); // heading swings east
    }
    expect(p.$2, closeTo(100, 0.5), reason: 'on the east branch');
    // The late turn is also read as a landmark ("the junction must have come
    // later than the model's distance said"), so the position along the
    // east branch may be pulled back from the 45 m the model reported.
    expect(p.$1, inInclusiveRange(15.0, 60.0));
  });

  test('stays on the main road through its bend, ignoring a side road', () {
    // Main road bends 20° right at the junction; a service road goes 10° left.
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2, 3],
          latLon: pts([(0, 0), (0, 100), (34.2, 194)]),
          highway: 'secondary',
        ),
        RoadWay(
          id: 2,
          nodes: [2, 4],
          latLon: pts([(0, 100), (-17.4, 198.5)]),
          highway: 'service',
        ),
      ]),
      plane,
    );
    final t = start(g, 0, 10, 0);
    final rnd = math.Random(3);
    late (double, double) p;
    for (var i = 0; i < 100; i++) {
      // Gyro follows the car along the main road, with ±5° noise.
      final along = 10 + 1.5 * i;
      p = t.advance(
        1.5,
        (along < 100 ? 0 : 20) + (rnd.nextDouble() - 0.5) * 10,
      );
    }
    expect(p.$1, greaterThan(0), reason: 'followed the main road');
  });

  test('turns act as landmarks: a 15 % distance error is corrected', () {
    // T-junction at 200 m: the road continues north, a branch goes east.
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2, 3],
          latLon: pts([(0, 0), (0, 200), (0, 400)]),
          highway: 'primary',
        ),
        RoadWay(
          id: 2,
          nodes: [2, 4],
          latLon: pts([(0, 200), (300, 200)]),
          highway: 'primary',
        ),
      ]),
      plane,
    );
    final t = start(g, 0, 0, 0);
    late (double, double) p;
    // True drive: 200 m north, then 100 m east; the model reports 85 %.
    for (var d = 0.0; d < 300; d += 1.5) {
      p = t.advance(1.5 * 0.85, d < 200 ? 0 : 90);
    }
    expect(p.$2, closeTo(200, 0.5), reason: 'took the east branch');
    expect(p.$1, closeTo(100, 20), reason: 'distance corrected by the turn');
  });

  test('starting between parallel roads, ends on the one the car follows', () {
    // Two northbound roads 15 m apart; at y = 100 road B bends north-east.
    final g = RoadGraph(
      RoadData([
        RoadWay(
          id: 1,
          nodes: [1, 2, 3],
          latLon: pts([(0, 0), (0, 100), (0, 300)]),
          oneway: 1,
          highway: 'primary',
        ),
        RoadWay(
          id: 2,
          nodes: [4, 5, 6],
          latLon: pts([(15, 0), (15, 100), (156, 241)]),
          oneway: 1,
          highway: 'primary',
        ),
      ]),
      plane,
    );
    // Start 6 m from A, 9 m from B — but the car is on B.
    final t = start(g, 6, 10, 0);
    late (double, double) p;
    for (var d = 10.0; d < 250; d += 1.5) {
      p = t.advance(1.5, d < 100 ? 0 : 45);
    }
    expect(p.$1, greaterThan(60), reason: 'on road B heading north-east');
  });
}
