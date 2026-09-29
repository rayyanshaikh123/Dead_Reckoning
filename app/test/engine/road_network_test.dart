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

  test(
    'tracker stays on the main road at a junction when heading is straight',
    () {
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
      final t = RoadTracker(g, g.snap(0, 10, headingDeg: 0)!);
      final (e, n) = t.advance(150, 0);
      expect(e, closeTo(0, 0.01));
      expect(n, closeTo(160, 0.01));
    },
  );

  test(
    'tracker turns when the heading turns, even a little after the junction',
    () {
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
      final t = RoadTracker(g, g.snap(0, 10, headingDeg: 0)!);
      // Crosses the junction heading north → straight on, 5 m past it.
      t.advance(95, 0);
      t.advance(10, 0); // 15 m past, still north
      // Heading swings east 45 m after the junction → re-routed onto the
      // east branch, 45 m along it.
      final (e, n) = t.advance(30, 90);
      expect(n, closeTo(100, 0.01));
      expect(e, closeTo(45, 0.01));
    },
  );

  test(
    'continuity: a small heading wobble does not leave the current road',
    () {
      // Main road bends 20° right at the junction; a side road goes 10° left.
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
      final t = RoadTracker(g, g.snap(0, 10, headingDeg: 0)!);
      final (e, _) = t.advance(150, 0); // gyro says "straight"
      expect(
        e,
        greaterThan(0),
        reason: 'followed the main road, not the side road',
      );
    },
  );
}
