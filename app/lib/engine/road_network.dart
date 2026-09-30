import 'dart:math' as math;

import 'navigation.dart';

/// One OpenStreetMap way in the app's compact road format
/// (see tools/app_export/export_osm.py and lib/data/roads/).
class RoadWay {
  const RoadWay({
    required this.id,
    required this.nodes,
    required this.latLon,
    this.oneway = 0,
    this.highway = '',
    this.tunnel = false,
  });

  factory RoadWay.fromJson(Map<String, dynamic> j) => RoadWay(
    id: j['id'] as int,
    nodes: (j['n'] as List).cast<int>(),
    latLon: [for (final v in j['g'] as List) (v as num).toDouble()],
    oneway: (j['ow'] as int?) ?? 0,
    highway: (j['hw'] as String?) ?? '',
    tunnel: (j['t'] as int?) == 1,
  );

  final int id;
  final List<int> nodes;

  /// lat, lon, lat, lon, … (one pair per node).
  final List<double> latLon;

  /// 1 = only along node order, −1 = only against it, 0 = both ways.
  final int oneway;
  final String highway;
  final bool tunnel;

  Map<String, dynamic> toJson() => {
    'id': id,
    'n': nodes,
    'g': latLon,
    'hw': highway,
    'ow': oneway,
    't': tunnel ? 1 : 0,
  };
}

class RoadData {
  const RoadData(this.ways);

  factory RoadData.fromJson(Map<String, dynamic> j) => RoadData([
    for (final w in j['ways'] as List)
      RoadWay.fromJson(w as Map<String, dynamic>),
  ]);

  final List<RoadWay> ways;

  Map<String, dynamic> toJson() => {
    'ways': [for (final w in ways) w.toJson()],
  };

  RoadData merge(RoadData other) {
    final seen = {for (final w in ways) w.id};
    return RoadData([
      ...ways,
      for (final w in other.ways)
        if (seen.add(w.id)) w,
    ]);
  }
}

/// A directed road segment between two graph nodes.
class RoadEdge {
  RoadEdge(
    this.from,
    this.to,
    this.length,
    this.bearing,
    this.tunnel,
    this.way,
    this.rank,
  );

  final int from;
  final int to;
  final double length;

  /// Degrees clockwise from north.
  final double bearing;
  final bool tunnel;

  /// Index of the OSM way this edge belongs to.
  final int way;

  /// Road importance, 0 (motorway) … 6 (service).
  final int rank;
}

int _rank(String highway) => switch (highway.replaceAll('_link', '')) {
  'motorway' => 0,
  'trunk' => 1,
  'primary' => 2,
  'secondary' => 3,
  'tertiary' => 4,
  'unclassified' || 'residential' || 'living_street' => 5,
  _ => 6,
};

/// Where on the graph the vehicle is: an edge and the distance along it.
class EdgePosition {
  const EdgePosition(this.edge, this.offset);

  final int edge;
  final double offset;
}

/// Directed road graph in a local east/north plane.
class RoadGraph {
  RoadGraph(RoadData data, LocalTangentPlane plane) {
    final index = <int, int>{};
    int nodeOf(int osmId, double lat, double lon) =>
        index.putIfAbsent(osmId, () {
          final (e, n) = plane.toLocal(lat, lon);
          _x.add(e);
          _y.add(n);
          _out.add([]);
          return _x.length - 1;
        });

    for (var wi = 0; wi < data.ways.length; wi++) {
      final w = data.ways[wi];
      final rank = _rank(w.highway);
      for (var i = 0; i + 1 < w.nodes.length; i++) {
        final a = nodeOf(w.nodes[i], w.latLon[2 * i], w.latLon[2 * i + 1]);
        final b = nodeOf(
          w.nodes[i + 1],
          w.latLon[2 * i + 2],
          w.latLon[2 * i + 3],
        );
        if (a == b) continue;
        if (w.oneway >= 0) _addEdge(a, b, w.tunnel, wi, rank);
        if (w.oneway <= 0) _addEdge(b, a, w.tunnel, wi, rank);
      }
    }
  }

  final _x = <double>[];
  final _y = <double>[];
  final _out = <List<int>>[];
  final edges = <RoadEdge>[];

  int get nodeCount => _x.length;

  void _addEdge(int a, int b, bool tunnel, int way, int rank) {
    final dx = _x[b] - _x[a], dy = _y[b] - _y[a];
    final len = math.sqrt(dx * dx + dy * dy);
    final bearing = (math.atan2(dx, dy) * 180 / math.pi + 360) % 360;
    edges.add(RoadEdge(a, b, len, bearing, tunnel, way, rank));
    _out[a].add(edges.length - 1);
  }

  List<int> outgoing(int node) => _out[node];

  (double, double) pointAt(EdgePosition p) {
    final e = edges[p.edge];
    final f = e.length == 0 ? 0.0 : (p.offset / e.length).clamp(0.0, 1.0);
    return (
      _x[e.from] + (_x[e.to] - _x[e.from]) * f,
      _y[e.from] + (_y[e.to] - _y[e.from]) * f,
    );
  }

  /// Edges within [maxDistance] metres of (east, north), best first, with a
  /// starting cost for [RoadTracker]: distance (10 m = 1) plus heading
  /// disagreement. Edges more than [maxAngle] off [headingDeg] are skipped.
  List<(EdgePosition, double)> snapCandidates(
    double east,
    double north, {
    double? headingDeg,
    double maxDistance = 30,
    double maxAngle = 60,
    int limit = 8,
  }) {
    final out = <(EdgePosition, double)>[];
    for (var i = 0; i < edges.length; i++) {
      final e = edges[i];
      if (e.length == 0) continue;
      final ax = _x[e.from], ay = _y[e.from];
      final dx = _x[e.to] - ax, dy = _y[e.to] - ay;
      final t = (((east - ax) * dx + (north - ay) * dy) / (e.length * e.length))
          .clamp(0.0, 1.0);
      final px = ax + dx * t, py = ay + dy * t;
      final d = math.sqrt(
        (east - px) * (east - px) + (north - py) * (north - py),
      );
      if (d > maxDistance) continue;
      var cost = d / 10;
      if (headingDeg != null) {
        final diff = angleDiff(e.bearing, headingDeg);
        if (diff > maxAngle) continue;
        cost += (diff / 30) * (diff / 30);
      }
      out.add((EdgePosition(i, t * e.length), cost));
    }
    out.sort((a, b) => a.$2.compareTo(b.$2));
    return out.length > limit ? out.sublist(0, limit) : out;
  }

  /// The single best [snapCandidates] match, or null.
  EdgePosition? snap(
    double east,
    double north, {
    double? headingDeg,
    double maxDistance = 30,
    double maxAngle = 60,
  }) {
    final c = snapCandidates(
      east,
      north,
      headingDeg: headingDeg,
      maxDistance: maxDistance,
      maxAngle: maxAngle,
      limit: 1,
    );
    return c.isEmpty ? null : c.first.$1;
  }
}

/// Smallest absolute difference between two bearings, 0–180°.
double angleDiff(double a, double b) {
  final d = ((a - b) % 360 + 360) % 360;
  return d > 180 ? 360 - d : d;
}

class _Hypothesis {
  _Hypothesis(this.edge, this.offset, this.cost, this.scale, [this.drift = 0]);

  int edge;
  double offset;
  double cost;

  /// This candidate's estimate of the gyro heading's drift (degrees).
  double drift;

  /// This candidate's belief about the speed error: it moves scale × ds.
  final double scale;
}

/// Follows the road graph during an outage while keeping several candidate
/// positions alive (multi-hypothesis map matching).
///
/// * It starts on every nearby road that fits the heading, not just one.
/// * At a junction each candidate splits into all branches (no U-turns).
/// * Every metre driven, each candidate pays for disagreement between its
///   road's direction and the gyro heading; leaving the current road and
///   dropping to a minor road cost a little extra. Candidates far behind the
///   best are dropped. The best candidate is the reported position, so a
///   wrong early choice is corrected as soon as the heading disagrees.
/// * Candidates also differ in how far they think the car has travelled
///   ([scales] × the model's distance, with a prior favouring 1.0). A real
///   turn only matches candidates that reached a junction at that moment,
///   so turns act as landmarks that correct the model's distance error.
/// * Each candidate slowly learns the gyro heading's drift against its own
///   road (≈ [driftLearnM] metres, capped at ±[maxDriftDeg]). Slow drift
///   and under-read curves are absorbed, but a turn is sudden, so candidates
///   are effectively compared on *turns*. On IO-VNBD, wrong-road outages had
///   41° median absolute heading error vs 12° for correct ones, so matching
///   absolute heading was the main cause of wrong branches.
class RoadTracker {
  RoadTracker(
    this.graph,
    List<(EdgePosition, double)> starts, {
    // ±15 % in 7 steps: on Drive M this kept all five README tunnels under
    // 10 % (±20 % let the long mountain tunnel run 30 % ahead) while roughly
    // doubling the SIH pass rate on random outages vs a single guess.
    this.scales = const [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
    this.scalePriorCost = 0.6,
    this.scaleCostPer10m = 0,
    this.maxHypotheses = 40,
    this.pruneMargin = 10,
    this.headingScale = 30,
    this.maxMismatchDeg = 60,
    this.wayChangeCost = 0.8,
    this.rankDropCost = 0.4,
    this.deadEndCost = 2,
    this.driftLearnM = 60,
    this.maxDriftDeg = 45,
  }) : assert(starts.isNotEmpty),
       _hyps = [
         for (final (p, c) in starts)
           for (final k in scales)
             _Hypothesis(
               p.edge,
               p.offset,
               c + scalePriorCost * ((k - 1) / 0.1) * ((k - 1) / 0.1) / 4,
               k,
             ),
       ];

  final RoadGraph graph;

  /// Distance scale factors tried for every starting road.
  final List<double> scales;

  /// Prior cost of a scale 0.2 away from 1.0 (grows quadratically).
  final double scalePriorCost;

  /// Ongoing cost per 10 m for a scale 0.1 away from 1.0 (quadratic). A
  /// distance correction must keep earning its place: on winding roads with
  /// poor gyro heading, bends further along can otherwise "match" the bad
  /// heading better and pull the estimate ahead.
  final double scaleCostPer10m;

  /// Candidates kept after each step.
  final int maxHypotheses;

  /// Candidates costing more than best + this are dropped.
  final double pruneMargin;

  /// Heading disagreement (degrees) that costs 1 per 10 m driven…
  final double headingScale;

  /// …capped here, so a candidate survives ~25 m of disagreement (OSM
  /// junction nodes and real turning points can be metres apart).
  final double maxMismatchDeg;

  /// Extra cost for leaving the current OSM way at a junction.
  final double wayChangeCost;

  /// Extra cost per step down in road class.
  final double rankDropCost;

  /// Cost per 10 m a candidate is stuck at a dead end.
  final double deadEndCost;

  /// Distance over which a candidate adapts to the gyro heading's drift.
  final double driftLearnM;

  /// Largest drift a candidate may explain away; beyond it a mismatch is a
  /// real disagreement (e.g. the wrong branch at a T-junction).
  final double maxDriftDeg;

  List<_Hypothesis> _hyps;

  _Hypothesis get _best => _hyps.first;

  EdgePosition get position => EdgePosition(_best.edge, _best.offset);
  int get hypothesisCount => _hyps.length;
  bool get onTunnel => graph.edges[_best.edge].tunnel;

  (double, double) advance(double ds, double? headingDeg) {
    final next = <_Hypothesis>[];
    for (final h in _hyps) {
      final dev = (h.scale - 1) / 0.1;
      h.cost += scaleCostPer10m * dev * dev * ds / 10;
      _move(h, ds * h.scale, next, 0);
    }
    if (headingDeg != null) {
      for (final h in next) {
        final bearing = graph.edges[h.edge].bearing;
        final d = math.min(
          angleDiff(bearing, (headingDeg - h.drift) % 360),
          maxMismatchDeg,
        );
        h.cost += (d / headingScale) * (d / headingScale) * ds / 10;
        final err = (headingDeg - bearing + 540) % 360 - 180;
        h.drift += (err - h.drift) * math.min(1.0, ds * h.scale / driftLearnM);
        h.drift = h.drift.clamp(-maxDriftDeg, maxDriftDeg);
      }
    }
    _hyps = _prune(next);
    return graph.pointAt(position);
  }

  void _move(_Hypothesis h, double ds, List<_Hypothesis> out, int depth) {
    var rem = ds;
    // Bounded walk: tiny or degenerate edges can't loop forever.
    for (var steps = 0; steps < 500; steps++) {
      final e = graph.edges[h.edge];
      final left = e.length - h.offset;
      if (rem <= left) {
        h.offset += rem;
        out.add(h);
        return;
      }
      rem -= left;
      final next = [
        for (final o in graph.outgoing(e.to))
          if (graph.edges[o].to != e.from) o,
      ];
      if (next.isEmpty) {
        h
          ..offset = e.length
          ..cost += deadEndCost * rem / 10;
        out.add(h);
        return;
      }
      if (next.length == 1 || depth > 3) {
        h
          ..edge = next.first
          ..offset = 0;
        continue;
      }
      for (final n in next) {
        final to = graph.edges[n];
        var c = h.cost;
        if (to.way != e.way) c += wayChangeCost;
        if (to.rank > e.rank) c += rankDropCost * (to.rank - e.rank);
        _move(_Hypothesis(n, 0, c, h.scale, h.drift), rem, out, depth + 1);
      }
      return;
    }
    out.add(h);
  }

  List<_Hypothesis> _prune(List<_Hypothesis> hs) {
    // Candidates that converged on the same spot are one candidate.
    final byPlace = <(int, int, double), _Hypothesis>{};
    for (final h in hs) {
      final key = (h.edge, (h.offset / 5).floor(), h.scale);
      final seen = byPlace[key];
      if (seen == null || h.cost < seen.cost) byPlace[key] = h;
    }
    final sorted = byPlace.values.toList()
      ..sort((a, b) => a.cost.compareTo(b.cost));
    final limit = sorted.first.cost + pruneMargin;
    return [
      for (final h in sorted.take(maxHypotheses))
        if (h.cost <= limit) h,
    ];
  }
}

/// Keeps a dead-reckoned position on a road during a GNSS outage.
abstract interface class RoadGuide {
  /// Starts at the last known position; false if there's no road to follow.
  bool begin(double east, double north, double? headingDeg);

  /// Moves [ds] metres along the road; returns the new position.
  (double, double) advance(double ds, double? headingDeg);
}

/// Follows one known polyline (the benchmark's ground-truth road).
class PolylineGuide implements RoadGuide {
  PolylineGuide(this.road, {this.startDistance = 0});

  final PolylineMatcher road;
  final double startDistance;
  double _d = 0;

  @override
  bool begin(double east, double north, double? headingDeg) {
    _d = startDistance;
    return true;
  }

  @override
  (double, double) advance(double ds, double? headingDeg) => road.at(_d += ds);
}

/// Follows the OpenStreetMap road graph.
class GraphGuide implements RoadGuide {
  GraphGuide(this.graph, {this.maxSnapDistance = 30, this.newTracker});

  final RoadGraph graph;
  final double maxSnapDistance;

  /// Builds the tracker (tests and tuning); defaults to [RoadTracker].
  final RoadTracker Function(RoadGraph, List<(EdgePosition, double)>)?
  newTracker;
  RoadTracker? _tracker;

  @override
  bool begin(double east, double north, double? headingDeg) {
    final starts = graph.snapCandidates(
      east,
      north,
      headingDeg: headingDeg,
      maxDistance: maxSnapDistance,
    );
    _tracker = starts.isEmpty
        ? null
        : (newTracker ?? RoadTracker.new)(graph, starts);
    return starts.isNotEmpty;
  }

  @override
  (double, double) advance(double ds, double? headingDeg) =>
      _tracker!.advance(ds, headingDeg);
}
