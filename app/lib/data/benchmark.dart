/// IDNN v5 benchmark results on blind Drive M (README Tables 1 & 2).
class BenchmarkScenario {
  const BenchmarkScenario({
    required this.id,
    required this.name,
    required this.distanceM,
    required this.durationS,
    required this.insDriftPct,
    required this.v5AlongTrackPct,
    required this.v5ExitErrorM,
    required this.v5ExitPct,
  });

  /// Replay asset id (`assets/replays/<id>.json`).
  final String id;
  final String name;
  final double distanceM;
  final int durationS;
  final double insDriftPct;
  final double v5AlongTrackPct;
  final double v5ExitErrorM;
  final double v5ExitPct;

  /// SIH target: under 10% along-track drift (README Table 1).
  bool get passesSih => v5AlongTrackPct < 10;
}

const benchmarkScenarios = <BenchmarkScenario>[
  BenchmarkScenario(
    id: 'highway_tunnel',
    name: 'Straight highway tunnel',
    distanceM: 648,
    durationS: 30,
    insDriftPct: 121.8,
    v5AlongTrackPct: 1.9,
    v5ExitErrorM: 14.3,
    v5ExitPct: 2.2,
  ),
  BenchmarkScenario(
    id: 'short_underpass',
    name: 'Short underpass',
    distanceM: 219,
    durationS: 30,
    insDriftPct: 431.4,
    v5AlongTrackPct: 2.8,
    v5ExitErrorM: 6.5,
    v5ExitPct: 3.0,
  ),
  BenchmarkScenario(
    id: 'medium_tunnel',
    name: 'Medium tunnel (stop & go)',
    distanceM: 448,
    durationS: 60,
    insDriftPct: 733.0,
    v5AlongTrackPct: 9.7,
    v5ExitErrorM: 41.7,
    v5ExitPct: 9.3,
  ),
  BenchmarkScenario(
    id: 'mountain_tunnel',
    name: 'Long mountain tunnel',
    distanceM: 1680,
    durationS: 120,
    insDriftPct: 355.5,
    v5AlongTrackPct: 5.1,
    v5ExitErrorM: 85.6,
    v5ExitPct: 5.1,
  ),
  BenchmarkScenario(
    id: 'city_canyon',
    name: 'Complex city canyon',
    distanceM: 600,
    durationS: 90,
    insDriftPct: 508.8,
    v5AlongTrackPct: 14.3,
    v5ExitErrorM: 83.4,
    v5ExitPct: 13.9,
  ),
];
