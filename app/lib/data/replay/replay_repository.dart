import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../engine/road_network.dart';
import 'replay.dart';

/// Python reference result for a bundled scenario (from index.json).
class ReplayReference {
  const ReplayReference({
    required this.alongTrackDriftPct,
    required this.exitErrorM,
  });

  final double alongTrackDriftPct;
  final double exitErrorM;
}

final replayReferencesProvider = FutureProvider<Map<String, ReplayReference>>((
  ref,
) async {
  final j = jsonDecode(
    await rootBundle.loadString('assets/replays/index.json'),
  ) as Map<String, dynamic>;
  return {
    for (final s in (j['scenarios'] as List).cast<Map<String, dynamic>>())
      s['id'] as String: ReplayReference(
        alongTrackDriftPct:
            ((s['reference'] as Map)['along_track_drift_pct'] as num)
                .toDouble(),
        exitErrorM: ((s['reference'] as Map)['exit_error_m'] as num).toDouble(),
      ),
  };
});

/// One bundled Drive M scenario, parsed.
final replayDataProvider = FutureProvider.family<ReplayData, String>((
  ref,
  id,
) async {
  final json = await rootBundle.loadString('assets/replays/$id.json');
  return ReplayData.parse(json);
});

/// OpenStreetMap roads around a bundled scenario (tools/app_export/export_osm.py).
final replayRoadsProvider = FutureProvider.family<RoadData, String>((
  ref,
  id,
) async {
  final json = await rootBundle.loadString('assets/replays/${id}_roads.json');
  return RoadData.fromJson(jsonDecode(json) as Map<String, dynamic>);
});
