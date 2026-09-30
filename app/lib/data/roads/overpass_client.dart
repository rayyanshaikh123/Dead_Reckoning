import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../engine/road_network.dart';

/// Downloads drivable roads from the public OpenStreetMap Overpass API and
/// converts them to the app's compact [RoadData] format (same filter as
/// tools/app_export/export_osm.py).
class OverpassClient {
  OverpassClient({http.Client? client}) : _client = client ?? http.Client();

  static const endpoints = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
  ];
  static const _highways =
      'motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|'
      'service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link';

  final http.Client _client;

  /// Roads inside the bounding box (degrees).
  Future<RoadData> fetch({
    required double south,
    required double west,
    required double north,
    required double east,
  }) async {
    final query =
        '[out:json][timeout:25];'
        'way["highway"~"^($_highways)\$"]["service"!~"parking_aisle|driveway"]["area"!="yes"]'
        '($south,$west,$north,$east);out body geom qt;';
    Object? lastError;
    for (final url in endpoints) {
      try {
        final res = await _client
            .post(
              Uri.parse(url),
              headers: {'User-Agent': 'IDR/1.0 (SIH dead-reckoning app)'},
              body: {'data': query},
            )
            .timeout(const Duration(seconds: 40));
        if (res.statusCode != 200) {
          lastError = 'HTTP ${res.statusCode}';
          continue;
        }
        return parse(jsonDecode(res.body) as Map<String, dynamic>);
      } catch (e) {
        lastError = e;
      }
    }
    throw OverpassException('$lastError');
  }

  /// Raw Overpass JSON → [RoadData].
  static RoadData parse(Map<String, dynamic> json) {
    final ways = <RoadWay>[];
    for (final el in (json['elements'] as List).cast<Map<String, dynamic>>()) {
      if (el['type'] != 'way' || el['geometry'] == null) continue;
      final tags = (el['tags'] as Map?)?.cast<String, dynamic>() ?? const {};
      final geom = (el['geometry'] as List).cast<Map<String, dynamic>>();
      ways.add(
        RoadWay(
          id: el['id'] as int,
          nodes: (el['nodes'] as List).cast<int>(),
          latLon: [
            for (final p in geom) ...[
              (p['lat'] as num).toDouble(),
              (p['lon'] as num).toDouble(),
            ],
          ],
          highway: (tags['highway'] as String?) ?? '',
          oneway: _oneway(tags),
          tunnel: tags['tunnel'] != null && tags['tunnel'] != 'no',
        ),
      );
    }
    return RoadData(ways);
  }

  static int _oneway(Map<String, dynamic> tags) {
    final ow = tags['oneway'];
    if (ow == 'yes' || ow == '1' || ow == 'true') return 1;
    if (ow == '-1') return -1;
    if (ow == 'no') return 0;
    final hw = tags['highway'];
    final junction = tags['junction'];
    if (hw == 'motorway' || hw == 'motorway_link') return 1;
    if (junction == 'roundabout' || junction == 'circular') return 1;
    return 0;
  }
}

class OverpassException implements Exception {
  OverpassException(this.message);

  final String message;

  @override
  String toString() => 'Road download failed: $message';
}
