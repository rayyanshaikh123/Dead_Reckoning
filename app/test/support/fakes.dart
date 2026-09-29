import 'dart:io';

import 'package:flutter_riverpod/misc.dart';
import 'package:idr_app/data/roads/overpass_client.dart';
import 'package:idr_app/data/roads/road_repository.dart';
import 'package:idr_app/engine/road_network.dart';

/// Overpass stand-in: never touches the network.
class FakeOverpass extends OverpassClient {
  FakeOverpass([this.data]);

  final RoadData? data;
  int calls = 0;

  @override
  Future<RoadData> fetch({
    required double south,
    required double west,
    required double north,
    required double east,
  }) async {
    calls++;
    final d = data;
    if (d == null) throw OverpassException('offline (test)');
    return d;
  }
}

/// Road providers backed by a temp directory and [FakeOverpass].
List<Override> roadOverrides({RoadData? data}) {
  final dir = Directory.systemTemp.createTempSync('idr_roads_');
  return [
    roadCacheProvider.overrideWithValue(RoadCache(Future.value(dir))),
    overpassClientProvider.overrideWithValue(FakeOverpass(data)),
  ];
}
