import 'dart:async';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_app/data/roads/overpass_client.dart';
import 'package:idr_app/data/roads/road_repository.dart';
import 'package:idr_app/data/sources/gnss_source.dart';
import 'package:idr_app/engine/road_network.dart';

import '../support/fakes.dart';

final _road = RoadData([
  RoadWay(
    id: 1,
    nodes: [1, 2],
    latLon: [52.39, -1.5, 52.42, -1.5],
    highway: 'primary',
  ),
]);

GnssFix _fix() => GnssFix(
  latitude: 52.405,
  longitude: -1.505,
  accuracyM: 5,
  speedMs: 10,
  headingDeg: 0,
  time: DateTime.now(),
);

void main() {
  test('Overpass JSON is parsed with OSM one-way rules', () {
    final data = OverpassClient.parse({
      'elements': [
        {
          'type': 'way',
          'id': 7,
          'nodes': [1, 2],
          'tags': {'highway': 'motorway'},
          'geometry': [
            {'lat': 52.0, 'lon': -1.0},
            {'lat': 52.1, 'lon': -1.0},
          ],
        },
        {
          'type': 'way',
          'id': 8,
          'nodes': [3, 4],
          'tags': {'highway': 'residential', 'oneway': '-1', 'tunnel': 'yes'},
          'geometry': [
            {'lat': 52.0, 'lon': -1.1},
            {'lat': 52.1, 'lon': -1.1},
          ],
        },
        {
          'type': 'way',
          'id': 9,
          'nodes': [5, 6],
          'tags': {
            'highway': 'primary',
            'junction': 'roundabout',
            'tunnel': 'no',
          },
          'geometry': [
            {'lat': 52.0, 'lon': -1.2},
            {'lat': 52.1, 'lon': -1.2},
          ],
        },
        {'type': 'node', 'id': 99},
      ],
    });
    expect(data.ways.map((w) => w.oneway), [1, -1, 1]);
    expect(data.ways.map((w) => w.tunnel), [false, true, false]);
    expect(data.ways.first.latLon, [52.0, -1.0, 52.1, -1.0]);
  });

  test('roads are downloaded once, cached, then served offline', () async {
    final dir = Directory.systemTemp.createTempSync('idr_roads_');
    final online = FakeOverpass(_road);
    final gnss = StreamController<GnssFix?>.broadcast();
    addTearDown(gnss.close);

    ProviderContainer make(FakeOverpass api) {
      final c = ProviderContainer(
        overrides: [
          roadCacheProvider.overrideWithValue(RoadCache(Future.value(dir))),
          overpassClientProvider.overrideWithValue(api),
          gnssProvider.overrideWith((ref) => gnss.stream),
        ],
      );
      c.listen(roadProvider, (_, _) {});
      c.listen(gnssProvider, (_, _) {});
      return c;
    }

    Future<void> settle() async {
      for (var i = 0; i < 20; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
    }

    final first = make(online);
    await settle();
    gnss.add(_fix());
    await settle();
    expect(first.read(roadProvider).status, RoadStatus.ready);
    expect(first.read(roadProvider).data!.ways.single.id, 1);
    expect(online.calls, 1);
    expect(await RoadCache(Future.value(dir)).count(), 1);
    first.dispose();

    // No network now: the cached cell still loads.
    final offline = FakeOverpass();
    final second = make(offline);
    await settle();
    gnss.add(_fix());
    await settle();
    expect(second.read(roadProvider).status, RoadStatus.ready);
    expect(second.read(roadProvider).data!.ways, isNotEmpty);
    expect(offline.calls, 0);
    second.dispose();
  });

  test(
    'without network or cache the controller reports offline, no crash',
    () async {
      final dir = Directory.systemTemp.createTempSync('idr_roads_');
      final gnss = StreamController<GnssFix?>.broadcast();
      addTearDown(gnss.close);
      final c = ProviderContainer(
        overrides: [
          roadCacheProvider.overrideWithValue(RoadCache(Future.value(dir))),
          overpassClientProvider.overrideWithValue(FakeOverpass()),
          gnssProvider.overrideWith((ref) => gnss.stream),
        ],
      );
      addTearDown(c.dispose);
      c.listen(roadProvider, (_, _) {});
      await Future<void>.delayed(const Duration(milliseconds: 10));
      gnss.add(_fix());
      for (var i = 0; i < 20; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
      expect(c.read(roadProvider).status, RoadStatus.offline);
      expect(c.read(roadProvider).data, isNull);
    },
  );
}
