import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path_provider/path_provider.dart';

import '../../engine/road_network.dart';
import '../sources/gnss_source.dart';
import 'overpass_client.dart';

/// Map area key: a 0.02° × 0.02° cell (≈ 2 × 1.4 km at UK latitudes).
typedef RoadCell = (int, int);

RoadCell roadCellOf(double lat, double lon) =>
    ((lat / RoadCache.cellDeg).floor(), (lon / RoadCache.cellDeg).floor());

/// Road data stored on the phone, one JSON file per cell, so tunnels you've
/// driven near before work fully offline.
class RoadCache {
  RoadCache(this._dir);

  static const cellDeg = 0.02;

  /// Downloads cover the cell plus this margin, so the road ahead is known
  /// well past the cell edge.
  static const marginDeg = 0.01;

  /// Cached roads older than this are refreshed when online.
  static const maxAge = Duration(days: 30);

  final Future<Directory> _dir;

  static RoadCache forApp() => RoadCache(
    getApplicationSupportDirectory().then((d) async {
      final dir = Directory('${d.path}/roads');
      await dir.create(recursive: true);
      return dir;
    }),
  );

  Future<File> _file(RoadCell c) async =>
      File('${(await _dir).path}/${c.$1}_${c.$2}.json');

  Future<({RoadData data, bool stale})?> read(RoadCell c) async {
    final f = await _file(c);
    if (!await f.exists()) return null;
    final age = DateTime.now().difference(await f.lastModified());
    final data = RoadData.fromJson(
      jsonDecode(await f.readAsString()) as Map<String, dynamic>,
    );
    return (data: data, stale: age > maxAge);
  }

  Future<void> write(RoadCell c, RoadData data) async =>
      (await _file(c)).writeAsString(jsonEncode(data.toJson()));

  Future<int> count() async =>
      (await _dir).list().where((e) => e.path.endsWith('.json')).length;

  Future<int> sizeBytes() async {
    var total = 0;
    await for (final e in (await _dir).list()) {
      if (e is File) total += await e.length();
    }
    return total;
  }

  Future<void> clear() async {
    await for (final e in (await _dir).list()) {
      if (e is File) await e.delete();
    }
  }
}

enum RoadStatus { idle, downloading, ready, offline }

class RoadState {
  const RoadState({
    this.status = RoadStatus.idle,
    this.data,
    this.cellsLoaded = 0,
    this.error,
  });

  final RoadStatus status;

  /// Roads for the cells around the vehicle (merged).
  final RoadData? data;
  final int cellsLoaded;
  final String? error;
}

final roadCacheProvider = Provider<RoadCache>((ref) => RoadCache.forApp());
final overpassClientProvider = Provider<OverpassClient>(
  (ref) => OverpassClient(),
);

/// Keeps road data loaded for where the vehicle is: from the phone's cache
/// when possible, otherwise downloaded while there's GPS (and network).
class RoadController extends Notifier<RoadState> {
  static const _retryAfter = Duration(minutes: 1);
  static const _keepCells = 6;

  final _loaded = <RoadCell, RoadData>{};
  final _order = <RoadCell>[];
  RoadCell? _current;
  bool _busy = false;
  DateTime? _failedAt;

  @override
  RoadState build() {
    ref.listen(gnssProvider, (_, next) {
      final f = next.value;
      if (f != null && f.accuracyM <= 50) {
        _ensure(roadCellOf(f.latitude, f.longitude));
      }
    });
    return const RoadState();
  }

  Future<void> _ensure(RoadCell cell) async {
    if (_busy || cell == _current && _loaded.containsKey(cell)) return;
    _current = cell;
    if (_loaded.containsKey(cell)) return;
    _busy = true;
    try {
      final cache = ref.read(roadCacheProvider);
      final cached = await cache.read(cell);
      if (cached != null && !cached.stale) {
        _add(cell, cached.data, RoadStatus.ready);
        return;
      }
      if (_failedAt != null &&
          DateTime.now().difference(_failedAt!) < _retryAfter) {
        if (cached != null) _add(cell, cached.data, RoadStatus.offline);
        return;
      }
      state = RoadState(
        status: RoadStatus.downloading,
        data: state.data,
        cellsLoaded: _loaded.length,
      );
      try {
        final (lat0, lon0) = (
          cell.$1 * RoadCache.cellDeg,
          cell.$2 * RoadCache.cellDeg,
        );
        final data = await ref
            .read(overpassClientProvider)
            .fetch(
              south: lat0 - RoadCache.marginDeg,
              west: lon0 - RoadCache.marginDeg,
              north: lat0 + RoadCache.cellDeg + RoadCache.marginDeg,
              east: lon0 + RoadCache.cellDeg + RoadCache.marginDeg,
            );
        await cache.write(cell, data);
        _failedAt = null;
        _add(cell, data, RoadStatus.ready);
      } catch (e) {
        debugPrint('$e');
        _failedAt = DateTime.now();
        if (cached != null) {
          _add(cell, cached.data, RoadStatus.offline); // stale but usable
        } else {
          state = RoadState(
            status: RoadStatus.offline,
            data: state.data,
            cellsLoaded: _loaded.length,
            error: '$e',
          );
        }
      }
    } catch (e) {
      // Storage problems must never take the engine down.
      debugPrint('road data unavailable: $e');
      state = RoadState(
        status: RoadStatus.offline,
        data: state.data,
        cellsLoaded: _loaded.length,
        error: '$e',
      );
    } finally {
      _busy = false;
    }
  }

  void _add(RoadCell cell, RoadData data, RoadStatus status) {
    _loaded[cell] = data;
    _order
      ..remove(cell)
      ..add(cell);
    while (_order.length > _keepCells) {
      _loaded.remove(_order.removeAt(0));
    }
    var merged = const RoadData([]);
    for (final c in _order) {
      merged = merged.merge(_loaded[c]!);
    }
    state = RoadState(
      status: status,
      data: merged,
      cellsLoaded: _loaded.length,
    );
  }

  /// Deletes all cached road data (Settings).
  Future<void> clearCache() async {
    await ref.read(roadCacheProvider).clear();
    _loaded.clear();
    _order.clear();
    _current = null;
    state = const RoadState();
  }
}

final roadProvider = NotifierProvider<RoadController, RoadState>(
  RoadController.new,
);

/// Number of cached areas and their size, for Settings.
final roadCacheStatsProvider = FutureProvider<({int areas, int bytes})>((
  ref,
) async {
  ref.watch(roadProvider);
  final cache = ref.read(roadCacheProvider);
  return (areas: await cache.count(), bytes: await cache.sizeBytes());
});
