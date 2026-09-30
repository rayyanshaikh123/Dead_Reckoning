import 'dart:convert';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path_provider/path_provider.dart';

/// A stretch of a drive without usable GNSS (real or tunnel test).
class Outage {
  const Outage({
    required this.start,
    required this.seconds,
    required this.distanceM,
    required this.simulated,
    this.exitErrorM,
    this.onRoad = false,
    this.handheld = false,
  });

  factory Outage.fromJson(Map<String, dynamic> j) => Outage(
    start: DateTime.parse(j['start'] as String),
    seconds: (j['seconds'] as num).toDouble(),
    distanceM: (j['distance_m'] as num).toDouble(),
    simulated: j['simulated'] as bool? ?? false,
    exitErrorM: (j['exit_error_m'] as num?)?.toDouble(),
    onRoad: j['on_road'] as bool? ?? false,
    handheld: j['handheld'] as bool? ?? false,
  );

  final DateTime start;
  final double seconds;

  /// Distance IDR estimated it travelled without GNSS.
  final double distanceM;
  final bool simulated;

  /// Distance between IDR's position and the first GNSS fix afterwards.
  final double? exitErrorM;

  /// Position was following the road map.
  final bool onRoad;

  /// The phone was not steadily mounted at some point during the outage.
  final bool handheld;

  double? get exitErrorPct => exitErrorM == null || distanceM < 1
      ? null
      : exitErrorM! / distanceM * 100;
  bool? get passesSih => exitErrorPct == null ? null : exitErrorPct! < 10;

  Map<String, dynamic> toJson() => {
    'start': start.toIso8601String(),
    'seconds': seconds,
    'distance_m': distanceM,
    'simulated': simulated,
    'exit_error_m': exitErrorM,
    'on_road': onRoad,
    'handheld': handheld,
  };
}

/// Downsampled position for the drive map; [gps] false = IDR estimate.
class TrackPoint {
  const TrackPoint(this.lat, this.lon, this.gps);

  final double lat;
  final double lon;
  final bool gps;
}

/// Everything shown about one recorded drive.
class DriveSummary {
  const DriveSummary({
    required this.id,
    required this.start,
    required this.end,
    required this.vehicleName,
    required this.distanceM,
    required this.maxSpeedMs,
    required this.outages,
    required this.track,
    this.inProgress = false,
    this.hasSensorLog = false,
    this.gyroRmseXKmh,
    this.gyroRmseYKmh,
  });

  factory DriveSummary.fromJson(Map<String, dynamic> j) {
    final t = [for (final v in j['track'] as List) (v as num).toDouble()];
    return DriveSummary(
      id: j['id'] as String,
      start: DateTime.parse(j['start'] as String),
      end: DateTime.parse(j['end'] as String),
      vehicleName: j['vehicle'] as String? ?? '',
      distanceM: (j['distance_m'] as num).toDouble(),
      maxSpeedMs: (j['max_speed_ms'] as num).toDouble(),
      outages: [
        for (final o in j['outages'] as List)
          Outage.fromJson(o as Map<String, dynamic>),
      ],
      track: [
        for (var i = 0; i + 2 < t.length; i += 3)
          TrackPoint(t[i], t[i + 1], t[i + 2] > 0),
      ],
      inProgress: j['in_progress'] as bool? ?? false,
      hasSensorLog: j['sensor_log'] as bool? ?? false,
      gyroRmseXKmh: (j['gyro_rmse_x_kmh'] as num?)?.toDouble(),
      gyroRmseYKmh: (j['gyro_rmse_y_kmh'] as num?)?.toDouble(),
    );
  }

  final String id;
  final DateTime start;
  final DateTime end;
  final String vehicleName;
  final double distanceM;
  final double maxSpeedMs;
  final List<Outage> outages;
  final List<TrackPoint> track;

  /// Still being written (or the app stopped mid-drive).
  final bool inProgress;
  final bool hasSensorLog;

  /// This drive's speed error vs GPS for each gyro mapping (km/h RMS).
  final double? gyroRmseXKmh;
  final double? gyroRmseYKmh;

  Duration get duration => end.difference(start);
  double get avgSpeedMs =>
      duration.inSeconds == 0 ? 0 : distanceM / duration.inSeconds;
  double get outageDistanceM => outages.fold(0, (s, o) => s + o.distanceM);

  Map<String, dynamic> toJson() => {
    'id': id,
    'start': start.toIso8601String(),
    'end': end.toIso8601String(),
    'vehicle': vehicleName,
    'distance_m': distanceM,
    'max_speed_ms': maxSpeedMs,
    'outages': [for (final o in outages) o.toJson()],
    'track': [
      for (final p in track) ...[
        double.parse(p.lat.toStringAsFixed(6)),
        double.parse(p.lon.toStringAsFixed(6)),
        if (p.gps) 1 else 0,
      ],
    ],
    'in_progress': inProgress,
    'sensor_log': hasSensorLog,
    'gyro_rmse_x_kmh': gyroRmseXKmh,
    'gyro_rmse_y_kmh': gyroRmseYKmh,
  };
}

/// Drives on the phone: `<id>.json` summary + optional `<id>.csv` sensor log.
class DriveStore {
  DriveStore(this._dir);

  final Future<Directory> _dir;

  static DriveStore forApp() => DriveStore(
    getApplicationSupportDirectory().then((d) async {
      final dir = Directory('${d.path}/drives');
      await dir.create(recursive: true);
      return dir;
    }),
  );

  Future<File> summaryFile(String id) async =>
      File('${(await _dir).path}/$id.json');
  Future<File> sensorLog(String id) async =>
      File('${(await _dir).path}/$id.csv');

  Future<void> save(DriveSummary d) async {
    final f = await summaryFile(d.id);
    // Write-then-rename so a crash never leaves a half-written summary.
    final tmp = File('${f.path}.tmp');
    await tmp.writeAsString(jsonEncode(d.toJson()));
    await tmp.rename(f.path);
  }

  /// All drives, newest first.
  Future<List<DriveSummary>> list() async {
    final out = <DriveSummary>[];
    await for (final e in (await _dir).list()) {
      if (e is! File || !e.path.endsWith('.json')) continue;
      try {
        out.add(
          DriveSummary.fromJson(
            jsonDecode(await e.readAsString()) as Map<String, dynamic>,
          ),
        );
      } catch (_) {
        // Skip unreadable files rather than failing the whole list.
      }
    }
    out.sort((a, b) => b.start.compareTo(a.start));
    return out;
  }

  Future<void> delete(String id) async {
    for (final f in [await summaryFile(id), await sensorLog(id)]) {
      if (await f.exists()) await f.delete();
    }
  }
}

final driveStoreProvider = Provider<DriveStore>((ref) => DriveStore.forApp());
