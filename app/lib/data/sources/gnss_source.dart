import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:geolocator/geolocator.dart';

import '../../state/settings.dart';

enum LocationAccess { unknown, granted, denied, deniedForever, serviceOff }

/// One GNSS fix, reduced to what the app uses.
class GnssFix {
  const GnssFix({
    required this.latitude,
    required this.longitude,
    required this.accuracyM,
    required this.speedMs,
    required this.headingDeg,
    required this.time,
  });

  factory GnssFix.fromPosition(Position p) => GnssFix(
    latitude: p.latitude,
    longitude: p.longitude,
    accuracyM: p.accuracy,
    // iOS reports -1 when speed/course are invalid.
    speedMs: p.speed >= 0 ? p.speed : null,
    headingDeg: p.heading >= 0 ? p.heading : null,
    time: p.timestamp,
  );

  final double latitude;
  final double longitude;
  final double accuracyM;
  final double? speedMs;
  final double? headingDeg;
  final DateTime time;
}

class LocationAccessController extends Notifier<LocationAccess> {
  @override
  LocationAccess build() => LocationAccess.unknown;

  /// Reads the current permission without prompting.
  Future<LocationAccess> refresh() async {
    state = await _read(prompt: false);
    return state;
  }

  /// Shows the system prompt if the user hasn't answered yet.
  Future<LocationAccess> request() async {
    state = await _read(prompt: true);
    return state;
  }

  Future<void> openSettings() => Geolocator.openAppSettings();

  Future<LocationAccess> _read({required bool prompt}) async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) {
        return LocationAccess.serviceOff;
      }
      var p = await Geolocator.checkPermission();
      if (prompt && p == LocationPermission.denied) {
        p = await Geolocator.requestPermission();
      }
      return switch (p) {
        LocationPermission.always ||
        LocationPermission.whileInUse => LocationAccess.granted,
        LocationPermission.deniedForever => LocationAccess.deniedForever,
        _ => LocationAccess.denied,
      };
    } catch (e) {
      debugPrint('location permission check failed: $e');
      return LocationAccess.denied;
    }
  }
}

final locationAccessProvider =
    NotifierProvider<LocationAccessController, LocationAccess>(
      LocationAccessController.new,
    );

/// Location settings. With [background] the stream keeps the whole app —
/// motion sensors, engine and drive recorder — alive with the screen off:
/// on iOS via the `location` background mode (Info.plist) with the blue
/// status-bar indicator, on Android via a foreground service notification.
LocationSettings _settings({required bool background}) {
  if (defaultTargetPlatform == TargetPlatform.iOS) {
    return AppleSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      activityType: ActivityType.automotiveNavigation,
      distanceFilter: 0,
      pauseLocationUpdatesAutomatically: false,
      allowBackgroundLocationUpdates: background,
      showBackgroundLocationIndicator: background,
    );
  }
  if (defaultTargetPlatform == TargetPlatform.android) {
    return AndroidSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      distanceFilter: 0,
      intervalDuration: const Duration(seconds: 1),
      foregroundNotificationConfig: background
          ? const ForegroundNotificationConfig(
              notificationTitle: 'IDR is running',
              notificationText: 'Tracking your drive, ready for GPS outages',
              enableWakeLock: true,
              setOngoing: true,
            )
          : null,
    );
  }
  return const LocationSettings(accuracy: LocationAccuracy.best);
}

/// Live GNSS fixes; emits null until permission is granted.
final gnssProvider = StreamProvider<GnssFix?>((ref) {
  final access = ref.watch(locationAccessProvider);
  final background = ref.watch(
    appSettingsProvider.select((s) => s.runInBackground),
  );
  if (access != LocationAccess.granted) return Stream.value(null);
  return Geolocator.getPositionStream(
    locationSettings: _settings(background: background),
  ).map(GnssFix.fromPosition);
});
