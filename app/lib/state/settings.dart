import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Loaded once in `main()` and injected with `overrideWithValue`.
final sharedPrefsProvider = Provider<SharedPreferences>(
  (ref) => throw UnimplementedError(
    'sharedPrefsProvider must be overridden in main()',
  ),
);

class AppSettings {
  const AppSettings({
    required this.onboarded,
    required this.vehicleName,
    required this.autoSensorNav,
    required this.autoRecord,
    required this.saveSensorLogs,
    required this.runInBackground,
  });

  /// True once the intro has been completed.
  final bool onboarded;
  final String vehicleName;

  /// Switch to sensor navigation automatically when GPS drops.
  final bool autoSensorNav;

  /// Start recording a drive automatically once the car is moving.
  final bool autoRecord;

  /// Keep the raw 10 Hz sensor log (CSV) with each drive.
  final bool saveSensorLogs;

  /// Keep GPS, sensors and the engine running with the screen off or while
  /// another app is in front (iOS background location / Android foreground
  /// service).
  final bool runInBackground;

  AppSettings copyWith({
    bool? onboarded,
    String? vehicleName,
    bool? autoSensorNav,
    bool? autoRecord,
    bool? saveSensorLogs,
    bool? runInBackground,
  }) => AppSettings(
    onboarded: onboarded ?? this.onboarded,
    vehicleName: vehicleName ?? this.vehicleName,
    autoSensorNav: autoSensorNav ?? this.autoSensorNav,
    autoRecord: autoRecord ?? this.autoRecord,
    saveSensorLogs: saveSensorLogs ?? this.saveSensorLogs,
    runInBackground: runInBackground ?? this.runInBackground,
  );
}

class AppSettingsController extends Notifier<AppSettings> {
  static const _kOnboarded = 'onboarded';
  static const _kVehicle = 'vehicle_name';
  static const _kAutoNav = 'auto_sensor_nav';
  static const _kAutoRecord = 'auto_record';
  static const _kSensorLogs = 'save_sensor_logs';
  static const _kBackground = 'run_in_background';
  static const defaultVehicleName = 'My car';

  SharedPreferences get _prefs => ref.read(sharedPrefsProvider);

  @override
  AppSettings build() {
    final p = ref.watch(sharedPrefsProvider);
    return AppSettings(
      onboarded: p.getBool(_kOnboarded) ?? false,
      vehicleName: p.getString(_kVehicle) ?? defaultVehicleName,
      autoSensorNav: p.getBool(_kAutoNav) ?? true,
      autoRecord: p.getBool(_kAutoRecord) ?? true,
      saveSensorLogs: p.getBool(_kSensorLogs) ?? true,
      runInBackground: p.getBool(_kBackground) ?? true,
    );
  }

  Future<void> completeOnboarding({required String vehicleName}) async {
    final name = vehicleName.trim().isEmpty
        ? defaultVehicleName
        : vehicleName.trim();
    await _prefs.setString(_kVehicle, name);
    await _prefs.setBool(_kOnboarded, true);
    state = state.copyWith(onboarded: true, vehicleName: name);
  }

  Future<void> resetOnboarding() async {
    await _prefs.setBool(_kOnboarded, false);
    state = state.copyWith(onboarded: false);
  }

  Future<void> setVehicleName(String name) async {
    final n = name.trim();
    if (n.isEmpty) return;
    await _prefs.setString(_kVehicle, n);
    state = state.copyWith(vehicleName: n);
  }

  Future<void> setAutoSensorNav(bool on) async {
    await _prefs.setBool(_kAutoNav, on);
    state = state.copyWith(autoSensorNav: on);
  }

  Future<void> setAutoRecord(bool on) async {
    await _prefs.setBool(_kAutoRecord, on);
    state = state.copyWith(autoRecord: on);
  }

  Future<void> setSaveSensorLogs(bool on) async {
    await _prefs.setBool(_kSensorLogs, on);
    state = state.copyWith(saveSensorLogs: on);
  }

  Future<void> setRunInBackground(bool on) async {
    await _prefs.setBool(_kBackground, on);
    state = state.copyWith(runInBackground: on);
  }
}

final appSettingsProvider =
    NotifierProvider<AppSettingsController, AppSettings>(
      AppSettingsController.new,
    );
