import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/sources/gnss_source.dart';
import 'live_engine.dart';
import 'settings.dart';

/// Where the position/speed shown to the user currently comes from.
enum NavMode {
  /// No usable GPS fix yet (or location is off).
  waiting,

  /// GPS fix, vehicle stationary.
  parked,

  /// GPS fix, moving.
  gps,

  /// No GPS — estimating from motion sensors (real outage or tunnel test).
  sensors,
}

/// A GPS fix older than this is treated as lost.
const _staleAfter = Duration(seconds: 5);

/// Ticks once a second so time-based values (fix age, tunnel timer) refresh.
final clockProvider = StreamProvider<DateTime>(
  (ref) => Stream.periodic(const Duration(seconds: 1), (_) => DateTime.now()),
);

/// "Tunnel test": hides GPS from the live engine so the user can watch
/// sensor-only navigation. The speed/heading captured here are only a
/// fallback for when the engine isn't running (e.g. no motion sensors).
class TunnelTest {
  const TunnelTest({this.since, this.speedMs = 0, this.headingDeg});

  final DateTime? since;
  final double speedMs;
  final double? headingDeg;

  bool get active => since != null;
}

class TunnelTestController extends Notifier<TunnelTest> {
  @override
  TunnelTest build() => const TunnelTest();

  void toggle() {
    if (state.active) {
      state = const TunnelTest();
      return;
    }
    final fix = ref.read(gnssProvider).value;
    state = TunnelTest(
      since: DateTime.now(),
      speedMs: fix?.speedMs ?? 0,
      headingDeg: fix?.headingDeg,
    );
  }
}

final tunnelTestProvider = NotifierProvider<TunnelTestController, TunnelTest>(
  TunnelTestController.new,
);

/// Everything the screens show, derived from GPS, the tunnel test and settings.
class NavView {
  const NavView({
    required this.mode,
    required this.vehicleName,
    required this.access,
    this.speedKmh,
    this.headingDeg,
    this.accuracyM,
    this.fix,
    this.tunnelSeconds = 0,
    this.tunnelDistanceM = 0,
    this.simulated = false,
    this.onRoad = false,
    this.mounted = true,
    this.estimateLat,
    this.estimateLon,
  });

  final NavMode mode;
  final String vehicleName;
  final LocationAccess access;
  final double? speedKmh;
  final double? headingDeg;
  final double? accuracyM;
  final GnssFix? fix;
  final int tunnelSeconds;
  final double tunnelDistanceM;

  /// Sensor mode was forced by the tunnel test rather than a real outage.
  final bool simulated;

  /// In sensor mode: following the OpenStreetMap road graph.
  final bool onRoad;

  /// Phone held steady by a mount. IDR's GPS-free estimate needs this.
  final bool mounted;

  /// Dead-reckoned position while in sensor mode (null until known).
  final double? estimateLat;
  final double? estimateLon;

  bool get inTunnel => mode == NavMode.sensors;
  bool get locationOff =>
      access != LocationAccess.granted && access != LocationAccess.unknown;

  String get statusLabel => switch (mode) {
    NavMode.waiting => locationOff ? 'Location is off' : 'Searching for GPS',
    NavMode.parked => 'Parked',
    NavMode.gps => 'Driving · GPS',
    NavMode.sensors => 'No GPS · sensors',
  };

  /// 0–5 dots from horizontal accuracy.
  int get gpsDots {
    final a = accuracyM;
    if (a == null) return 0;
    if (a <= 5) return 5;
    if (a <= 10) return 4;
    if (a <= 20) return 3;
    if (a <= 50) return 2;
    return 1;
  }

  /// 0–1 position confidence for the gauge.
  double get confidence => switch (mode) {
    NavMode.waiting => 0,
    NavMode.parked ||
    NavMode.gps => (1 - (accuracyM ?? 60) / 60).clamp(0.05, 1.0),
    NavMode.sensors =>
      (0.95 * (1 - tunnelSeconds / 300)).clamp(0.2, 0.95) * (mounted ? 1 : 0.3),
  };

  String get speedLabel => speedKmh == null ? '--' : '${speedKmh!.round()}';

  String get headingLabel {
    final h = headingDeg;
    if (h == null) return '—';
    const names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    return '${h.round()}° ${names[((h % 360) / 45).round() % 8]}';
  }

  String get tunnelClock {
    final t = tunnelSeconds;
    return '${(t ~/ 60).toString().padLeft(2, '0')}:${(t % 60).toString().padLeft(2, '0')}';
  }
}

final navViewProvider = Provider<NavView>((ref) {
  final now = ref.watch(clockProvider).value ?? DateTime.now();
  final fix = ref.watch(gnssProvider).value;
  final tunnel = ref.watch(tunnelTestProvider);
  final settings = ref.watch(appSettingsProvider);
  final access = ref.watch(locationAccessProvider);
  final engine = ref.watch(liveEngineProvider);
  final out = engine.running ? engine.output : null;

  // A real outage only counts once GNSS has been healthy this session.
  final realOutage =
      !tunnel.active &&
      settings.autoSensorNav &&
      engine.hadGnss &&
      (out?.blackout ?? false);

  if (tunnel.active || realOutage) {
    final engineBlackout = out != null && out.blackout;
    final t = engineBlackout
        ? out.blackoutSamples ~/ 10
        : now.difference(tunnel.since ?? now).inSeconds.clamp(0, 1 << 30);
    return NavView(
      mode: NavMode.sensors,
      vehicleName: settings.vehicleName,
      access: access,
      speedKmh: out != null ? out.speed * 3.6 : tunnel.speedMs * 3.6,
      headingDeg: out?.headingDeg ?? tunnel.headingDeg,
      tunnelSeconds: t,
      tunnelDistanceM: engineBlackout
          ? out.distanceSinceEntry
          : tunnel.speedMs * t,
      simulated: tunnel.active,
      onRoad: out?.onRoad ?? false,
      mounted: engine.mounted,
      estimateLat: engine.estimateLat,
      estimateLon: engine.estimateLon,
      fix: fix,
    );
  }

  final fresh = fix != null && now.difference(fix.time) <= _staleAfter;
  if (!fresh) {
    return NavView(
      mode: NavMode.waiting,
      vehicleName: settings.vehicleName,
      access: access,
      fix: fix,
    );
  }

  final speed = fix.speedMs ?? 0;
  final moving = speed >= 0.5;
  return NavView(
    mode: moving ? NavMode.gps : NavMode.parked,
    vehicleName: settings.vehicleName,
    access: access,
    speedKmh: speed * 3.6,
    // Course over ground is meaningless when stationary.
    headingDeg: moving ? fix.headingDeg : null,
    accuracyM: fix.accuracyM,
    fix: fix,
    mounted: engine.mounted,
  );
});

/// One second of history: the speed shown to the user and the AI's own
/// (filtered) estimate, both km/h.
typedef SpeedSample = ({double? shown, double? ai});

/// Last 61 one-second samples, oldest first.
class SpeedHistoryController extends Notifier<List<SpeedSample>> {
  static const length = 61;

  @override
  List<SpeedSample> build() {
    ref.listen(clockProvider, (_, _) {
      final out = ref.read(liveEngineProvider).output;
      final sample = (
        shown: ref.read(navViewProvider).speedKmh,
        ai: out == null ? null : out.filteredSpeed * 3.6,
      );
      final next = [...state, sample];
      state = next.length > length ? next.sublist(next.length - length) : next;
    });
    return const [];
  }
}

final speedHistoryProvider =
    NotifierProvider<SpeedHistoryController, List<SpeedSample>>(
      SpeedHistoryController.new,
    );
