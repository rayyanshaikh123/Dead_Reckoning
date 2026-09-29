# IDR — Flutter app

On-device Intelligent Dead Reckoning: runs the IDNN v5 speed model and its physics filters live at 10 Hz from phone sensors, and keeps estimating position through GNSS outages.

```bash
cd app
flutter pub get
flutter run          # Android or iOS
flutter test
```

## Layout

```
lib/
  app/        router (4-tab shell + pushed routes) and MaterialApp
  core/theme  design tokens (colours, radii, spacing) and Sometype Mono type scale
  core/widgets  design-system kit: tiles, toggle, dot indicator, dot-matrix chart,
                vertical gauge, line-art car, bottom nav, placeholder map
  features/   dashboard, live, map, profile, history, replay, gallery
  engine/     pure-Dart IDNN v5 dead-reckoning engine (see below)
  state/      Riverpod providers: settings, nav view, speed history
  data/       GPS source, model loader, benchmark numbers
```

The **widget gallery** (`[menu]` → Widget gallery) shows every component.

## Engine (`lib/engine/`)

This is a pure-Dart port of the IDNN v5 pipeline with no Flutter imports. It runs:

sensor frame → 18 features → IDNN v5 (378→256→128→64→1) → kinematic gate → feed-forward EKF → cruise lock → ZUPT → GNSS calibrator (during outages) → heading and position (dead reckoning or road snapping).

- **Entry point:** `DrPipeline.push(frame, gnss:)`. Output lags input by 0.5 s: 0.1 s for the central-difference jerk and 0.4 s for ZUPT's centred window. This keeps the maths identical to the benchmark.
- **Parity:** `test/engine/` checks every stage against Python goldens (see `tools/app_export/`).
- **Speed:** one inference takes about 0.14 ms on a laptop VM. The Settings screen shows the time measured on the phone.

## Road matching (`lib/engine/road_network.dart`, `lib/data/roads/`)

While GPS is healthy, the app downloads OpenStreetMap roads for the 2 km area around you (Overpass API) and saves them on the phone, so areas you've driven before work offline.

When GPS drops, IDR snaps to the road you're on and follows the road graph. At junctions it takes the branch that best matches the gyro heading, with a preference for staying on the same road. If the heading swings to another branch within 60 m, it switches.

Replay results on IO-VNBD Drive M (`test/engine/benchmark_test.dart`):

| Tunnel | Benchmark protocol (snap to true track) | OpenStreetMap roads (what the app does live) |
|---|---|---|
| Highway, 648 m | 14.4 m (2.2%) | **12.3 m (1.9%)** |
| Underpass, 219 m | 6.5 m (3.0%) | **4.9 m (2.2%)** |
| Medium, 448 m | 41.8 m (9.3%) | **38.9 m (8.7%)** |
| Mountain, 1680 m | 85.8 m (5.1%) | **49.4 m (2.9%)** |
| City canyon, 600 m | 83.2 m (13.9%) | **50.2 m (8.4%)** |

## Drive recording (`lib/state/drive_recorder.dart`, `lib/data/drives/`)

- **Automatic.** A drive starts after 5 s above 4 m/s (GPS) and ends after 3 min stationary. Recordings under 300 m are discarded.
- **Per drive**, stored on the phone:
  - `<id>.json`: summary, downsampled track (GPS vs IDR estimate), and every outage with its measured exit error. It is saved every 30 s, so a killed app keeps the drive.
  - `<id>.csv` (optional): the raw 10 Hz sensor log, with IO-VNBD-style column names so it can feed retraining. Gyro columns are the physical X/Y/Z. The IDR speed columns lag the sensors by 0.5 s.
- **Gyro-mapping check.** A shadow engine runs the alternative gyro mapping (`GyroMapping.yawIsY`). Both are scored against GPS speed while driving, per drive (drive report) and cumulatively (Settings → AI model details). Whichever is consistently lower should become the default.

## Status

All planned phases (0–7) are done:
- design system, splash/onboarding, live GPS
- the pure-Dart engine, verified against Python, with the Drive M benchmark reproduced
- the Replay player
- live sensors with mount alignment
- OSM road matching with an offline cache
- automatic drive recording, History and reports
- background running, error banners and error log, app icon, CI (`.github/workflows/app.yml`), and release signing setup (see [RELEASE.md](RELEASE.md))

**Not yet verified:** the Android build (no Android SDK on the dev machine), and accuracy on real drives. The first drives will also show which gyro mapping to keep (Settings → AI model → details).
