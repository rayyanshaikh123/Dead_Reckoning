# IDR — Flutter app

On-device Intelligent Dead Reckoning. The app runs the IDNN v5 speed model and its physics filters live at 10 Hz from phone sensors, and keeps estimating position through GNSS outages. Android and iOS; everything runs on the phone.

```bash
cd app
flutter pub get
flutter run --release    # Android or iOS (keeps app data; `flutter install` wipes it)
flutter test             # engine parity, benchmark, simulation, widgets
```

Build an Android APK (needs the Android SDK, platform 36, and JDK 17):

```bash
flutter build apk --release
# → build/app/outputs/flutter-apk/app-release.apk
```

Release signing is described in [RELEASE.md](RELEASE.md). Without `android/key.properties`, release builds are signed with the debug key — fine for sideloading, not for the Play Store.

## What's in the app

| Screen | What it does |
|---|---|
| **Home** | Speed, heading, GPS quality, calibration state. Tiles for **Tunnel test** (hide GPS on a real drive) and **Simulate**. |
| **Live** | The engine's view: model speed vs GPS, pipeline stages (gate, EKF, cruise lock, ZUPT), raw sensors. |
| **Map** | Your position: blue while GPS is used, orange while IDR is estimating. |
| **Simulation** | A real recorded drive on the map; you open and close tunnels yourself (see below). |
| **Replay** | The five benchmark tunnels from the README, each with its scorecard. |
| **History** | Every drive, recorded automatically, with each outage scored. |
| **Settings** | Calibration, background running, offline roads, AI model details. |

## Layout

```
lib/
  app/          router (4-tab shell + pushed routes) and MaterialApp
  core/theme    design tokens (colours, radii, spacing) and the Sometype Mono type scale
  core/widgets  design-system kit: tiles, toggle, dot indicator, dot-matrix chart,
                vertical gauge, line-art car, location puck, bottom nav, map
  features/     dashboard, live, map, simulation, replay, history, profile, onboarding, gallery
  engine/       pure-Dart IDNN v5 dead-reckoning engine (see below)
  state/        Riverpod providers: settings, live engine, nav view, drive recorder
  data/         sensors, GPS, model loader, OSM roads, drives, replays and simulation
tool/
  evaluate.dart full-drive accuracy evaluation (see "Accuracy")
```

The **widget gallery** (`[menu]` → Widget gallery, debug builds) shows every component.

## Engine (`lib/engine/`)

This is a pure-Dart port of the IDNN v5 pipeline with no Flutter imports. It runs:

sensor frame → 18 features → IDNN v5 (378→256→128→64→1) → kinematic gate → feed-forward EKF → cruise lock → ZUPT → GNSS calibrator (during outages) → heading and position (dead reckoning or road following).

- **Entry point:** `DrPipeline.push(frame, gnss:)`. Pass `gnss: null` and it is in an outage; pass a fix again and it reports the exit error. Output lags input by 0.5 s: 0.1 s for the central-difference jerk and 0.4 s for ZUPT's centred window. This keeps the maths identical to the benchmark.
- **Parity:** `test/engine/` checks every stage against Python goldens (see `tools/app_export/`).
- **Speed:** one inference takes about 0.14 ms on a laptop VM; the Settings screen shows the time measured on the phone.
- **Handheld phones:** `MountStability` watches tilt wobble. A phone held in the hand (wobble over 4°) pauses calibration, so hand motion can't corrupt it.

## Road matching (`lib/engine/road_network.dart`, `lib/data/roads/`)

While GPS is healthy, the app downloads OpenStreetMap roads for the 2 km around you (Overpass API) and saves them on the phone, so areas you've driven before work offline.

When GPS drops, IDR snaps to the road you're on and follows the road graph. At junctions it doesn't commit to one branch: it keeps up to 40 hypotheses alive, each combining a route choice with a distance scale (±15 %, to absorb the model's speed error) and a gyro heading drift. Each is scored by how well the road's direction matches the gyro heading, with small penalties for changing road or taking a minor road. Unlikely hypotheses are pruned; the best one gives the position.

Replay results on IO-VNBD Drive M (`test/engine/benchmark_test.dart`):

| Tunnel | Benchmark protocol (snap to true track) | OpenStreetMap roads (what the app does live) |
|---|---|---|
| Highway, 648 m | 14.4 m (2.2 %) | **12.3 m (1.9 %)** |
| Underpass, 219 m | 6.5 m (3.0 %) | **4.9 m (2.2 %)** |
| Medium, 448 m | 41.8 m (9.3 %) | **14.3 m (3.2 %)** |
| Mountain, 1680 m | 85.8 m (5.1 %) | **113.7 m (6.8 %)** |
| City canyon, 600 m | 83.2 m (13.9 %) | **36.2 m (6.0 %)** |

All five pass the SIH target (under 10 %) on OSM roads.

## Test simulation (`lib/features/simulation/`, `lib/data/replay/simulation.dart`)

Home → **Simulate** (or the menu → Simulation). A real recorded drive plays on the map, and you drive into and out of "tunnels" whenever you like:

- **Enter tunnel / Exit tunnel** cuts and restores GPS, as often as you want. The banner at the top switches between *Open road · GPS* and *In tunnel · 00:39 · no GPS*.
- **On the map:** the car is blue on GPS and orange while IDR estimates; the real car is a hollow ghost. Each tunnel is a dark band with its entrance and exit, IDR's path is orange, and a tag such as `#1 · 4 m` shows how far off IDR was when GPS came back (green under 10 %, orange over).
- **Metrics:** IDR speed vs true speed, heading vs true heading, a one-minute speed chart, the active pipeline stages, and per tunnel: time without GPS, distance driven vs IDR's measurement, the error in metres and as a share of the distance, the largest gap along the way, and pass/fail against the SIH 10 %. A log keeps every tunnel.
- **Choices:** five roads (the default, *Tight city loops*, has 16 turns in 2.5 km), playback at 1×–8×, and what keeps IDR on the road: the OSM map (the live app's method), a known route (as with turn-by-turn navigation), or nothing (pure speed × heading).

The drives are real IO-VNBD recordings, so the AI sees real sensor data, and the recording's own GPS is the truth. Invented sensor data would say nothing about the model.

**Expect a mix of results.** Over 40 s tunnels started every 20 s along each route, about half finish under 10 %. *Tight city loops* on the OSM map does best (8 of 14, median 7.5 %). Stretches where the model misjudges speed, or where the map takes the wrong branch, show up in orange; they aren't hidden.

## Drive recording (`lib/state/drive_recorder.dart`, `lib/data/drives/`)

- **Automatic.** A drive starts after 5 s above 4 m/s (GPS) and ends after 3 min stationary. Recordings under 300 m are discarded.
- **Per drive**, stored on the phone:
  - `<id>.json`: summary, downsampled track (GPS vs IDR estimate), and every outage with its measured exit error. It is saved every 30 s, so a killed app keeps the drive.
  - `<id>.csv` (optional): the raw 10 Hz sensor log, with IO-VNBD-style column names so it can feed retraining. Gyro columns are the physical X/Y/Z. The IDR speed columns lag the sensors by 0.5 s.
- **Gyro-mapping check.** A shadow engine runs the alternative gyro mapping (`GyroMapping.yawIsY`). Both are scored against GPS speed while driving, per drive (drive report) and cumulatively (Settings → AI model details). Whichever is consistently lower should become the default.

## Accuracy on the whole drive (`tool/evaluate.dart`)

The five README tunnels turn out to be favourable picks. Across **145 random GPS outages** on the full 105 km Drive M (30–120 s each; `dart run tool/evaluate.dart`, reports in `data/eval/`):

| Method | Median exit error | 90th percentile | SIH pass rate (< 10 %) |
|---|---|---|---|
| IDR on OSM roads (live app) | 90 m (16.6 %) | 74.5 % | **31 %** |
| Hold last GPS speed (no IDR) | 362 m (69.2 %) | 139.5 % | 3 % |

`--fixed` (the five README tunnels inside the same run) reproduces the benchmark: 100 % pass. So the evaluator is sound and the gap is real.

- **IDR is about 4× better than doing nothing, but misses the SIH target on most random outages.** The multi-hypothesis road matcher raised the pass rate from 17 % to 31 %.
- **Main limit: model speed accuracy.** Speed error is 15.7 km/h RMSE (Python gives the same). Over 105 km the errors cancel (total distance +0.3 %), but over a 30–120 s outage they don't: the median distance drift is 15.7 %.
  - Retraining is the lever. Use the logger's "Pitch" (true yaw rate), which training never saw. Fix the ~1.3 s phone/vehicle offset. Add more drives, e.g. the app's own sensor logs.
- **Second limit: junctions.** 30 % of OSM outages still end on the wrong road.

## Status

All planned phases (0–7) are done:
- design system, splash/onboarding, live GPS
- the pure-Dart engine, verified against Python, with the Drive M benchmark reproduced
- Replay, and the interactive test Simulation
- live sensors with mount alignment and handheld detection
- multi-hypothesis OSM road matching with an offline cache
- automatic drive recording, History and reports
- background running, error banners and error log, app icon, CI (`.github/workflows/app.yml`), and release signing setup (see [RELEASE.md](RELEASE.md))

**Not yet verified:** accuracy on real drives with this app, and the Android build on a real device. The first drives will also show which gyro mapping to keep (Settings → AI model → details).
