# Releasing IDR

## Before every release

```bash
cd app
dart format lib test && flutter analyze && flutter test   # same checks CI runs
```

- Bump `version:` in `pubspec.yaml`. The number after `+` must increase with every store upload.
- If the model was retrained, re-run the scripts in `tools/app_export/` (weights, goldens, replays, OSM roads). The tests will fail if the Dart engine and Python disagree.
- If the icon changed, run `tools/app_export/.venv/bin/python tools/app_export/make_icon.py`.

## iOS

| Goal | What you need | Command |
|---|---|---|
| Your own phone | free Apple ID team (apps expire after 7 days) | `flutter build ios --release && flutter install --release` |
| TestFlight / App Store | paid Apple Developer Program, and a unique bundle ID registered there (`com.sihidr.idrApp`) | `flutter build ipa`, then upload `build/ios/ipa/*.ipa` with Transporter or Xcode |

For App Store review:
- **Background location.** IDR declares `UIBackgroundModes: location` to keep navigating and recording with the screen off. Review will ask why; the answer is "turn-by-turn style navigation continuity through GNSS outages". Users can switch this off in Settings → Run in background.
- **Privacy labels.** Location, motion and drive logs stay on the device. Nothing is uploaded; data is only shared when the user taps *Share*. The only network calls are map tiles (OpenStreetMap) and road data (OpenStreetMap Overpass), which carry the map area but no user identity.

## Android

1. Create an upload key, once, and keep it safe:
   ```bash
   keytool -genkey -v -keystore ~/idr-upload.jks -keyalg RSA -keysize 2048 -validity 10000 -alias idr
   ```
2. Create `android/key.properties` (git-ignored):
   ```properties
   storeFile=/Users/<you>/idr-upload.jks
   storePassword=...
   keyAlias=idr
   keyPassword=...
   ```
3. Build: `flutter build appbundle` for Play Store, or `flutter build apk --release` for direct install.

Without `key.properties`, release builds are signed with the debug key. That's fine for testing, but Play won't accept it.

**Android hasn't been compiled yet.** There was no Android SDK on the development Mac. The first Android build should check:
- the sensor stream (Live → `[details]`: about 10 Hz, gravity ≈ 9.81);
- the background notification, which appears when location is on and "Run in background" is enabled;
- that the Android 13+ notification permission prompt appears.

## Attribution and fair use

- **Map tiles:** the standard OpenStreetMap tiles (`tile.openstreetmap.org`, no API key), darkened in the app. "© OpenStreetMap contributors" is shown on every map.
  - OSM's tile usage policy allows light use with an identifying User-Agent (set) and client-side caching (on). It does **not** cover an app with many users. Before a public launch, switch to a commercial tile provider that has a dark style, e.g. CARTO (now needs a key), Stadia Maps or MapTiler:
    ```bash
    flutter build ios --release --dart-define=MAP_TILES='https://…/{z}/{x}/{y}.png?api_key=YOUR_KEY'
    ```
    With `MAP_TILES` set, the in-app darkening filter is skipped, because the provider's dark style is used as-is.
- Road data: OpenStreetMap (ODbL), via the public Overpass API. The app fetches at most one ~2 km area at a time, caches it for 30 days, and waits at least a minute between retries.
