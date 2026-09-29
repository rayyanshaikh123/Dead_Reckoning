# Canonical sensor frame: what IDNN v5 actually saw in training

These findings come from analysing IO-VNBD Drive M (`S-M.csv` / `V-M.csv`) on 2026-09-29. Phase 4 (live sensors) must feed the model inputs that match them.

## 1. Gyro inputs are `[Yaw, Yaw, Roll]`, not `[Yaw, Pitch, Roll]`

`detect_smartphone_columns` ([src/baseline_graphs.py](../../src/baseline_graphs.py)) finds columns with `find_column`, which checks each column against every pattern. For `gy`, the pattern `gyroscope.*y` matches the "y" in **"GYROSCOPE Yaw"** before it can reach "Pitch". The result is:

| feature | column used |
|---|---|
| `gyro_x` (feature 6) | `GYROSCOPE Yaw (rad/s)` |
| `gyro_y` (feature 7) | `GYROSCOPE Yaw (rad/s)` (duplicate) |
| `gyro_z` (feature 8) | `GYROSCOPE Roll (rad/s)` |

The same holds for `jerk_gx/gy/gz`. **The Pitch column never reached the model**, and neither did any other training drive's Pitch column, because the logger's column names are the same everywhere. The model is still valid, since it learned from these inputs, but the app has to reproduce the mapping. The replay files do this automatically because `export_replay.py` uses the same mapping.

## 2. In this logger, "Pitch" is the car's yaw rate

The phone lay flat, screen up, for the whole drive:

- gravity ≈ (0.00, 0.00, 9.807) m/s², with 1st–99th percentile of X and Y within ±0.05
- mean accelerometer Z is 9.85 m/s²

Correlation with the vehicle's own `Yaw Rate` sensor, using speed > 3 m/s and a 1.3 s lag:

| phone column | corr |
|---|---|
| GYROSCOPE **Pitch** | **+0.86** |
| GYROSCOPE Yaw | +0.07 |
| GYROSCOPE Roll | −0.13 |

So, for this flat phone, the column the logger calls "Pitch" is rotation about the vertical axis. Combined with finding 1, the model has **no direct yaw-rate input**. Its gyro features carry the two horizontal rotation rates (road pitch and roll, i.e. suspension motion). That fits a model that estimates *speed* from vibration and dynamics rather than from turning.

## 3. Phone and vehicle logs are offset by about 1.3 s

Although the dataset is labelled "synchronised", the best phone-to-vehicle correlation occurs at a 13–26 sample lag, and the lag drifts over the 2.9 h drive. Training used zero lag, so the model learned to predict speed about 1 s "late" relative to the phone signals. This limits accuracy, but the app doesn't need to act on it.

## 4. Mount angle of the training phone

Cornering gives an unambiguous direction on the phone's own clock: centripetal acceleration = GPS speed × "Pitch" rate, and it points to the car's left. The best fit (correlation 0.65) puts the car's **left** at 61.5° from the phone's +x axis, so **forward is at 331.5°**. Forward acceleration then correlates positively with changes in GPS speed, which confirms the sign.

**The training phone's +x axis pointed 28.5° to the left of the car's nose.**

## 5. How the app feeds live sensors (Phase 4, `app/lib/engine/frame_adapter.dart`)

1. **Canonical frame.** The native code delivers Android conventions. iOS CoreMotion values are negated and multiplied by 9.80665; the gyro is unchanged.
2. **Flatten.** Rotate each frame so gravity points along +z (`rotationToUp`).
3. **Learn the mount.** `MountAlignment` finds the car's forward direction in the flattened frame from cornering: the correlation of speed × yaw rate with horizontal acceleration. It only uses samples above 4 m/s. Evidence has a 5-minute half-life, and the result is saved per install. Until it's learned, the default assumes a portrait phone in a dash cradle, screen facing the driver, which puts forward at +y.
4. **Training axes.** Express the frame in axes where x sits 28.5° left of forward (from section 4).
5. **Gyro columns.** Rebuild them as `[Yaw, Yaw, Roll]` = `[ω_x, ω_x, ω_y]` (`GyroMapping.yawIsX`). `yawIsY` is the alternative. **Still to confirm:** record drives and keep whichever mapping gives the lower speed error against GPS.
6. **Heading.** Heading keeps using the untouched physical frame (full gyro vector projected onto gravity).
