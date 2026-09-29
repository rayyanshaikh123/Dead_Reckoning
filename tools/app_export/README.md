# app_export

Bridges the Python IDNN v5 pipeline to the Flutter app's pure-Dart engine (`app/lib/engine/`).

```bash
# one-time setup (from repo root)
python3 -m venv tools/app_export/.venv
tools/app_export/.venv/bin/pip install torch numpy "pandas<3" scipy pyproj matplotlib   # src/ needs pandas 2 (writable arrays)

# export weights -> app/assets/models/idnn_v5.{bin,json}
tools/app_export/.venv/bin/python tools/app_export/export_model.py

# regenerate parity goldens -> app/test/golden/v5_pipeline.json
tools/app_export/.venv/bin/python tools/app_export/make_goldens.py

# cut the 5 README tunnels out of Drive M -> app/assets/replays/ (needs the dataset, see below)
tools/app_export/.venv/bin/python tools/app_export/export_replay.py

# OSM roads around each scenario -> app/assets/replays/<id>_roads.json (needs network)
tools/app_export/.venv/bin/python tools/app_export/export_osm.py

# check the Dart engine against all of it
cd app && flutter test test/engine
```

| Script | What it does |
|---|---|
| `export_model.py` | Loads `results/crossdrive_v5_model.pth`, folds BatchNorm into the dense layers, and writes a float32 blob plus a manifest (input/output normalisation, tensor offsets, feature order). It checks that the folded weights reproduce the torch model (max diff about 2e-6). |
| `export_replay.py` | Cuts the five README outage scenarios out of IO-VNBD Drive M (60 s lead-in, the outage, and 30 s after). It records the Python result for each in `index.json`; `test/engine/benchmark_test.dart` must reproduce it. |
| `export_osm.py` | Downloads OpenStreetMap roads around each replay scenario (Overpass) into `app/assets/replays/<id>_roads.json`, for the road-matching benchmark. |
| `make_goldens.py` | Runs a synthetic 90 s drive through the project's own v5 functions (`extract_features_v5`, `predict`, gate, EKF, cruise lock, ZUPT, `OnlineGNSSCalibrator`, `dead_reckon_2d`, `map_match_to_road`) and saves every stage's output. |

Re-run the scripts whenever the model is retrained.

**Dataset.** Only Drive M is needed (about 42 MB). It comes from https://github.com/onyekpeu/IO-VNBD (Git LFS) and goes here:
`data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/{S-M,V-M}.csv`

**Sensor axes.** The column mapping the model was trained with has a quirk: the gyro inputs are `[Yaw, Yaw, Roll]`. See [CANONICAL_FRAME.md](CANONICAL_FRAME.md).
