"""
Download OpenStreetMap roads around each replay scenario so the app's road
matcher can be benchmarked on real map data (instead of snapping to the
ground-truth track, which is what src/simulate_blackout.py does).

Writes app/assets/replays/<id>_roads.json in the app's compact road format:
  {"ways": [{"id": int, "n": [node ids], "g": [lat, lon, lat, lon, ...],
             "hw": highway, "ow": 1 | -1 | 0, "t": 0 | 1}]}

Usage (from repo root, needs network; ~5 small Overpass requests):
    tools/app_export/.venv/bin/python tools/app_export/export_osm.py
"""

import json
import os
import time
import urllib.parse
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPLAYS = os.path.join(REPO, "app", "assets", "replays")
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# Same filter as the app (lib/data/roads/overpass_client.dart).
HIGHWAYS = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|"
            "service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
MARGIN_DEG = 0.004  # ~400 m


def query(s, w, n, e):
    return (
        f'[out:json][timeout:60];'
        f'way["highway"~"^({HIGHWAYS})$"]["service"!~"parking_aisle|driveway"]["area"!="yes"]'
        f'({s},{w},{n},{e});out body geom qt;'
    )


def fetch(q):
    last = None
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                req = urllib.request.Request(
                    url,
                    data=urllib.parse.urlencode({"data": q}).encode(),
                    headers={"User-Agent": "IDR/1.0 (SIH dead-reckoning research)"},
                )
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.loads(r.read())
            except Exception as ex:  # noqa: BLE001
                last = ex
                time.sleep(3 * (attempt + 1))
    raise SystemExit(f"Overpass failed: {last}")


def oneway(tags):
    ow = tags.get("oneway", "")
    if ow in ("yes", "1", "true"):
        return 1
    if ow == "-1":
        return -1
    if ow == "no":
        return 0
    if tags.get("highway") in ("motorway", "motorway_link") or tags.get("junction") in ("roundabout", "circular"):
        return 1
    return 0


def compact(data):
    ways = []
    for el in data["elements"]:
        if el["type"] != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        g = []
        for p in el["geometry"]:
            g += [round(p["lat"], 7), round(p["lon"], 7)]
        ways.append({
            "id": el["id"], "n": el["nodes"], "g": g,
            "hw": tags.get("highway", ""), "ow": oneway(tags),
            "t": 1 if tags.get("tunnel") not in (None, "no") else 0,
        })
    return {"ways": ways}


def main():
    index = json.load(open(os.path.join(REPLAYS, "index.json")))
    for sc in index["scenarios"]:
        sid = sc["id"]
        d = json.load(open(os.path.join(REPLAYS, f"{sid}.json")))
        lat, lon = d["gt"]["lat"], d["gt"]["lon"]
        s, n = min(lat) - MARGIN_DEG, max(lat) + MARGIN_DEG
        w, e = min(lon) - MARGIN_DEG * 1.6, max(lon) + MARGIN_DEG * 1.6
        roads = compact(fetch(query(s, w, n, e)))
        out = os.path.join(REPLAYS, f"{sid}_roads.json")
        with open(out, "w") as f:
            json.dump(roads, f, separators=(",", ":"))
        print(f"{sid:16s} {len(roads['ways']):4d} ways  {os.path.getsize(out) / 1024:6.0f} KB")
        time.sleep(2)  # be polite to the public Overpass servers


if __name__ == "__main__":
    main()
