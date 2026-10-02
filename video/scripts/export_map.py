"""Export the 'highway tunnel' tunnel test (IO-VNBD Drive M) from the app's replay assets to public/map.json.

Route and roads in local metres around the outage, IDR's along-track distance during the
30 s GPS outage (calibrated speed, scaled to the published 660 m), and normalised sensor traces.
Run from the repo root:  python3 video/scripts/export_map.py
"""
import json, math, statistics as st

SRC = 'app/assets/replays/'
d = json.load(open(SRC + 'highway_tunnel.json')); g = d['gt']; e = d['python']['enhanced']
ref = next(s for s in json.load(open(SRC + 'index.json'))['scenarios'] if s['id'] == 'highway_tunnel')['reference']
bias = ref['bias']; o0, o1 = d['outage_start'], d['outage_end']
lat0, lon0 = g['lat'][(o0 + o1) // 2], g['lon'][(o0 + o1) // 2]
k = math.cos(math.radians(lat0)) * 111320
P = lambda la, lo: [round((lo - lon0) * k, 2), round((la - lat0) * 110540, 2)]
i0, i1 = 2650, 3600
route = [P(g['lat'][i], g['lon'][i]) for i in range(i0, i1)]
cum = [0.0]
for a, b in zip(route, route[1:]): cum.append(cum[-1] + math.dist(a, b))
sc = ref['pred_distance_m'] / sum(max(0, e[i] - bias) * 0.1 for i in range(o0, o1))
sidr, acc = [], cum[o0 - i0]
for i in range(o0, o1 + 1):
    sidr.append(round(acc, 2)); acc += max(0, e[i] - bias) * 0.1 * sc
ways = []
for w in json.load(open(SRC + 'highway_tunnel_roads.json'))['ways']:
    pts = [P(w['g'][j], w['g'][j + 1]) for j in range(0, len(w['g']), 2)]
    if any(abs(x) < 1600 and abs(y) < 1100 for x, y in pts): ways.append({'hw': w['hw'], 'p': pts})
sens = {}
for key in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
    v = d['sensors'][key][i0:i1]; mu = st.mean(v); sd = st.pstdev(v) or 1
    sens[key] = [round((x - mu) / sd, 3) for x in v]
out = dict(route=route, cum=[round(c, 2) for c in cum], speed=[g['speed_ms'][i] for i in range(i0, i1)], i0=i0,
           o0=o0 - i0, o1=o1 - i0, sidr=sidr, ways=ways, sens=sens,
           vest=[round(max(0, x - bias) * 3.6, 1) for x in e[i0:i1]])
json.dump(out, open('video/public/map.json', 'w'))
print('route', len(route), 'ways', len(ways), 'idr distance', round(sidr[-1] - sidr[0], 1))
