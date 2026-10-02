# IDR — product story video

A 40-second explainer for IDR, made with [Remotion](https://www.remotion.dev): the problem (GPS drops in tunnels and plain sensor math drifts), how IDR works (sense, think, stay on the road), a replay of a real tunnel test from the app's benchmark data, and the logo.

```bash
cd video
npm install
npm run studio      # preview and scrub in the browser
npm run render      # → out/idr-story.mp4 (1920×1080, 24 fps)
```

The 3D scenes use WebGL and render on the GPU by default (`--gl=angle`, set in `remotion.config.ts`). On a machine without a GPU, add `--gl=swiftshader` (much slower). For 4K, add `--scale=2`.

## How it's built

- `src/IDRStory.tsx` — the composition: a WebGL canvas, the React overlays, the score and the voiceover lines.
- `src/engine.js` — three.js scenes: the 3D phone (light-beam reveal, x-rayed IMU, hero shot) and the top-down map of the real drive (GPS loss, drift, IDR replay). `update(t)` poses everything for a time `t` and returns screen anchors for the labels; `draw()` renders.
- `src/Overlays.tsx` — all text, HUD, satellites and the live sensor panel, as React driven by the current frame.
- `src/timeline.ts` — section timings, voiceover cues and how the score ducks under the voice.

## Data and audio

- `public/map.json` comes from the app's replay assets (`app/assets/replays/highway_tunnel*.json`, IO-VNBD Drive M): `python3 video/scripts/export_map.py`. The IDR dot is the calibrated model speed integrated along the road; the red "plain math" track is an illustration tuned to the benchmark's 792 m exit error.
- `public/audio/score.wav` (music and sound design, synthesised) and `public/audio/v*.wav` (voiceover) are made by `scripts/vo.py` then `scripts/score.py`.
