# IDR — website

A scroll-driven 3D showcase for IDR, built with Next.js 16 and react-three-fiber. It opens on the car in the dark with its headlights on, turns into the hero, then follows the car into a tunnel where GPS is lost and IDR takes over, back out, and up to a top view of the sensors it uses. It ends on the app screens and a download button.

```bash
cd web
npm install
npm run dev                  # http://localhost:3000
npm run build && npm start   # judge speed on this, not on `dev`
```

If npm fails with a permission error on `~/.npm`, that cache folder is owned by root. Either run `sudo chown -R $(id -u) ~/.npm` once, or prefix npm commands with `npm_config_cache=/tmp/npm-cache`.

## How it works

- **One animation loop** (`src/components/useStoryLoop.ts`) reads the scroll position, smooths it, runs the intro, and writes to the shared `story` state (`src/lib/story.ts`). Nothing re-renders React while scrolling: the page text overlays subscribe with `onStory` and write straight to the DOM.
- **The 3D canvas renders on demand** (`frameloop="demand"`): only when something moved, and not at all once it's scrolled out of view.
- **Everything is a function of scroll progress `s`** (`src/lib/timeline.ts`): camera path, car turn, the drive's speed profile (pull away, cruise, brake), lane wander, when the world, stage and labels fade. The seven story chapters and their windows are in `src/lib/chapters.ts`.
- **The scene** (`src/components/scene/`): `Scene.tsx` (lights, camera, loading), `World.tsx` (terrain, tunnel, trees, road, sky), `Car.tsx` (built-in car, headlights, tail glow), `ModelCar.tsx` (a provided car model), `Sensors.tsx` (sensor labels in the top view), `glow.ts` (light glows).

## The car model

The site uses `public/models/car.glb` if it exists, otherwise a built-in stylised car. `next.config.ts` checks for it when the build or dev server starts, so restart `npm run dev` after adding one.

The current model is a 1982 Mercedes W201 (Sketchfab). Its original (26 MB) lives in `model-src/` (git-ignored) and is converted with:

```bash
npm run optimize:car -- model-src/1982_mercedes_w201.glb
```

`scripts/optimize-car.mjs` removes the interior and engine, turns the windows into dark tinted glass, tints the paint graphite, renames the lamps so the site can light them, splits each wheel into four corners so they spin, converts textures to WebP (≤ 2048 px) and compresses the geometry. The result is 4.3 MB.

The part names it looks for are specific to that model. For another car, adjust the patterns at the top of the script, and if the car faces the wrong way, set `yaw` in `src/lib/carModel.ts`.

## Download button

The **Download APK** buttons link to `idr.apk` on the repo's latest GitHub release:

```
https://github.com/rayyanshaikh123/Dead_Reckoning/releases/latest/download/idr.apk
```

GitHub serves it as `application/vnd.android.package-archive` with `Content-Disposition: attachment`, so Android phones offer to install it. The APK isn't committed or deployed with the site. To ship a new build, publish a new release with the APK attached as `idr.apk`; the site picks it up without a redeploy. See [`app/RELEASE.md`](../app/RELEASE.md#apk-for-the-website).

To link elsewhere instead, set `NEXT_PUBLIC_APK_URL` at build time:

```bash
NEXT_PUBLIC_APK_URL=https://…/idr.apk npm run build
```

## Deploying to Vercel

The site is fully static (every route is prerendered), so it needs no server settings or environment variables.

1. In Vercel, **Add New → Project** and import this GitHub repo.
2. Set **Root Directory** to `web`. The Next.js app lives here, not at the repo root. Vercel detects Next.js and uses `npm install` / `next build` on its own.
3. Optional: set `NEXT_PUBLIC_APK_URL` under **Environment Variables** to override the download link. Without it, the buttons link to the latest GitHub release.
4. Deploy.

Don't set `NEXT_DIST_DIR` on Vercel. Vercel expects the build in `.next`.

## Performance

The target is 60 fps on a MacBook Air, with headroom for 120 Hz screens.

- **No post-processing.** Bloom cost about 5 ms a frame and pushed frames past the 16.7 ms budget. Glows are additive sprites instead (`glow.ts`), and anti-aliasing is the GPU's own multisampling.
- **Resolution adapts.** The scene renders up to 2× within a pixel budget, and steps down if frames miss the display's refresh rate.
- **Shaders compile behind the loading screen**, so nothing stalls mid-scroll.
- **The car's shadow** comes from a small map covering only the area around the car, and is only redrawn while its light is on.
- **Repeated objects are instanced:** about 1,400 trees, the tunnel lights, rails and lamps each take a single draw call.

Safari limits page animation to about 60 fps by default. To test 120 Hz there: Settings → Advanced → *Show features for web developers*, then Develop → Feature Flags → untick *Prefer Page Rendering Updates near 60fps*.

URL switches for measuring: `?fixed` holds the resolution steady, and `?perf` exposes the renderer as `window.__idr`.

## Test builds without breaking a running server

`npm run build` rewrites `.next`, which a running `npm start` or `npm run dev` is serving from, and that causes "ChunkLoadError" 500s. Build test copies elsewhere:

```bash
NEXT_DIST_DIR=.next-verify npm run build
NEXT_DIST_DIR=.next-verify npx next start -p 3456
```
