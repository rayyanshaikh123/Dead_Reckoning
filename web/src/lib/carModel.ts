// Drop a car model at `public/models/car.glb` and the site uses it instead of
// the built-in stylised car (next.config.ts checks for it at build/dev start,
// so restart `npm run dev` after adding it). GLB (binary glTF) with Draco compression is best;
// keep it under ~5 MB so the page loads fast.
//
// The model is normalised automatically: scaled to `length`, set on the
// ground and centred. Parts are found by their mesh / material names:
//   headlights  — "headlight", "headlamp", "front light", "drl"
//   tail lights — "taillight", "tail light", "rear light", "brake"
//   wheels      — "wheel", "tire", "tyre", "rim"  (spun while driving)
// If the car ends up facing backwards or sideways, set `yaw` (radians).

export const CAR_MODEL = {
  url: process.env.NEXT_PUBLIC_CAR_MODEL ?? "",
  /** Extra turn about the vertical axis so the nose points forwards: 0, Math.PI/2, Math.PI or -Math.PI/2. */
  yaw: 0,
  /** Length the model is scaled to (metres). */
  length: 4.7,
};
