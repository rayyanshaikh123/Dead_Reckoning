// Timing for the 40 s story cut (120 BPM, so one beat = 0.5 s).
export const FPS = 24;
export const DURATION_S = 40;

// Voiceover lines: [file, start (s), length (s)]
export const VO: [string, number, number][] = [
  ['v1', 1.6, 2.01],   // Every map app has a blind spot.
  ['v2', 5.3, 2.14],   // Tunnels. Underpasses. City canyons.
  ['v3', 7.7, 2.66],   // The moment GPS drops, your map freezes.
  ['v4', 10.5, 2.87],  // And phone sensors alone drift hundreds of meters.
  ['v5', 14.2, 2.74],  // IDR reads your phone's motion sensors.
  ['v6', 17.1, 4.16],  // An on-device AI turns them into speed and heading, ten times a second,
  ['v7', 21.4, 4.16],  // and keeps you on the road. No GPS. No internet. No extra hardware.
  ['v8', 26.0, 2.75],  // Six hundred forty-eight meters with no signal.
  ['v9', 31.45, 2.8],  // IDR comes out within fourteen meters.
  ['v10', 36.45, 2.54], // IDR. Lose the signal. Not the road.
];

export const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lin = (t: number, a: number, b: number) => clamp((t - a) / (b - a));
export const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
export const eo = (x: number) => 1 - Math.pow(1 - x, 3);
export const ei = (x: number) => x * x * x;
export const eio = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
/** 0 → 1 between a and b, holds, then 1 → 0 between c and d. */
export const env = (t: number, a: number, b: number, c: number, d: number) => Math.min(lin(t, a, b), 1 - lin(t, c, d));

/** How much the score is pulled down under the voice (0 = not at all, 1 = fully). */
export const duck = (t: number) => {
  let d = 0;
  for (const [, t0, len] of VO) d = Math.max(d, env(t, t0 - 0.18, t0 - 0.02, t0 + len + 0.1, t0 + len + 0.26));
  return d;
};
