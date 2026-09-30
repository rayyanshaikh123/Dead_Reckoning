// Scroll-driven story: every 3D value is a function of `s`, the scroll
// progress through the pinned story section (0 = top, 1 = end).
//
//   intro      (before scrolling) the car in the dark, headlights on
//   0.00–0.14  hero: the car turns on a showcase stage
//   0.14–0.22  camera drops behind the car, the road fades in
//   0.22–0.66  drive: GPS → tunnel (GPS lost, IDR on sensors) → exit
//   0.66–0.80  camera rises to a top view, the world fades away
//   0.80–0.94  top view: the phone's sensors are called out
//   0.94–1.00  pull back, hand over to the app carousel

export type Vec3 = [number, number, number];
type Key<T> = readonly [number, T];

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
export const ease = (t: number) => t * t * (3 - 2 * t);

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

/** Piecewise eased interpolation between keyframes (sorted by s). */
export function num(keys: readonly Key<number>[], s: number): number {
  if (s <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [s1, v1] = keys[i];
    if (s <= s1) {
      const [s0, v0] = keys[i - 1];
      return lerp(v0, v1, ease((s - s0) / (s1 - s0)));
    }
  }
  return keys[keys.length - 1][1];
}

export function vec(keys: readonly Key<Vec3>[], s: number): Vec3 {
  return [0, 1, 2].map((k) => num(keys.map(([t, v]) => [t, v[k]] as const), s)) as Vec3;
}

/** Fraction of the way through [a, b], clamped (no easing). */
export const span = (s: number, a: number, b: number) => clamp01((s - a) / (b - a));

// ---- geometry of the drive (metres; the car drives towards -z) ----------

export const TUNNEL_START = 90; // distance at the tunnel mouth
export const TUNNEL_END = 290; // distance at the tunnel exit
export const DRIVE_TOTAL = 360; // distance driven by the end of the drive
export const EXIT_ERROR_M = 12; // IDR vs GPS at the exit (highway benchmark: 12.3 m)

// The drive is a real speed profile: pull away, cruise at a steady speed
// through the tunnel, brake to a stop. (Not an eased curve — that makes the
// car creep, then race, then creep.)
const DRIVE_FROM = 0.22;
const DRIVE_TO = 0.66;
const ACCEL = 0.035; // share of scroll spent pulling away
const BRAKE = 0.035; // …and braking
const CRUISE = DRIVE_TOTAL / (DRIVE_TO - DRIVE_FROM - ACCEL / 2 - BRAKE / 2); // metres per unit of s
/** ∫₀ˣ smoothstep — distance covered while speed ramps up smoothly. */
const rampArea = (x: number) => x * x * x - (x * x * x * x) / 2;

/** Distance driven along the road. */
export function distance(s: number) {
  if (s <= DRIVE_FROM) return 0;
  if (s >= DRIVE_TO) return DRIVE_TOTAL;
  let d = CRUISE * ACCEL * rampArea(Math.min(1, (s - DRIVE_FROM) / ACCEL));
  const cruiseFrom = DRIVE_FROM + ACCEL;
  const cruiseTo = DRIVE_TO - BRAKE;
  if (s > cruiseFrom) d += CRUISE * (Math.min(s, cruiseTo) - cruiseFrom);
  if (s > cruiseTo) {
    const x = (s - cruiseTo) / BRAKE;
    d += CRUISE * BRAKE * (x - rampArea(x));
  }
  return d;
}

/** Speed as a share of cruising speed (0 standing, 1 cruising). */
export function speedShare(s: number) {
  if (s <= DRIVE_FROM || s >= DRIVE_TO) return 0;
  if (s < DRIVE_FROM + ACCEL) return ease((s - DRIVE_FROM) / ACCEL);
  if (s > DRIVE_TO - BRAKE) return 1 - ease((s - (DRIVE_TO - BRAKE)) / BRAKE);
  return 1;
}

/**
 * Small, realistic lane wander while driving (metres sideways, by distance):
 * nobody drives a perfectly straight line. Zero when parked.
 */
export function laneOffset(d: number) {
  const onRoad = Math.min(1, d / 25, (DRIVE_TOTAL - d) / 25);
  return Math.max(0, onRoad) * (Math.sin(d * 0.021) * 0.16 + Math.sin(d * 0.053 + 1.3) * 0.05);
}

export type Phase = "gps" | "sensors" | "back";
export function phase(s: number): Phase {
  const d = distance(s);
  if (d < TUNNEL_START) return "gps";
  if (d < TUNNEL_END) return "sensors";
  return "back";
}

// ---- car ------------------------------------------------------------------

/**
 * The model faces +x; +π/2 turns it to drive towards -z. The showroom turn
 * finishes (pointing down the road) before the road fades in, so the car
 * never slides sideways on it.
 */
export const carYaw = (s: number) =>
  num(
    [
      [0.0, 0.75],
      [0.065, -0.35],
      [0.13, Math.PI / 2],
    ],
    s,
  );

export const carScale = (s: number) => num([[0.94, 1], [1.0, 0.78]], s);

// ---- camera -------------------------------------------------------------

export const cameraPosition = (s: number) =>
  vec(
    [
      [0.0, [5.4, 1.7, 5.2]],
      [0.07, [-5.6, 1.5, 4.4]],
      [0.14, [-4.6, 2.4, 7.6]],
      [0.22, [-1.9, 2.9, 9.6]],
      [0.44, [1.6, 2.6, 9.2]],
      [0.62, [-1.5, 2.9, 9.4]],
      [0.7, [3.4, 4.2, 6.4]],
      [0.8, [0.0, 12.5, 0.02]],
      [0.94, [0.0, 11.0, 0.02]],
      [1.0, [0.0, 17.0, 0.02]],
    ],
    s,
  );

export const cameraTarget = (s: number) =>
  vec(
    [
      [0.0, [0, 0.65, 0]],
      [0.14, [0, 0.7, 0]],
      [0.22, [0, 0.9, -5]],
      [0.62, [0, 0.9, -6]],
      [0.7, [0, 0.5, -1]],
      [0.8, [0, 0, 0]],
    ],
    s,
  );

/**
 * Intro shot: low, in front of the car, looking into the headlights. Blends
 * into the hero camera by orbiting round the car (never through it).
 */
export function introCamera(intro: number, hero: Vec3): Vec3 {
  const t = ease(clamp01(intro));
  // The nose points along +x rotated by the yaw; as an xz angle that's -yaw.
  const a0 = -carYaw(0);
  const r0 = 6.4;
  const y0 = 0.85;
  const a1 = Math.atan2(hero[2], hero[0]);
  const r1 = Math.hypot(hero[0], hero[2]);
  let da = a1 - a0;
  if (da > Math.PI) da -= 2 * Math.PI;
  if (da < -Math.PI) da += 2 * Math.PI;
  const a = a0 + da * t;
  const r = lerp(r0, r1, t);
  return [Math.cos(a) * r, lerp(y0, hero[1], t), Math.sin(a) * r];
}

/** 1 in the hero (car framed right of the headline), easing to 0 as it fades. */
export const heroShift = (s: number) => num([[0, 1], [0.1, 0]], s);

// ---- world ----------------------------------------------------------------

/** Road, tunnel and scenery (fade in behind the hero, out for the top view). */
export const worldOpacity = (s: number) =>
  num(
    [
      [0.14, 0],
      [0.22, 1],
      [0.7, 1],
      [0.78, 0],
    ],
    s,
  );

/** Showcase stage under the car (hero and top view). */
export const stageOpacity = (s: number) =>
  num(
    [
      [0.0, 1],
      [0.16, 0],
      [0.72, 0],
      [0.8, 1],
    ],
    s,
  );

/** Sensor callouts in the top view. */
export const sensorsOpacity = (s: number) => num([[0.8, 0], [0.85, 1], [0.93, 1], [0.97, 0]], s);

/** "GPS back" markers after the exit. */
export const exitMarkersOpacity = (s: number) => {
  const d = distance(s);
  return d < TUNNEL_END ? 0 : num([[0.6, 1], [0.7, 1], [0.74, 0]], s);
};

/** Driving HUD (speed, GPS state). */
export const hudOpacity = (s: number) => num([[0.19, 0], [0.23, 1], [0.66, 1], [0.7, 0]], s);

/** Is anything animating on its own clock (so the scene must keep rendering)? */
export const selfAnimating = (s: number) =>
  (phase(s) === "sensors" && worldOpacity(s) > 0) || sensorsOpacity(s) > 0;
