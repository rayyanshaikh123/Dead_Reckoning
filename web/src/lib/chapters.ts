import { EXIT_ERROR_M } from "./timeline";

/** Optional visual under a chapter's text. */
export type Detail =
  | { kind: "stats"; items: readonly (readonly [string, string])[] }
  | { kind: "pipeline"; steps: readonly string[] }
  | { kind: "gps"; on: boolean }
  | { kind: "bars"; items: readonly (readonly [string, number, string])[] };

export type Chapter = {
  /** Scroll window [from, to] where the chapter is on screen. */
  at: readonly [number, number];
  label: string;
  title: string;
  body: string;
  detail?: Detail;
  /** Anchor for the nav links. */
  id?: string;
};

// Every number here is from the project: the model and its benchmarks.
export const CHAPTERS: readonly Chapter[] = [
  {
    at: [0.155, 0.3],
    label: "GPS healthy",
    title: "IDR learns your car",
    body: "While GPS is strong, IDR calibrates itself: your car’s idle vibration, how the phone sits in its mount, and how speed feels to the sensors.",
    detail: {
      kind: "stats",
      items: [
        ["10 Hz", "sensor reads"],
        ["18", "motion features"],
        ["2.1 s", "of memory"],
      ],
    },
  },
  {
    at: [0.31, 0.385],
    label: "GPS lost",
    title: "Satellites can’t reach you",
    body: "Inside the tunnel the GPS signal is gone. Most map apps freeze, or keep guessing from your last known speed.",
    detail: { kind: "gps", on: false },
  },
  {
    at: [0.39, 0.455],
    label: "IDR takes over",
    title: "An AI reads the car’s motion",
    body: "IDNN v5, a small neural network, turns vibration and motion into speed ten times a second — on the phone, with no internet.",
    detail: {
      kind: "stats",
      items: [
        ["140k", "parameters"],
        ["0.2 ms", "per step"],
        ["0", "servers"],
      ],
    },
  },
  {
    at: [0.46, 0.515],
    label: "Physics check",
    title: "Physics keeps the AI honest",
    body: "Each estimate passes through physics filters: impossible jumps are rejected, cruising is held steady, and stops snap to zero.",
    detail: { kind: "pipeline", steps: ["AI", "Gate", "EKF", "Cruise", "ZUPT"] },
  },
  {
    at: [0.52, 0.575],
    label: "The road",
    title: "The map knows where the road goes",
    body: "IDR follows the road network from OpenStreetMap, keeping every possible route alive at a junction until the motion picks one.",
    detail: { kind: "pipeline", steps: ["Speed", "Heading", "Roads", "Position"] },
  },
  {
    at: [0.585, 0.7],
    label: "GPS back",
    title: `Off by ${EXIT_ERROR_M} m after 648 m`,
    body: "1.9 % drift on the highway-tunnel benchmark, against a Smart India Hackathon target of under 10 %.",
    detail: {
      kind: "bars",
      items: [
        ["IDR", 1.9, "1.9 %"],
        ["SIH target", 10, "< 10 %"],
      ],
    },
  },
  {
    at: [0.8, 0.955],
    label: "The hardware",
    title: "Just the phone you already have",
    body: "No dongle, no CAN-bus, no roof antenna. The phone’s motion sensors and GPS are all IDR needs.",
    id: "sensors",
  },
];

/** Fade a chapter in over the first part of its window and out over the last. */
export function chapterOpacity(c: Chapter, s: number) {
  const [a, b] = c.at;
  const edge = Math.min(0.03, (b - a) * 0.28);
  if (s <= a || s >= b) return 0;
  return Math.min(1, (s - a) / edge, (b - s) / edge);
}
