import { GPS, PHONE, REAR_WHEEL, type Point } from "./car";

export type Callout = {
  at: Point;
  title: string;
  /** One or two sentences for the label. */
  text: string;
  /** A few words for the phone legend. */
  short: string;
};

export const CALLOUTS: Callout[] = [
  {
    at: PHONE,
    title: "Motion sensors",
    text: "Accelerometer and gyroscope, read ten times a second from the phone in its mount.",
    short: "accelerometer + gyroscope, 10× a second",
  },
  {
    at: GPS,
    title: "GPS",
    text: "While it’s there, IDR calibrates against it. When it drops, IDR takes over.",
    short: "calibrates IDR while it lasts",
  },
  {
    at: REAR_WHEEL,
    title: "Nothing else",
    text: "No OBD dongle, no CAN-bus wiring. The AI runs on the phone, offline.",
    short: "no dongle, no wiring, no internet",
  },
];

/**
 * The callout elements, registered by the DOM overlay and positioned every
 * frame by the 3D scene (which knows where the anchors are on screen).
 */
export const calloutEls: (HTMLElement | null)[] = CALLOUTS.map(() => null);
