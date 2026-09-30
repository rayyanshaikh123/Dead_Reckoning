// Points on the car the story talks about (car coordinates, metres; nose
// towards +x, y up). Plain numbers so the DOM overlay can import them without
// pulling three.js into the main bundle.

export type Point = readonly [number, number, number];

/** Phone in its dash mount — the only hardware IDR needs. */
export const PHONE: Point = [0.62, 1.12, 0.32];
/** Where GPS comes from (the phone's receiver, drawn above the roof). */
export const GPS: Point = [-0.2, 1.62, 0];
/** Rear wheel — "nothing else is wired to the car". */
export const REAR_WHEEL: Point = [-1.45, 0.4, 0.95];
