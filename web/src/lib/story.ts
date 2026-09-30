// Shared state for the scroll story. One requestAnimationFrame loop (see
// `useStoryLoop`) advances it; the 3D scene and the DOM overlays read it.
// Nothing here causes React renders — overlays write to the DOM directly.

export type StoryFrame = {
  /** Smoothed scroll progress through the story (0–1). */
  s: number;
  /** Intro: 0 = the car in the dark, headlights only; 1 = the lit hero. */
  intro: number;
};

export const story = {
  /** Raw scroll progress (what the page says). */
  target: 0,
  /** Smoothed progress (what everything renders). */
  s: 0,
  /** Eased intro progress. */
  intro: 0,
  /** Headlight brightness, left and right (0–1, flickers while igniting). */
  lampL: 0,
  lampR: 0,
  /**
   * Shader warm-up behind the loading screen: 1 = show fading scenery, 2 = show
   * it opaque (so both shader variants get compiled up front), 0 = normal.
   */
  warmup: 0,
  /** 3D assets loaded (0–1), for the loading screen. */
  loaded: 0,
  /** Set by the scene: request one more rendered frame. */
  invalidate: () => {},
};

type Updater = (f: StoryFrame) => void;
const updaters = new Set<Updater>();

/** Registers a DOM overlay to be updated whenever the story moves. */
export function onStory(fn: Updater) {
  updaters.add(fn);
  fn({ s: story.s, intro: story.intro });
  return () => {
    updaters.delete(fn);
  };
}

export function emitStory() {
  const f = { s: story.s, intro: story.intro };
  updaters.forEach((fn) => fn(f));
}
