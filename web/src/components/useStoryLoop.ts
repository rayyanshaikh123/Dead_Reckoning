"use client";

import { useEffect, useRef, useState } from "react";

import { emitStory, story } from "@/lib/story";
import { selfAnimating } from "@/lib/timeline";

/**
 * Intro, in order:
 *   loading   black screen while the 3D scene loads
 *   dark      the loading screen fades; the car is there, unlit
 *   igniting  headlights flicker on (left, then right)
 *   waiting   headlights on, "scroll" hint; waits for a scroll or a moment
 *   rising    studio lights come up, the camera orbits into the hero
 *   done      the page scrolls; the story is driven by scroll
 */
export type IntroStage = "loading" | "dark" | "igniting" | "waiting" | "rising" | "done";

const DARK_MS = 500;
const IGNITE_MS = 1100;
const WAIT_MS = 900;
const RISE_MS = 1600;
const RISE_FAST_MS = 1000;

/** Headlight ignition: a couple of flickers, then full. `t` in seconds. */
function flicker(t: number) {
  const keys: [number, number][] = [
    [0, 0],
    [0.05, 0.75],
    [0.11, 0.08],
    [0.19, 0.95],
    [0.25, 0.3],
    [0.33, 0.85],
    [0.6, 1],
  ];
  if (t <= 0) return 0;
  for (let i = 1; i < keys.length; i++) {
    const [t1, v1] = keys[i];
    if (t <= t1) {
      const [t0, v0] = keys[i - 1];
      return v0 + ((v1 - v0) * (t - t0)) / (t1 - t0);
    }
  }
  return 1;
}

/**
 * The page's single animation loop: reads scroll, smooths it, runs the intro,
 * updates DOM overlays (`onStory`) and asks the canvas for a frame only when
 * something on it changed.
 */
export function useStoryLoop(storyRef: React.RefObject<HTMLElement | null>, ready: boolean) {
  const [stage, setStage] = useState<IntroStage>("loading");
  const readyRef = useRef(false);
  const request = useRef(false);

  useEffect(() => {
    readyRef.current = ready;
  }, [ready]);

  // Scroll stays locked until the intro has handed over to the hero.
  useEffect(() => {
    document.documentElement.classList.toggle("intro-lock", stage !== "done");
  }, [stage]);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    // Came back mid-page (reload, back button): skip the intro.
    const skip = reduced || window.scrollY > 40;
    let current: IntroStage = "loading";
    let since = performance.now();
    const go = (next: IntroStage, now: number) => {
      current = next;
      since = now;
      setStage(next);
    };

    const onInput = (e: Event) => {
      if (e instanceof KeyboardEvent && !["ArrowDown", "PageDown", " ", "Enter"].includes(e.key)) return;
      request.current = true;
    };
    window.addEventListener("wheel", onInput, { passive: true });
    window.addEventListener("touchmove", onInput, { passive: true });
    window.addEventListener("keydown", onInput);

    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    if (storyRef.current) io.observe(storyRef.current);

    let raf = 0;
    let last = performance.now();
    let riseMs = RISE_MS;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const t = now - since;
      const prev = { s: story.s, intro: story.intro, l: story.lampL, r: story.lampR };

      // ---- intro --------------------------------------------------------
      switch (current) {
        case "loading":
          if (readyRef.current) {
            if (skip) {
              story.intro = 1;
              story.lampL = story.lampR = 1;
              go("done", now);
            } else go("dark", now);
          }
          break;
        case "dark":
          if (t > DARK_MS) go("igniting", now);
          break;
        case "igniting":
          story.lampL = flicker(t / 1000);
          story.lampR = flicker(t / 1000 - 0.14);
          if (t > IGNITE_MS) {
            story.lampL = story.lampR = 1;
            go(request.current ? "rising" : "waiting", now);
          }
          break;
        case "waiting":
          if (request.current || t > WAIT_MS) {
            riseMs = request.current ? RISE_FAST_MS : RISE_MS;
            go("rising", now);
          }
          break;
        case "rising":
          story.intro = Math.min(1, t / riseMs);
          if (story.intro >= 1) go("done", now);
          break;
      }

      // ---- scroll ---------------------------------------------------------
      const el = storyRef.current;
      if (el) {
        const r = el.getBoundingClientRect();
        story.target = Math.min(1, Math.max(0, -r.top / (r.height - window.innerHeight)));
      }
      const gap = story.target - story.s;
      story.s = Math.abs(gap) < 1e-5 ? story.target : story.s + gap * (1 - Math.exp(-9 * dt));

      const changed =
        prev.s !== story.s || prev.intro !== story.intro || prev.l !== story.lampL || prev.r !== story.lampR;
      if (changed) emitStory();
      if (visible && (changed || selfAnimating(story.s) || current === "loading")) story.invalidate();
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      window.removeEventListener("wheel", onInput);
      window.removeEventListener("touchmove", onInput);
      window.removeEventListener("keydown", onInput);
      document.documentElement.classList.remove("intro-lock");
    };
  }, [storyRef]);

  return { stage, start: () => (request.current = true) };
}
