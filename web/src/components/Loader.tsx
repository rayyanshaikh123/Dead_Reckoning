"use client";

import { useEffect, useRef, useState } from "react";

import { story } from "@/lib/story";

import styles from "./loader.module.css";

/**
 * Black screen with a thin progress line while the 3D scene loads. When it's
 * ready this fades away on the car sitting in the dark — the headlights then
 * come on in the scene itself.
 */
export function Loader({ ready }: { ready: boolean }) {
  const [gone, setGone] = useState(false);
  const bar = useRef<HTMLSpanElement>(null);
  const label = useRef<HTMLSpanElement>(null);

  // Real download progress when there's a model; otherwise ease towards 90 %.
  useEffect(() => {
    let shown = 0;
    const id = setInterval(() => {
      const target = ready ? 1 : Math.max(story.loaded * 0.95, Math.min(0.9, shown + (0.9 - shown) * 0.06));
      shown += (target - shown) * (ready ? 0.5 : 0.35);
      if (bar.current) bar.current.style.transform = `scaleX(${shown.toFixed(3)})`;
      if (label.current) label.current.textContent = `${Math.round(shown * 100)}%`;
    }, 50);
    return () => clearInterval(id);
  }, [ready]);

  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(() => setGone(true), 1000);
    return () => clearTimeout(t);
  }, [ready]);

  if (gone) return null;
  return (
    <div className={`${styles.loader} ${ready ? styles.leaving : ""}`} role="status" aria-live="polite">
      <div className={styles.center}>
        <p className={styles.brand}>
          IDR<span className={styles.dot} />
        </p>
        <span className={styles.track}>
          <span ref={bar} className={styles.bar} />
        </span>
        <p className={styles.status}>
          <span>{ready ? "ready" : "starting the engine"}</span>
          <span ref={label}>0%</span>
        </p>
      </div>
    </div>
  );
}
