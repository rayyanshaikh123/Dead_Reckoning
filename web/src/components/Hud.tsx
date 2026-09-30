"use client";

import { useEffect, useRef } from "react";

import { onStory } from "@/lib/story";
import { EXIT_ERROR_M, TUNNEL_START, distance, hudOpacity, phase, speedShare } from "@/lib/timeline";

import styles from "./hud.module.css";

/** Speed shown in the drive: GPS says 72 when cruising; IDR's estimate wanders by a km/h. */
const speedAt = (s: number, d: number, ph: string) =>
  Math.round((ph === "sensors" ? 72 + Math.sin(d / 9) * 1.2 : 72) * speedShare(s));

/**
 * Driving read-out, styled like the app's dashboard: GPS signal (5 dots),
 * speed with its source, and time/distance on sensors alone.
 */
export function Hud() {
  const root = useRef<HTMLDivElement>(null);
  const speed = useRef<HTMLSpanElement>(null);
  const source = useRef<HTMLSpanElement>(null);
  const signal = useRef<HTMLSpanElement>(null);
  const state = useRef<HTMLSpanElement>(null);
  const rightLabel = useRef<HTMLSpanElement>(null);
  const rightValue = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let lastKey = "";
    let lastOpacity = "";
    return onStory(({ s }) => {
      const r = root.current;
      if (!r) return;
      const o = hudOpacity(s).toFixed(3);
      if (o !== lastOpacity) {
        lastOpacity = o;
        r.style.opacity = o;
        r.style.visibility = o !== "0.000" ? "visible" : "hidden";
      }
      if (o === "0.000") return;

      const d = distance(s);
      const ph = phase(s);
      const onSensors = Math.max(0, d - TUNNEL_START);
      const secs = Math.round(onSensors / 20); // ~72 km/h
      const key = `${ph}|${speedAt(s, d, ph)}|${secs}|${Math.round(onSensors)}`;
      if (key === lastKey) return;
      lastKey = key;

      r.dataset.phase = ph;
      speed.current!.textContent = String(speedAt(s, d, ph));
      source.current!.textContent = ph === "sensors" ? "IDR estimate" : "from GPS";
      signal.current!.dataset.on = String(ph !== "sensors");
      state.current!.textContent = ph === "gps" ? "locked" : ph === "sensors" ? "no signal" : "back";
      if (ph === "back") {
        rightLabel.current!.textContent = "Exit error";
        rightValue.current!.textContent = `${EXIT_ERROR_M} m`;
      } else {
        rightLabel.current!.textContent = "On sensors";
        rightValue.current!.textContent = `00:${String(secs).padStart(2, "0")} · ${String(Math.round(onSensors)).padStart(3, "0")} m`;
      }
    });
  }, []);

  return (
    <div ref={root} className={styles.hud} data-phase="gps" aria-hidden>
      <div className={styles.cell}>
        <span className={styles.label}>Signal</span>
        <span className={styles.row}>
          <span ref={signal} className={styles.dots} data-on="true">
            {[0, 1, 2, 3, 4].map((i) => (
              <i key={i} />
            ))}
          </span>
          <span ref={state} className={styles.state}>
            locked
          </span>
        </span>
      </div>
      <span className={styles.divider} />
      <div className={`${styles.cell} ${styles.center}`}>
        <span className={styles.speedRow}>
          <span ref={speed} className={styles.speed}>
            72
          </span>
          <span className={styles.unit}>km/h</span>
        </span>
        <span ref={source} className={styles.source}>
          from GPS
        </span>
      </div>
      <span className={styles.divider} />
      <div className={`${styles.cell} ${styles.end}`}>
        <span ref={rightLabel} className={styles.label}>
          On sensors
        </span>
        <span ref={rightValue} className={styles.value}>
          00:00 · 000 m
        </span>
      </div>
    </div>
  );
}
