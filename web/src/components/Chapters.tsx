"use client";

import { useEffect, useRef } from "react";

import { CHAPTERS, chapterOpacity, type Detail } from "@/lib/chapters";
import { onStory } from "@/lib/story";

import styles from "./chapters.module.css";

const pad = (n: number) => String(n).padStart(2, "0");

function DetailView({ detail }: { detail: Detail }) {
  switch (detail.kind) {
    case "stats":
      return (
        <dl className={styles.stats}>
          {detail.items.map(([v, l]) => (
            <div key={l}>
              <dt>{l}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      );
    case "pipeline":
      return (
        <ol className={styles.pipeline}>
          {detail.steps.map((step, i) => (
            <li key={step} className={i === detail.steps.length - 1 ? styles.pipeLast : undefined}>
              {step}
            </li>
          ))}
        </ol>
      );
    case "gps":
      return (
        <div className={styles.gps}>
          <span className={styles.gpsLabel}>GPS</span>
          <span className={styles.dots} data-on={detail.on}>
            {[0, 1, 2, 3, 4].map((i) => (
              <i key={i} />
            ))}
          </span>
          <span className={styles.gpsState}>{detail.on ? "locked" : "no signal"}</span>
        </div>
      );
    case "bars": {
      const max = Math.max(...detail.items.map(([, v]) => v));
      return (
        <div className={styles.bars}>
          {detail.items.map(([label, v, text], i) => (
            <div key={label} className={styles.bar}>
              <span className={styles.barLabel}>{label}</span>
              <span className={styles.barTrack}>
                <span className={i === 0 ? styles.barFillAccent : styles.barFill} style={{ width: `${(v / max) * 100}%` }} />
              </span>
              <span className={styles.barValue}>{text}</span>
            </div>
          ))}
        </div>
      );
    }
  }
}

/** The story text: one chapter at a time, cross-fading with scroll. */
export function Chapters() {
  const els = useRef<(HTMLElement | null)[]>([]);
  const shown = useRef<number[]>(CHAPTERS.map(() => -1));
  useEffect(
    () =>
      onStory(({ s }) => {
        CHAPTERS.forEach((c, i) => {
          const el = els.current[i];
          if (!el) return;
          const o = Math.round(chapterOpacity(c, s) * 500) / 500;
          if (o === shown.current[i]) return; // untouched: no style work
          shown.current[i] = o;
          const [a, b] = c.at;
          // rises in from below, leaves upwards
          const dir = s < (a + b) / 2 ? 1 : -1;
          el.style.opacity = o.toFixed(3);
          el.style.visibility = o > 0.001 ? "visible" : "hidden";
          el.style.transform = `translate3d(0, ${(dir * (1 - o) * 28).toFixed(1)}px, 0)`;
        });
      }),
    [],
  );
  return (
    <div className={styles.layer}>
      {CHAPTERS.map((c, i) => (
        <article
          key={c.title}
          ref={(el) => {
            els.current[i] = el;
          }}
          className={styles.chapter}
        >
          <p className={styles.kicker}>
            <span className={styles.index}>
              {pad(i + 1)} <span className={styles.of}>/ {pad(CHAPTERS.length)}</span>
            </span>
            <span className={styles.rule} />
            {c.label}
          </p>
          <h2 className={styles.title}>{c.title}</h2>
          <p className={styles.body}>{c.body}</p>
          {c.detail && <DetailView detail={c.detail} />}
        </article>
      ))}
    </div>
  );
}

/** Right-edge chapter index: where you are in the story. */
export function ChapterRail() {
  const root = useRef<HTMLDivElement>(null);
  const items = useRef<(HTMLElement | null)[]>([]);
  const last = useRef("");
  useEffect(
    () =>
      onStory(({ s, intro }) => {
        const r = root.current;
        if (!r) return;
        const opacity = Math.min(intro, s > 0.12 && s < 0.97 ? 1 : 0).toFixed(2);
        let active = -1;
        CHAPTERS.forEach((c, i) => {
          if (s >= c.at[0] - 0.01) active = i;
        });
        const key = `${opacity}|${active}`;
        if (key === last.current) return;
        last.current = key;
        r.style.opacity = opacity;
        items.current.forEach((el, i) => el?.toggleAttribute("data-active", i === active));
      }),
    [],
  );
  return (
    <div ref={root} className={styles.rail} aria-hidden>
      {CHAPTERS.map((c, i) => (
        <span
          key={c.title}
          ref={(el) => {
            items.current[i] = el;
          }}
          className={styles.railItem}
        >
          <span className={styles.railLabel}>
            {pad(i + 1)} · {c.label}
          </span>
          <span className={styles.railTick} />
        </span>
      ))}
    </div>
  );
}
