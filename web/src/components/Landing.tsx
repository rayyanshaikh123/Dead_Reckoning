"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";

import { APK_URL, SOURCE_URL } from "@/lib/download";
import { onStory } from "@/lib/story";
import { ease, sensorsOpacity } from "@/lib/timeline";

import { AppCarousel } from "./AppCarousel";
import { ChapterRail, Chapters } from "./Chapters";
import { Hud } from "./Hud";
import { Loader } from "./Loader";
import { SensorCallouts, SensorLegend } from "./SensorCallouts";
import styles from "./landing.module.css";
import { useStoryLoop } from "./useStoryLoop";

const Scene = dynamic(() => import("./scene/Scene"), { ssr: false });

/** Story height in viewport heights; the 3D stage stays pinned through it. */
const STORY_VH = 1300;
/** Scroll progress of the sensors chapter (for the nav link). */
const SENSORS_AT = 0.87;

const smooth = (a: number, b: number, v: number) => ease(Math.min(1, Math.max(0, (v - a) / (b - a))));

export function Landing() {
  const story = useRef<HTMLElement>(null);
  const nav = useRef<HTMLElement>(null);
  const hero = useRef<HTMLDivElement>(null);
  const legend = useRef<HTMLElement | null>(null);
  const [ready, setReady] = useState(false);
  const onReady = useCallback(() => setReady(true), []);
  const { stage, start } = useStoryLoop(story, ready);

  // Things that follow the story, written straight to the DOM (only on change).
  useEffect(() => {
    let last = "";
    return onStory(({ s, intro }) => {
        const key = `${Math.min(s, 0.1).toFixed(4)}|${intro.toFixed(3)}|${sensorsOpacity(s).toFixed(3)}`;
        if (key === last) return;
        last = key;
        const inHero = smooth(0.5, 1, intro) * (1 - smooth(0, 0.08, s));
        if (hero.current) {
          hero.current.style.opacity = inHero.toFixed(3);
          hero.current.style.visibility = inHero > 0.001 ? "visible" : "hidden";
          hero.current.style.transform = `translate3d(0, ${((1 - smooth(0.5, 1, intro)) * 24 - smooth(0, 0.08, s) * 40).toFixed(1)}px, 0)`;
        }
        if (nav.current) nav.current.style.opacity = smooth(0.3, 1, intro).toFixed(3);
        if (legend.current) {
          const a = sensorsOpacity(s);
          legend.current.style.opacity = a.toFixed(3);
          legend.current.style.visibility = a > 0.001 ? "visible" : "hidden";
        }
    });
  }, []);

  return (
    <>
      <Loader ready={ready} />

      <header ref={nav} className={styles.nav}>
        <a href="#top" className={styles.wordmark} aria-label="IDR home">
          IDR<span className={styles.wordDot} />
        </a>
        <nav className={styles.links}>
          <a className="bracket" href="#story">
            story
          </a>
          <a className="bracket" href="#sensors">
            sensors
          </a>
          <a className="bracket" href="#app">
            app
          </a>
          <a className={`bracket ${styles.navCta}`} href="#download">
            download
          </a>
        </nav>
      </header>

      <main id="top">
        <section ref={story} id="story" className={styles.story} style={{ height: `${STORY_VH}vh` }}>
          <div className={styles.stage}>
            <Scene onReady={onReady} />
            <div className={styles.vignette} aria-hidden />
            <SensorCallouts />
            <SensorLegend
              refCallback={(el) => {
                legend.current = el;
              }}
            />
            <Chapters />
            <Hud />
            <ChapterRail />

            <div ref={hero} className={styles.hero}>
              <p className={styles.kicker}>Intelligent dead reckoning</p>
              <h1 className={styles.headline}>
                Keeps navigating
                <br />
                when GPS drops.
              </h1>
              <p className={styles.lede}>
                An on-device AI that reads your phone’s motion sensors to track your car through tunnels, underpasses
                and city canyons.
              </p>
              <div className={styles.heroActions}>
                <a className={styles.primary} href={APK_URL} download>
                  Download APK <span aria-hidden>↓</span>
                </a>
                <a className={`bracket ${styles.secondary}`} href="#app">
                  see the app
                </a>
              </div>
            </div>

            <button
              type="button"
              className={`${styles.hint} ${stage === "waiting" || stage === "igniting" ? styles.hintOn : ""}`}
              onClick={start}
              tabIndex={stage === "waiting" ? 0 : -1}
            >
              <span className={styles.hintLine} />
              scroll to start
            </button>
          </div>
          <span id="sensors" className={styles.anchor} style={{ top: `${SENSORS_AT * (STORY_VH - 100)}vh` }} />
        </section>

        <AppCarousel />

        <section id="download" className={styles.download}>
          <div className={styles.downloadInner}>
            <p className={styles.kicker}>Get IDR</p>
            <h2 className={styles.downloadTitle}>Try it on your next tunnel.</h2>
            <p className={styles.downloadText}>
              Android app. Mount the phone, drive once with GPS so IDR can calibrate — then it keeps you on the map when
              the signal drops. The AI runs on the phone; your drives stay on it.
            </p>
            <div className={styles.heroActions}>
              <a className={styles.primary} href={APK_URL} download>
                Download APK <span aria-hidden>↓</span>
              </a>
              <a className={`bracket ${styles.secondary}`} href={SOURCE_URL} target="_blank" rel="noreferrer">
                source on GitHub
              </a>
            </div>
          </div>
        </section>

        <footer className={styles.footer}>
          <div className={styles.stats}>
            {[
              ["0.2 ms", "per AI step, on the phone"],
              ["1.9 %", "exit error, highway tunnel"],
              ["105 km", "blind test drive (IO-VNBD)"],
              ["0", "extra hardware"],
            ].map(([v, l]) => (
              <div key={l} className={styles.stat}>
                <p className={styles.statValue}>{v}</p>
                <p className={styles.statLabel}>{l}</p>
              </div>
            ))}
          </div>
          <p className={styles.credits}>
            IDR · IDNN v5 · built for Smart India Hackathon · maps © OpenStreetMap contributors
          </p>
        </footer>
      </main>
    </>
  );
}
