"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

import { APK_URL } from "@/lib/download";

import styles from "./carousel.module.css";

// Real screens from the IDR app (rendered from the Flutter app's widgets).
const SCREENS = [
  { src: "/screens/01_intro.webp", title: "Onboarding", text: "Three screens, then you’re driving." },
  { src: "/screens/02_home.webp", title: "Home", text: "Speed, heading and GPS quality at a glance." },
  { src: "/screens/03_tunnel_test.webp", title: "Tunnel test", text: "Hide GPS and watch IDR take over." },
  { src: "/screens/04_live.webp", title: "Live", text: "The last minute of speed, and what IDR is doing." },
  { src: "/screens/06_history.webp", title: "History", text: "Every drive recorded automatically." },
  { src: "/screens/07_drive_report.webp", title: "Drive report", text: "Each GPS outage, scored against GPS." },
  { src: "/screens/08_settings.webp", title: "Settings", text: "Calibration, background running, offline maps." },
] as const;

const N = SCREENS.length;
const pad = (n: number) => String(n).padStart(2, "0");
/** Signed distance from the centre, wrapped so the ring never runs out. */
const wrap = (d: number) => ((((d + N / 2) % N) + N) % N) - N / 2;
const mod = (i: number) => ((i % N) + N) % N;
const spacingFor = (w: number) => (w < 700 ? w * 0.5 : Math.min(300, w * 0.2));
/** Scroll → carousel position, resting on each screen for most of its share. */
const stepped = (x: number) => {
  const i = Math.floor(x);
  const t = Math.min(1, Math.max(0, (x - i - 0.3) / 0.4));
  return i + t * t * (3 - 2 * t);
};

/**
 * App screens as an endless coverflow. While the section is pinned, scrolling
 * turns it one screen at a time; at the end of the section it stops on the
 * last screen and the page moves on. Buttons and dragging turn it too.
 */
export function AppCarousel() {
  const section = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const slides = useRef<(HTMLElement | null)[]>([]);
  const manual = useRef(0);
  const drag = useRef<null | { x: number; start: number }>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let pos = 0;
    let shown = -1;
    let running = false;

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const sec = section.current;
      const st = stage.current;
      if (!sec || !st) return;
      const r = sec.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, -r.top / (r.height - window.innerHeight)));
      const want = stepped(p * (N - 1)) + manual.current;
      if (Math.abs(want - pos) < 1e-4 && shown >= 0) return; // settled: no work
      pos += (want - pos) * (1 - Math.exp(-10 * dt));
      if (Math.abs(want - pos) < 1e-4) pos = want;

      const spacing = spacingFor(st.clientWidth);
      slides.current.forEach((el, i) => {
        if (!el) return;
        const d = wrap(i - pos);
        const ad = Math.abs(d);
        const sign = Math.sign(d);
        const opacity = ad <= 1 ? 1 - 0.3 * ad : Math.max(0, 0.7 - (ad - 1) * 0.42);
        // Close neighbours spread out; further ones tuck in behind.
        const x = sign * (Math.min(ad, 1) * spacing + Math.max(0, ad - 1) * spacing * 0.6);
        el.style.transform =
          `translate3d(calc(-50% + ${x.toFixed(1)}px), -50%, ${(-ad * 140).toFixed(0)}px) ` +
          `rotateY(${(-Math.max(-1.5, Math.min(1.5, d)) * 20).toFixed(1)}deg) scale(${(1 - Math.min(ad, 2.5) * 0.1).toFixed(3)})`;
        el.style.opacity = opacity.toFixed(3);
        const level = ad < 0.4 ? 0 : ad < 1.4 ? 1 : 2;
        if (el.dataset.blur !== String(level)) el.dataset.blur = String(level);
        el.style.zIndex = String(100 - Math.round(ad * 10));
        el.style.visibility = opacity > 0.01 ? "visible" : "hidden";
      });
      const a = mod(Math.round(pos));
      if (a !== shown) {
        shown = a;
        setActive(a);
      }
    };

    // Only animate while the section is on screen.
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !running) {
        running = true;
        last = performance.now();
        raf = requestAnimationFrame(tick);
      } else if (!e.isIntersecting && running) {
        running = false;
        cancelAnimationFrame(raf);
      }
    });
    if (section.current) io.observe(section.current);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  const step = (n: number) => {
    manual.current += n;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    drag.current = { x: e.clientX, start: manual.current };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current || !stage.current) return;
    manual.current = drag.current.start - (e.clientX - drag.current.x) / spacingFor(stage.current.clientWidth);
  };
  const onPointerUp = () => {
    if (!drag.current) return;
    drag.current = null;
    manual.current = Math.round(manual.current);
  };

  return (
    <section ref={section} id="app" className={styles.section} aria-roledescription="carousel" aria-label="App screens">
      <div className={styles.sticky}>
        <div className={styles.glow} aria-hidden />
        <header className={styles.head}>
          <div>
            <p className={styles.kicker}>The app</p>
            <h2 className={styles.title}>Everything IDR knows, on one screen.</h2>
          </div>
          <a className={styles.cta} href={APK_URL} download>
            Download APK <span aria-hidden>↓</span>
          </a>
        </header>

        <div
          ref={stage}
          className={styles.stage}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          tabIndex={0}
          aria-label="Drag, or use the arrow keys, to see more screens"
          onKeyDown={(e) => {
            if (e.key === "ArrowRight") step(1);
            if (e.key === "ArrowLeft") step(-1);
          }}
        >
          {SCREENS.map((s, i) => (
            <figure
              key={s.src}
              ref={(el) => {
                slides.current[i] = el;
              }}
              className={styles.slide}
              aria-hidden={i !== active}
            >
              <div className={styles.phone}>
                <Image
                  src={s.src}
                  alt={`IDR app — ${s.title} screen`}
                  width={589}
                  height={1278}
                  sizes="(max-width: 700px) 56vw, 260px"
                  className={styles.screen}
                  draggable={false}
                />
              </div>
            </figure>
          ))}
        </div>

        <div className={styles.foot}>
          <div className={styles.caption} aria-live="polite">
            <span className={styles.count}>
              {pad(active + 1)} <span className={styles.of}>/ {pad(N)}</span>
            </span>
            <span className={styles.captionTitle}>{SCREENS[active].title}</span>
            <span className={styles.captionText}>{SCREENS[active].text}</span>
          </div>
          <div className={styles.controls}>
            <div className={styles.dots} aria-hidden>
              {SCREENS.map((s, i) => (
                <span key={s.src} className={i === active ? styles.dotOn : ""} />
              ))}
            </div>
            <button className="bracket" onClick={() => step(-1)} aria-label="Previous screen">
              prev
            </button>
            <button className="bracket" onClick={() => step(1)} aria-label="Next screen">
              next
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
