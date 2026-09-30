import { CALLOUTS, calloutEls } from "@/lib/callouts";

import styles from "./callouts.module.css";

const index = (i: number) => String(i + 1).padStart(2, "0");

/**
 * Sensor labels for the top view: plain DOM over the canvas. The scene moves
 * each marker onto its point on the car (see `SensorAnchors`). On phones the
 * labels become a numbered legend under the car.
 */
export function SensorCallouts() {
  return (
    <div className={styles.layer} aria-hidden>
      {CALLOUTS.map((c, i) => (
        <div
          key={c.title}
          ref={(el) => {
            calloutEls[i] = el;
          }}
          className={styles.callout}
        >
          <span className={styles.marker}>
            <span className={styles.markerNum}>{index(i)}</span>
          </span>
          <span className={styles.leader} />
          <div className={styles.label}>
            <p className={styles.title}>
              <span className={styles.num}>{index(i)}</span>
              {c.title}
            </p>
            <p className={styles.text}>{c.text}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Phone layout: the same labels as a list (shown by CSS below 720 px). */
export function SensorLegend({ refCallback }: { refCallback: (el: HTMLElement | null) => void }) {
  return (
    <ol ref={refCallback} className={styles.legend} aria-label="Sensors IDR uses">
      {CALLOUTS.map((c, i) => (
        <li key={c.title}>
          <span className={styles.num}>{index(i)}</span>
          <span>
            <span className={styles.title}>{c.title}</span>
            <span className={styles.text}>{c.short}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}
