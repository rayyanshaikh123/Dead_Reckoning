"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { PHONE } from "@/lib/car";
import { CALLOUTS, calloutEls } from "@/lib/callouts";
import { story } from "@/lib/story";
import { phase, sensorsOpacity, worldOpacity } from "@/lib/timeline";

/** Orange rings spreading from the phone while the sensors are doing the work. */
export function SensorPulse() {
  const rings = useRef<THREE.Mesh[]>([]);
  useFrame((state) => {
    const s = story.s;
    const on = Math.max(phase(s) === "sensors" ? worldOpacity(s) : 0, sensorsOpacity(s));
    rings.current.forEach((r, i) => {
      if (!r) return;
      const t = (state.clock.elapsedTime * 0.7 + i / 3) % 1;
      r.scale.setScalar(0.3 + t * 2.4);
      (r.material as THREE.MeshBasicMaterial).opacity = on * (1 - t) * 0.8;
      r.visible = on > 0.01;
    });
  });
  return (
    <group position={PHONE}>
      {[0, 1, 2].map((i) => (
        <mesh
          key={i}
          ref={(m) => {
            if (m) rings.current[i] = m;
          }}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <ringGeometry args={[0.46, 0.5, 64]} />
          <meshBasicMaterial color="#f06131" transparent opacity={0} depthWrite={false} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
}

/** Gap between the car's right-most marker and the label column (px). */
const COLUMN_GAP = 90;
/** Minimum vertical gap between two labels (px). */
const GAP = 14;
/** Below this width the labels become a legend (see callouts.module.css). */
const LEGEND_BELOW = 720;

/**
 * Invisible points on the car. Each frame, projects them to the screen and
 * moves the DOM callouts (`SensorCallouts`) onto them: a marker on the car, a
 * leader line, and labels aligned in one column to the right.
 */
export function SensorAnchors() {
  const anchors = useRef<THREE.Object3D[]>([]);
  const p = useMemo(() => new THREE.Vector3(), []);
  const pts = useMemo(() => CALLOUTS.map(() => ({ x: 0, y: 0, h: 0 })), []);
  useFrame(({ camera, size }) => {
    const a = sensorsOpacity(story.s);
    if (a < 0.01) {
      calloutEls.forEach((el) => el && (el.style.visibility = "hidden"));
      return;
    }
    anchors.current.forEach((o, i) => {
      if (!o) return;
      o.getWorldPosition(p).project(camera);
      pts[i].x = (p.x * 0.5 + 0.5) * size.width;
      pts[i].y = (-p.y * 0.5 + 0.5) * size.height;
    });
    const legend = size.width < LEGEND_BELOW;
    const column = Math.max(...pts.map((q) => q.x)) + COLUMN_GAP;

    // Labels keep their anchor's height unless they'd collide.
    const order = pts.map((_, i) => i).sort((m, n) => pts[m].y - pts[n].y);
    let floor = -Infinity;
    order.forEach((i) => {
      const el = calloutEls[i];
      if (!el) return;
      const { x, y } = pts[i];
      el.style.visibility = "visible";
      el.style.opacity = String(a);
      el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
      if (legend) return;
      const leader = el.children[1] as HTMLElement;
      const label = el.children[2] as HTMLElement;
      // Reading offsetHeight after writing styles forces a layout; measure once.
      if (!pts[i].h) pts[i].h = label.offsetHeight;
      const top = Math.max(y, floor);
      floor = top + pts[i].h + GAP;
      const dy = top - y;
      leader.style.width = `${Math.max(0, column - x - 6).toFixed(1)}px`;
      label.style.transform = `translate(${(column - x + 8).toFixed(1)}px, ${(dy - 9).toFixed(1)}px)`;
    });
  });
  return (
    <>
      {CALLOUTS.map((c, i) => (
        <object3D
          key={c.title}
          position={c.at}
          ref={(o) => {
            if (o) anchors.current[i] = o;
          }}
        />
      ))}
    </>
  );
}
