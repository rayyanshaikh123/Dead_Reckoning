"use client";

import { Environment, Lightformer, useProgress } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { CAR_MODEL } from "@/lib/carModel";
import { smoothstep } from "@/lib/noise";
import { story } from "@/lib/story";
import {
  cameraPosition,
  cameraTarget,
  carScale,
  carYaw,
  distance,
  ease,
  heroShift,
  introCamera,
  laneOffset,
  speedShare,
  TUNNEL_END,
  TUNNEL_START,
  stageOpacity,
  worldOpacity,
} from "@/lib/timeline";

import { ProceduralCar } from "./Car";
import { ModelCar } from "./ModelCar";
import { SensorAnchors, SensorPulse } from "./Sensors";
import { Backdrop, TunnelGlow, World } from "./World";

/** Film offset (mm) that slides the car right of the hero headline. */
const HERO_FILM_OFFSET = -5;
const MAX_DPR = 2;
/**
 * Most pixels we'll render per frame. The browser upscales the canvas; with
 * built-in multisampling the difference is hard to see, the speed-up isn't.
 */
const PIXEL_BUDGET = 4.6e6; // full Retina sharpness for a 1440×900 window

/** Highest resolution worth rendering for this screen and window size. */
function maxDpr() {
  const byBudget = Math.sqrt(PIXEL_BUDGET / (window.innerWidth * window.innerHeight));
  return Math.max(0.75, Math.min(window.devicePixelRatio, MAX_DPR, byBudget));
}
const INTRO_AIM = new THREE.Vector3(0, 0.55, 0);

/** Hands the scene's `invalidate` to the story loop (renders on demand). */
function Invalidator() {
  const invalidate = useThree((s) => s.invalidate);
  const get = useThree((s) => s.get);
  useEffect(() => {
    story.invalidate = () => invalidate();
    // `?perf`: expose the renderer for benchmarking (scripts/bench.mjs).
    if (window.location.search.includes("perf")) Object.assign(window, { __idr: { get, story } });
    return () => {
      story.invalidate = () => {};
    };
  }, [invalidate, get]);
  return null;
}

/**
 * Keeps frames at the display's own refresh rate (60, 120 Hz…): the shortest
 * back-to-back frame interval seen is the refresh period; when frames run
 * slower than that, render fewer pixels; when there's headroom, more.
 * Idle gaps (renders are on demand) are ignored.
 */
function AdaptiveResolution() {
  const stats = useRef({ last: 0, sum: 0, n: 0, dpr: 0, period: 1000 / 60, good: 0 });
  // `?fixed` holds the resolution (for measuring frame cost).
  const fixed = useMemo(() => window.location.search.includes("fixed"), []);
  useFrame((state) => {
    if (fixed) return;
    const st = stats.current;
    const now = performance.now();
    const dt = now - st.last;
    st.last = now;
    const max = maxDpr();
    if (!st.dpr) st.dpr = max;
    if (dt > 60) return;
    if (dt > 4) st.period = Math.min(st.period, dt);
    st.sum += dt;
    if (++st.n < 30) return;
    const avg = st.sum / st.n;
    st.sum = st.n = 0;
    let next = st.dpr;
    if (avg > st.period * 1.3) {
      next = st.dpr - 0.15;
      st.good = 0;
    } else if (avg < st.period * 1.06 && ++st.good >= 4) {
      next = st.dpr + 0.1;
      st.good = 0;
    }
    const clamped = Math.min(max, Math.max(0.8, next));
    if (Math.abs(clamped - st.dpr) > 0.01) {
      st.dpr = clamped;
      state.setDpr(clamped);
    }
  });
  return null;
}

function CameraRig() {
  const aim = useMemo(() => new THREE.Vector3(), []);
  const pos = useMemo(() => new THREE.Vector3(), []);
  useFrame((state) => {
    const camera = state.camera as THREE.PerspectiveCamera;
    const s = story.s;
    const intro = story.intro;
    aim.set(...cameraTarget(s));
    pos.set(...cameraPosition(s));
    if (intro < 1) {
      pos.set(...introCamera(intro, [pos.x, pos.y, pos.z]));
      aim.lerp(INTRO_AIM, 1 - ease(intro));
    }
    // Shots are framed for landscape; on portrait screens back off along the
    // line of sight so the car still fits across.
    const pullBack = camera.aspect < 1 ? Math.min(1.8, (0.95 / camera.aspect) ** 0.75) : 1;
    pos.sub(aim).multiplyScalar(pullBack).add(aim);
    camera.position.copy(pos);
    camera.lookAt(aim);
    const offset = camera.aspect > 1.15 ? HERO_FILM_OFFSET * heroShift(s) * ease(intro) : 0;
    if (Math.abs(camera.filmOffset - offset) > 1e-3) {
      camera.filmOffset = offset;
      camera.updateProjectionMatrix();
    }
  });
  return null;
}

/** Studio lights that come up as the intro ends (the car starts in the dark). */
function Lights() {
  const amb = useRef<THREE.AmbientLight>(null);
  const key = useRef<THREE.DirectionalLight>(null);
  const sky = useRef<THREE.HemisphereLight>(null);
  useFrame((state) => {
    const lit = ease(story.intro);
    const world = worldOpacity(story.s);
    if (amb.current) amb.current.intensity = 0.02 + 0.26 * lit;
    // The key light casts the car's shadow; underground it all but goes out.
    const d = distance(story.s);
    const underground = smoothstep(TUNNEL_START - 4, TUNNEL_START + 6, d) * (1 - smoothstep(TUNNEL_END - 6, TUNNEL_END + 4, d));
    const k = key.current;
    if (k) {
      k.intensity = 1.1 * lit * (1 - 0.45 * world) * (1 - 0.85 * underground);
      // Only re-render the shadow while the light is actually on (not in the
      // dark intro, barely in the tunnel). Switching castShadow itself would
      // recompile every shader, so the map is just left alone instead.
      // It must exist from the first frame, though: lit surfaces sample it,
      // and with no map yet they fail to draw at all (the dark intro).
      k.shadow.autoUpdate = false;
      if (k.intensity > 0.2 || !k.shadow.map) k.shadow.needsUpdate = true;
    }
    // Moonlit night over the road (always present, so the light count is fixed).
    if (sky.current) sky.current.intensity = 1.6 * world;
    // A faint floor of reflections in the dark intro, so the body's outline
    // shows around the headlights instead of vanishing.
    state.scene.environmentIntensity = 0.14 + (0.86 - 0.25 * world) * lit;
  });
  return (
    <>
      <ambientLight ref={amb} intensity={0} />
      {/* The car never leaves the origin (the world moves), so a tight shadow
          frustum around it gives a sharp shadow from a small map. */}
      <directionalLight
        ref={key}
        position={[6, 10, 4]}
        intensity={0}
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={-4}
        shadow-camera-right={4}
        shadow-camera-top={4}
        shadow-camera-bottom={-4}
        shadow-camera-near={2}
        shadow-camera-far={24}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
        shadow-radius={3}
      />
      <hemisphereLight ref={sky} args={["#5a6f94", "#0d100e", 0]} />
    </>
  );
}

/** Soft radial texture for the stage floor and the blob shadow. */
function useRadialTexture(inner: string, outer: string) {
  return useMemo(() => {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d")!;
    const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
    grad.addColorStop(0, inner);
    grad.addColorStop(1, outer);
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [inner, outer]);
}

const RINGS = [
  { r: 4.2, w: 0.03, color: "#f06131", alpha: 0.9 },
  { r: 5.6, w: 0.015, color: "#5c5d5d", alpha: 0.5 },
];

/** Showcase floor for the intro, the hero and the top view. */
function Stage() {
  const fade = useRadialTexture("#ffffff", "#000000");
  const floor = useRef<THREE.MeshStandardMaterial>(null);
  const rings = useRef<THREE.MeshBasicMaterial[]>([]);
  useFrame(() => {
    const a = stageOpacity(story.s);
    const lit = ease(story.intro);
    // The floor is lit, so in the dark intro it only shows the headlight pool.
    if (floor.current) floor.current.opacity = a;
    rings.current.forEach((m, i) => m && (m.opacity = a * lit * RINGS[i].alpha));
  });
  return (
    <group position={[0, 0.001, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[16, 64]} />
        <meshStandardMaterial ref={floor} color="#2a2b2b" roughness={0.55} metalness={0.2} alphaMap={fade} transparent depthWrite={false} />
      </mesh>
      {RINGS.map((ring, i) => (
        <mesh key={ring.r} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
          <ringGeometry args={[ring.r, ring.r + ring.w, 128]} />
          <meshBasicMaterial
            ref={(m) => {
              if (m) rings.current[i] = m;
            }}
            color={ring.color}
            transparent
            opacity={0}
            toneMapped={false}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}

type ModelState = "none" | "glb";

function Vehicle({ model }: { model: ModelState }) {
  const group = useRef<THREE.Group>(null);
  const body = useRef<THREE.Group>(null);
  const dist = useRef(0);
  const motion = useRef({ d: 0, v: 0, acc: 0 });
  // Soft contact darkening under the car (the key light adds the real shadow).
  const shadow = useRadialTexture("rgba(0,0,0,0.6)", "rgba(0,0,0,0)");
  useFrame((_, delta) => {
    const s = story.s;
    const d = distance(s);
    dist.current = d;
    const g = group.current;
    const b = body.current;
    if (!g || !b) return;

    // Sideways lane wander, and the car steering along it.
    const x = laneOffset(d);
    const slope = (laneOffset(d + 0.5) - laneOffset(d - 0.5)) / 1; // metres sideways per metre
    g.position.x = x;
    g.rotation.y = carYaw(s) - Math.atan(slope);
    g.scale.setScalar(carScale(s));

    // Pitch from real acceleration: squat pulling away, dive when braking.
    const dt = Math.min(Math.max(delta, 1 / 240), 0.05);
    const m = motion.current;
    const v = (d - m.d) / dt;
    m.acc += ((v - m.v) / dt - m.acc) * (1 - Math.exp(-6 * dt));
    m.d = d;
    m.v = v;
    const share = speedShare(s);
    b.rotation.z = Math.max(-0.02, Math.min(0.02, m.acc * 0.00004));
    // Lean into the curve of the lane wander, and road texture through the springs.
    const curve = laneOffset(d + 1) - 2 * x + laneOffset(d - 1);
    b.rotation.x = Math.max(-0.012, Math.min(0.012, curve * 2.5 * share));
    b.position.y = (Math.sin(d * 1.7) * 0.004 + Math.sin(d * 4.3 + 0.7) * 0.0025) * share;
  });
  return (
    <group ref={group}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.004, 0]} scale={[5.6, 2.6, 1]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial map={shadow} transparent depthWrite={false} />
      </mesh>
      <group ref={body}>
        {model === "glb" ? <ModelCar distanceRef={dist} /> : <ProceduralCar distanceRef={dist} />}
        <SensorPulse />
        <SensorAnchors />
      </group>
    </group>
  );
}

/** Reports model download progress to the loading screen. */
function ProgressBridge() {
  const { progress } = useProgress();
  useEffect(() => {
    story.loaded = progress / 100;
  }, [progress]);
  return null;
}

/**
 * While the loading screen still covers the page, compiles every shader the
 * story will need — the scenery in both its fading and its opaque form — so
 * nothing stalls mid-scroll. Then signals that the page can be shown.
 */
function ReadySignal({ onReady }: { onReady: () => void }) {
  const frames = useRef(0);
  useFrame((state) => {
    if (frames.current < 0) return;
    const f = ++frames.current;
    if (f === 1) story.warmup = 1;
    else if (f === 2) {
      state.gl.compile(state.scene, state.camera);
      story.warmup = 2;
    } else if (f === 3) {
      state.gl.compile(state.scene, state.camera);
      story.warmup = 0;
    } else if (f >= 5) {
      frames.current = -1;
      onReady();
      return;
    }
    state.invalidate();
  });
  return null;
}

export default function Scene({ onReady }: { onReady: () => void }) {
  const dpr = useMemo(() => maxDpr(), []);
  // The provided car model if there is one (found at build time).
  const model: ModelState = CAR_MODEL.url ? "glb" : "none";

  return (
    <Canvas
      frameloop="demand"
      dpr={dpr}
      camera={{ fov: 38, near: 0.1, far: 800, position: [4, 0.85, -3.5] }}
      // No post-processing: glows are sprites (see glow.ts), anti-aliasing is
      // the GPU's own multisampling — nearly free on tile-based (Apple) GPUs,
      // where every full-screen pass is expensive.
      gl={{ antialias: true, stencil: false, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.shadowMap.enabled = true;
        gl.shadowMap.type = THREE.PCFShadowMap;
        gl.setClearColor("#0f1010");
      }}
    >
      <fog attach="fog" args={["#161c26", 40, 300]} />
      <Invalidator />
      <AdaptiveResolution />
      <ProgressBridge />
      <CameraRig />
      <Lights />
      {/* Studio reflections built from light panels (no HDR download). */}
      <Environment resolution={512} frames={1}>
        <Lightformer form="rect" intensity={2.4} color="#ffffff" scale={[14, 5, 1]} position={[0, 8, 0]} rotation-x={Math.PI / 2} />
        <Lightformer form="rect" intensity={1.8} color="#ffffff" scale={[16, 0.8, 1]} position={[0, 2.2, 7]} rotation-y={Math.PI} />
        <Lightformer form="rect" intensity={1.8} color="#ffffff" scale={[16, 0.8, 1]} position={[0, 2.2, -7]} />
        <Lightformer form="rect" intensity={1.2} color="#ffffff" scale={[8, 1.2, 1]} position={[-8, 2, 0]} rotation-y={Math.PI / 2} />
        <Lightformer form="rect" intensity={1.8} color="#f06131" scale={[8, 0.5, 1]} position={[8, 1, 0]} rotation-y={-Math.PI / 2} />
        <Lightformer form="rect" intensity={0.25} color="#8c8d8d" scale={[20, 20, 1]} position={[0, -2, 0]} rotation-x={-Math.PI / 2} />
      </Environment>

      <Backdrop />
      <Stage />
      <World />
      <TunnelGlow />
      <Suspense fallback={null}>
        <Vehicle model={model} />
        <ReadySignal onReady={onReady} />
      </Suspense>

    </Canvas>
  );
}
