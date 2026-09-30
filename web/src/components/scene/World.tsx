"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

import { fbm, rng, smoothstep } from "@/lib/noise";
import { story } from "@/lib/story";
import {
  EXIT_ERROR_M,
  TUNNEL_END as TE,
  TUNNEL_START as TS,
  distance,
  exitMarkersOpacity,
  worldOpacity,
} from "@/lib/timeline";

import { glowPointsMaterial, glowSpriteMaterial } from "./glow";

// The drive: a two-lane road running towards -z through a mountain, with the
// tunnel between TS and TE (metres along the road). The car stays at the
// origin; the world slides towards +z by the distance driven.
//
// Local coordinates: road centre at x = 0, the car's lane centre at LANE_X.
// The whole world group sits at x = -LANE_X so the car (world x = 0) is in
// its lane.

const LANE_X = 2;
const ROAD_W = 8;
const TUNNEL_R = 6;
const ROAD_FROM = 40; // road starts behind the car
const ROAD_TO = 560;
const PORTAL_HALF = 12; // half-width of the portal face
const BLOCK_H = 11; // height of the portal face

// ---- terrain ----------------------------------------------------------------

/** 0 away from the mountain, 1 over the middle of the tunnel. */
function mountain(d: number) {
  return Math.min(smoothstep(TS - 50, TS + 40, d), 1 - smoothstep(TE - 40, TE + 50, d));
}

function natural(x: number, d: number) {
  const ax = Math.abs(x);
  const hills = 2 + 12 * fbm(x * 0.017 + 3.1, d * 0.017) * smoothstep(8, 60, ax);
  const ridge = mountain(d) * (52 + 40 * fbm(x * 0.028, d * 0.028 + 7)) * (1 - 0.55 * smoothstep(50, 190, ax));
  return hills + ridge;
}

/** Half-width of the flat road cut; a wide forecourt in front of each portal. */
function cutHalf(d: number) {
  const fore = Math.max(smoothstep(TS - 44, TS - 24, d) * (1 - smoothstep(TS, TS + 1, d)), smoothstep(TE + 44, TE + 24, d) * smoothstep(TE - 1, TE, d));
  return 7 + (PORTAL_HALF + 0.5 - 7) * fore;
}

/** Ground height at local (x, d). The portal faces cover the tunnel mouths. */
export function terrainHeight(x: number, d: number) {
  const ax = Math.abs(x);
  const n = natural(x, d);
  if (d >= TS && d <= TE) {
    if (ax > PORTAL_HALF + 0.01) return n;
    const inward = smoothstep(TS, TS + 16, d) * (1 - smoothstep(TE - 16, TE, d));
    return BLOCK_H - 0.3 + Math.max(0, n - BLOCK_H) * inward;
  }
  const c = cutHalf(d);
  return n * smoothstep(c, c + 9, ax) - 0.06;
}

function useTerrain() {
  return useMemo(() => {
    const DX = 2;
    const DZ = 4;
    const xs: number[] = [];
    for (let x = -200; x <= 200; x += DX) xs.push(x);
    // Rows land exactly on the portals (TS, TE are multiples of DZ apart).
    const ds: number[] = [];
    for (let k = -26; k <= 118; k++) ds.push(TS + k * DZ);
    const nx = xs.length;
    const pos = new Float32Array(nx * ds.length * 3);
    const col = new Float32Array(nx * ds.length * 3);
    const grass = new THREE.Color("#24332a");
    const dry = new THREE.Color("#343a31");
    const rock = new THREE.Color("#3b3e40");
    const c = new THREE.Color();
    ds.forEach((d, j) => {
      xs.forEach((x, i) => {
        const h = terrainHeight(x, d);
        const k = (j * nx + i) * 3;
        pos[k] = x;
        pos[k + 1] = h;
        pos[k + 2] = -d;
        // steeper → rock; higher → drier
        const slope = Math.abs(terrainHeight(x + DX, d) - h) / DX + Math.abs(terrainHeight(x, d + DZ) - h) / DZ;
        c.copy(grass).lerp(dry, smoothstep(10, 45, h) * 0.8).lerp(rock, smoothstep(0.5, 1.4, slope));
        const v = 0.85 + 0.3 * fbm(x * 0.1, d * 0.1, 2);
        col[k] = c.r * v;
        col[k + 1] = c.g * v;
        col[k + 2] = c.b * v;
      });
    });
    const index: number[] = [];
    for (let j = 0; j < ds.length - 1; j++) {
      const d0 = ds[j];
      const d1 = ds[j + 1];
      // Leave the strip in front of each portal open (the face fills it).
      const portalStrip = (d0 === TS - DZ && d1 === TS) || (d0 === TE && d1 === TE + DZ);
      for (let i = 0; i < nx - 1; i++) {
        if (portalStrip && Math.abs(xs[i]) < PORTAL_HALF && Math.abs(xs[i + 1]) <= PORTAL_HALF) continue;
        const a = j * nx + i;
        const b = a + 1;
        const cc = a + nx;
        const dd = cc + 1;
        index.push(a, b, cc, b, dd, cc); // counter-clockwise from above: faces up
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(index);
    g.computeVertexNormals();
    return g;
  }, []);
}

// ---- textures (drawn once, tiny) ------------------------------------------

function canvasTexture(w: number, h: number, draw: (g: CanvasRenderingContext2D) => void, repeat: [number, number]) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d")!);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(...repeat);
  t.anisotropy = 16; // clamped to what the GPU supports; keeps the road sharp to the horizon
  return t;
}

function speckle(g: CanvasRenderingContext2D, w: number, h: number, n: number, alpha: number) {
  const r = rng(w * 31 + h);
  for (let i = 0; i < n; i++) {
    const v = Math.floor(r() * 255);
    g.fillStyle = `rgba(${v},${v},${v},${alpha * r()})`;
    g.fillRect(r() * w, r() * h, 1 + r() * 1.5, 1 + r() * 1.5);
  }
}

function useTextures() {
  return useMemo(() => {
    const asphalt = canvasTexture(
      256,
      256,
      (g) => {
        g.fillStyle = "#1b1c1d";
        g.fillRect(0, 0, 256, 256);
        speckle(g, 256, 256, 5000, 0.18);
      },
      [2, (ROAD_TO + ROAD_FROM) / 8],
    );
    // Board-formed concrete: horizontal pour lines, a little staining.
    const concrete = canvasTexture(
      256,
      256,
      (g) => {
        g.fillStyle = "#2e3031";
        g.fillRect(0, 0, 256, 256);
        speckle(g, 256, 256, 3000, 0.1);
        const r = rng(5);
        for (let y = 0; y < 256; y += 16) {
          g.fillStyle = `rgba(0,0,0,${0.12 + r() * 0.12})`;
          g.fillRect(0, y, 256, 1);
          g.fillStyle = `rgba(255,255,255,${0.02 + r() * 0.03})`;
          g.fillRect(0, y + 1, 256, 7);
        }
        const stain = g.createLinearGradient(0, 0, 0, 256);
        stain.addColorStop(0, "rgba(0,0,0,0)");
        stain.addColorStop(1, "rgba(0,0,0,0.25)");
        g.fillStyle = stain;
        g.fillRect(0, 0, 256, 256);
      },
      [0.12, 0.12],
    );
    // Tunnel lining: tiled lower walls at both edges (u≈0, u≈1), dark vault.
    const lining = canvasTexture(
      512,
      64,
      (g) => {
        g.fillStyle = "#1c1e1e";
        g.fillRect(0, 0, 512, 64);
        for (const [x0, x1] of [
          [0, 120],
          [392, 512],
        ]) {
          g.fillStyle = "#5d6061";
          g.fillRect(x0, 0, x1 - x0, 64);
          g.strokeStyle = "rgba(0,0,0,0.25)";
          for (let x = x0; x <= x1; x += 15) {
            g.beginPath();
            g.moveTo(x, 0);
            g.lineTo(x, 64);
            g.stroke();
          }
          for (let y = 0; y <= 64; y += 16) {
            g.beginPath();
            g.moveTo(x0, y);
            g.lineTo(x1, y);
            g.stroke();
          }
        }
        // a thin orange line where the tiles end
        g.fillStyle = "#f06131";
        g.fillRect(120, 0, 3, 64);
        g.fillRect(389, 0, 3, 64);
      },
      [1, (TE - TS) / 4],
    );
    return { asphalt, concrete, lining };
  }, []);
}

// ---- fade -------------------------------------------------------------------

type Fadeable = THREE.Material & { opacity: number };

/**
 * Fades every material in the group with `opacity(s)`. Materials are only
 * transparent while fading (opaque otherwise — much cheaper to draw).
 */
function useFade(ref: React.RefObject<THREE.Object3D | null>, opacity: (s: number) => number) {
  const mats = useRef<{ m: Fadeable; base: number; transparent: boolean }[] | null>(null);
  const last = useRef(-1);
  useFrame(() => {
    const g = ref.current;
    if (!g) return;
    if (!mats.current) {
      mats.current = [];
      const seen = new Set<THREE.Material>();
      g.traverse((o) => {
        if (o.userData.ownFade) return;
        const list = (o as THREE.Mesh).material;
        for (const m of Array.isArray(list) ? list : list ? [list] : []) {
          if (seen.has(m)) continue;
          seen.add(m);
          mats.current!.push({ m: m as Fadeable, base: (m as Fadeable).opacity, transparent: m.transparent });
        }
      });
    }
    const a = story.warmup === 1 ? 0.5 : story.warmup === 2 ? 1 : opacity(story.s);
    if (a === last.current) return;
    last.current = a;
    g.visible = a > 0.002;
    const fading = a < 0.999;
    for (const f of mats.current) {
      const want = f.transparent || fading;
      if (f.m.transparent !== want) {
        f.m.transparent = want;
        f.m.needsUpdate = true;
      }
      f.m.opacity = f.base * a;
    }
  });
}

// ---- pieces -------------------------------------------------------------------

function Instanced({
  geometry,
  material,
  matrices,
  colors,
}: {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  matrices: THREE.Matrix4[];
  colors?: THREE.Color[];
}) {
  return (
    <instancedMesh
      args={[geometry, material, matrices.length]}
      frustumCulled={false}
      ref={(m) => {
        if (!m) return;
        matrices.forEach((mx, i) => m.setMatrixAt(i, mx));
        colors?.forEach((c, i) => m.setColorAt(i, c));
        m.instanceMatrix.needsUpdate = true;
        if (m.instanceColor) m.instanceColor.needsUpdate = true;
      }}
    />
  );
}

const M = new THREE.Matrix4();
const Q = new THREE.Quaternion();
const S = new THREE.Vector3();
const P = new THREE.Vector3();
const E = new THREE.Euler();
function mat(x: number, y: number, z: number, sx = 1, sy = 1, sz = 1, ry = 0) {
  return M.compose(P.set(x, y, z), Q.setFromEuler(E.set(0, ry, 0)), S.set(sx, sy, sz)).clone();
}

/** One glow per position, all in a single draw call. */
function GlowPoints({ at, color, size, opacity = 1 }: { at: THREE.Vector3[]; color: string; size: number; opacity?: number }) {
  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(at), [at]);
  const material = useMemo(() => glowPointsMaterial(color, size, opacity), [color, size, opacity]);
  return <points geometry={geometry} material={material} frustumCulled={false} />;
}

/** Stretches of road outside the tunnel (for rails, lamps). */
const OPEN_ROAD: [number, number][] = [
  [-ROAD_FROM, TS - 3],
  [TE + 3, ROAD_TO - 60],
];

function Road({ asphalt }: { asphalt: THREE.Texture }) {
  const len = ROAD_TO + ROAD_FROM;
  const mid = ROAD_FROM - len / 2;
  const dashes = useMemo(() => {
    const out: THREE.Matrix4[] = [];
    for (let z = ROAD_FROM; z > -ROAD_TO; z -= 9) out.push(mat(0, 0.012, z));
    return out;
  }, []);
  const dashGeo = useMemo(() => new THREE.PlaneGeometry(0.15, 3).rotateX(-Math.PI / 2), []);
  const lineMat = useMemo(() => new THREE.MeshBasicMaterial({ color: "#bdbdbd" }), []);
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, mid]} receiveShadow>
        <planeGeometry args={[ROAD_W + 1, len]} />
        <meshStandardMaterial map={asphalt} roughness={0.92} metalness={0} />
      </mesh>
      {[-1, 1].map((side) => (
        <mesh key={side} rotation={[-Math.PI / 2, 0, 0]} position={[side * (ROAD_W / 2 - 0.25), 0.011, mid]} material={lineMat}>
          <planeGeometry args={[0.14, len]} />
        </mesh>
      ))}
      <Instanced geometry={dashGeo} material={lineMat} matrices={dashes} />
      {/* ground under everything (shows in the forecourts) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.08, mid]}>
        <planeGeometry args={[420, len + 200]} />
        <meshLambertMaterial color="#141816" />
      </mesh>
    </group>
  );
}

function Rails() {
  const { posts, rails } = useMemo(() => {
    const posts: THREE.Matrix4[] = [];
    const rails: THREE.Matrix4[] = [];
    for (const [a, b] of OPEN_ROAD) {
      for (let d = a; d < b; d += 4) {
        for (const side of [-1, 1]) {
          const x = side * (ROAD_W / 2 + 0.9);
          posts.push(mat(x, 0.4, -d));
          rails.push(mat(x, 0.68, -(d + 2)));
        }
      }
    }
    return { posts, rails };
  }, []);
  const postGeo = useMemo(() => new THREE.BoxGeometry(0.12, 0.8, 0.12), []);
  const railGeo = useMemo(() => new THREE.BoxGeometry(0.06, 0.26, 4.02), []);
  const metal = useMemo(() => new THREE.MeshStandardMaterial({ color: "#8a8d8f", metalness: 0.7, roughness: 0.4 }), []);
  return (
    <>
      <Instanced geometry={postGeo} material={metal} matrices={posts} />
      <Instanced geometry={railGeo} material={metal} matrices={rails} />
    </>
  );
}

function StreetLamps() {
  const { poles, heads } = useMemo(() => {
    const poles: THREE.Matrix4[] = [];
    const heads: THREE.Matrix4[] = [];
    let k = 0;
    for (const [a, b] of OPEN_ROAD) {
      for (let d = a + 10; d < b; d += 34, k++) {
        const side = k % 2 ? 1 : -1;
        const x = side * (ROAD_W / 2 + 1.6);
        poles.push(mat(x, 4, -d));
        heads.push(mat(x - side * 1.1, 7.9, -d));
      }
    }
    return { poles, heads };
  }, []);
  const poleGeo = useMemo(() => new THREE.CylinderGeometry(0.07, 0.11, 8, 8), []);
  const headGeo = useMemo(() => new THREE.BoxGeometry(1.4, 0.12, 0.35), []);
  const poleMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#3a3c3d", metalness: 0.5, roughness: 0.5 }), []);
  const headMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: new THREE.Color("#ffd29a").multiplyScalar(2.6), toneMapped: false }),
    [],
  );
  const glowAt = useMemo(() => heads.map((m) => new THREE.Vector3().setFromMatrixPosition(m).add(new THREE.Vector3(0, -0.15, 0))), [heads]);
  return (
    <>
      <Instanced geometry={poleGeo} material={poleMat} matrices={poles} />
      <Instanced geometry={headGeo} material={headMat} matrices={heads} />
      <GlowPoints at={glowAt} color="#ffcf8f" size={5} opacity={0.8} />
    </>
  );
}

/** Low-poly pine: trunk + three cones, coloured per vertex. */
function pineGeometry() {
  const paint = (g: THREE.BufferGeometry, hex: string) => {
    const c = new THREE.Color(hex);
    const n = g.attributes.position.count;
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) arr.set([c.r, c.g, c.b], i * 3);
    g.setAttribute("color", new THREE.BufferAttribute(arr, 3));
    return g.toNonIndexed();
  };
  const parts = [
    paint(new THREE.CylinderGeometry(0.1, 0.17, 1.6, 6).translate(0, 0.8, 0), "#2b221b"),
    paint(new THREE.ConeGeometry(1.5, 2.4, 11).translate(0, 2.0, 0), "#223529"),
    paint(new THREE.ConeGeometry(1.22, 2.2, 11).translate(0, 3.0, 0), "#263b2d"),
    paint(new THREE.ConeGeometry(0.92, 2.0, 11).translate(0, 4.0, 0), "#2b4332"),
    paint(new THREE.ConeGeometry(0.55, 1.7, 11).translate(0, 5.0, 0), "#314b38"),
  ].map((g) => {
    g.deleteAttribute("uv");
    return g;
  });
  return mergeGeometries(parts)!;
}

function Trees() {
  const geometry = useMemo(() => pineGeometry(), []);
  const material = useMemo(() => new THREE.MeshLambertMaterial({ vertexColors: true }), []);
  const { matrices, colors } = useMemo(() => {
    const r = rng(7);
    const matrices: THREE.Matrix4[] = [];
    const colors: THREE.Color[] = [];
    for (let tries = 0; matrices.length < 1400 && tries < 12000; tries++) {
      const side = r() < 0.5 ? -1 : 1;
      // denser near the road, thinning out
      const x = side * (10 + Math.pow(r(), 1.7) * 170);
      const d = -ROAD_FROM + r() * (ROAD_TO + ROAD_FROM - 20);
      const ax = Math.abs(x);
      const nearPortal = (Math.abs(d - TS) < 26 || Math.abs(d - TE) < 26) && ax < PORTAL_HALF + 6;
      if (nearPortal) continue;
      if (ax < cutHalf(d) + 3 && (d < TS || d > TE)) continue;
      // clearings, so it isn't a uniform carpet
      if (fbm(x * 0.03, d * 0.03, 2) < 0.38) continue;
      const y = terrainHeight(x, d) - 0.15;
      const s = 1.1 + r() * 1.6;
      matrices.push(mat(x, y, -d, s, s * (0.9 + r() * 0.4), s, r() * 6.3));
      const v = 0.75 + r() * 0.45;
      colors.push(new THREE.Color(v, v * (0.95 + r() * 0.1), v));
    }
    return { matrices, colors };
  }, []);
  return <Instanced geometry={geometry} material={material} matrices={matrices} colors={colors} />;
}

function Tunnel({ concrete, lining }: { concrete: THREE.Texture; lining: THREE.Texture }) {
  const len = TE - TS;
  const zMid = -(TS + len / 2);

  // The mountain's concrete front: a slab with the arch cut out, extruded
  // through the whole mountain (its end caps are the two portal faces).
  const block = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-PORTAL_HALF, 0);
    s.lineTo(-TUNNEL_R - 0.4, 0);
    s.absarc(0, 0, TUNNEL_R + 0.4, Math.PI, 0, true);
    s.lineTo(PORTAL_HALF, 0);
    s.lineTo(PORTAL_HALF, BLOCK_H);
    s.lineTo(-PORTAL_HALF, BLOCK_H);
    s.lineTo(-PORTAL_HALF, 0);
    const g = new THREE.ExtrudeGeometry(s, { depth: len, bevelEnabled: false, curveSegments: 32 });
    return g;
  }, [len]);
  const shell = useMemo(() => {
    const g = new THREE.CylinderGeometry(TUNNEL_R, TUNNEL_R, len, 40, 1, true, -Math.PI / 2, Math.PI);
    g.rotateX(-Math.PI / 2);
    return g;
  }, [len]);

  const { strips, niches, ribs } = useMemo(() => {
    const strips: THREE.Matrix4[] = [];
    const niches: THREE.Matrix4[] = [];
    const ribs: THREE.Matrix4[] = [];
    const yTop = Math.sqrt(TUNNEL_R ** 2 - 2.1 ** 2) - 0.12;
    for (let d = TS + 3; d < TE - 2; d += 6) for (const x of [-2.1, 2.1]) strips.push(mat(x, yTop, -d));
    for (let d = TS + 15; d < TE; d += 30) for (const x of [-TUNNEL_R + 0.12, TUNNEL_R - 0.12]) niches.push(mat(x, 1.1, -d));
    for (let d = TS + 12; d < TE; d += 12) ribs.push(mat(0, 0, -d));
    return { strips, niches, ribs };
  }, []);
  const stripGeo = useMemo(() => new THREE.BoxGeometry(0.22, 0.05, 3.4), []);
  const stripGlow = useMemo(() => strips.map((m) => new THREE.Vector3().setFromMatrixPosition(m).add(new THREE.Vector3(0, -0.25, 0))), [strips]);
  const stripMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: new THREE.Color("#fff3dc").multiplyScalar(2.4), toneMapped: false }),
    [],
  );
  const nicheGeo = useMemo(() => new THREE.BoxGeometry(0.05, 0.5, 0.9), []);
  const nicheMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: new THREE.Color("#f06131").multiplyScalar(2.2), toneMapped: false }),
    [],
  );
  const ribGeo = useMemo(() => new THREE.TorusGeometry(TUNNEL_R - 0.04, 0.05, 5, 40, Math.PI), []);
  const ribMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#2f3131", roughness: 0.8 }), []);

  return (
    <group>
      <mesh geometry={block} position={[0, 0, -TE]}>
        <meshStandardMaterial attach="material-0" map={concrete} roughness={0.85} />
        <meshLambertMaterial attach="material-1" color="#1b1d1d" />
      </mesh>
      <mesh geometry={shell} position={[0, 0, zMid]}>
        <meshStandardMaterial map={lining} side={THREE.BackSide} roughness={0.7} />
      </mesh>
      {/* raised walkways */}
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * (ROAD_W / 2 + 0.95), 0.12, zMid]}>
          <boxGeometry args={[1.9, 0.24, len]} />
          <meshStandardMaterial map={concrete} roughness={0.9} />
        </mesh>
      ))}
      {/* portal frames with the IDR-orange band */}
      {[TS, TE].map((d, k) => (
        <group key={d} position={[0, 0, -d + (k === 0 ? 0.02 : -0.02)]}>
          <mesh>
            <torusGeometry args={[TUNNEL_R + 0.75, 0.4, 10, 48, Math.PI]} />
            <meshStandardMaterial color="#4a4c4d" roughness={0.7} />
          </mesh>
          <mesh position={[0, 0, k === 0 ? 0.42 : -0.42]}>
            <torusGeometry args={[TUNNEL_R + 0.75, 0.05, 6, 64, Math.PI]} />
            <meshBasicMaterial color={new THREE.Color("#f06131").multiplyScalar(2.5)} toneMapped={false} />
          </mesh>
        </group>
      ))}
      <Instanced geometry={stripGeo} material={stripMat} matrices={strips} />
      <GlowPoints at={stripGlow} color="#fff0d8" size={2.6} opacity={0.55} />
      <Instanced geometry={nicheGeo} material={nicheMat} matrices={niches} />
      <Instanced geometry={ribGeo} material={ribMat} matrices={ribs} />
    </group>
  );
}

/**
 * One light that follows the nearest ceiling strip: light sweeping over the
 * car. Lives outside the fading world (a light that appears or disappears
 * makes every shader recompile), so it dims instead.
 */
export function TunnelGlow() {
  const ref = useRef<THREE.PointLight>(null);
  useFrame(() => {
    const l = ref.current;
    if (!l) return;
    const d = distance(story.s);
    const inside = smoothstep(TS - 6, TS + 2, d) * (1 - smoothstep(TE - 2, TE + 6, d));
    l.intensity = 40 * inside * worldOpacity(story.s);
    const k = Math.round((d - TS - 3) / 6);
    l.position.set(0, 5, d - (TS + 3 + k * 6));
  });
  return <pointLight ref={ref} color="#fff1dc" intensity={0} distance={16} decay={2} />;
}

/** Orange line on the road: the stretch IDR tracked on sensors alone. */
function SensorTrail() {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const d = distance(story.s);
    const to = Math.min(d, TE);
    const len = Math.max(0.001, to - TS);
    m.visible = d > TS;
    m.scale.set(1, len, 1);
    m.position.z = -(TS + len / 2);
  });
  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]} position={[LANE_X, 0.02, 0]}>
      <planeGeometry args={[0.22, 1]} />
      <meshBasicMaterial color={new THREE.Color("#f06131").multiplyScalar(2.2)} toneMapped={false} />
    </mesh>
  );
}

/** When GPS returns: IDR's estimate (orange) vs the GPS fix (blue), 12 m apart. */
function ExitMarkers() {
  const group = useRef<THREE.Group>(null);
  const mats = useRef<THREE.MeshBasicMaterial[]>([]);
  useFrame(() => {
    const g = group.current;
    if (!g) return;
    const a = exitMarkersOpacity(story.s) * worldOpacity(story.s);
    g.visible = a > 0.01;
    mats.current.forEach((m) => m && (m.opacity = a));
  });
  const pin = (color: string, z: number, i: number) => (
    <group position={[LANE_X, 0, -(TE + z)]}>
      <mesh position={[0, 1.8, 0]} userData={{ ownFade: true }}>
        <sphereGeometry args={[0.26, 20, 12]} />
        <meshBasicMaterial
          ref={(m) => {
            if (m) mats.current[i * 2] = m;
          }}
          color={new THREE.Color(color).multiplyScalar(2)}
          toneMapped={false}
          transparent
        />
      </mesh>
      <mesh position={[0, 0.9, 0]} userData={{ ownFade: true }}>
        <cylinderGeometry args={[0.025, 0.025, 1.8, 6]} />
        <meshBasicMaterial
          ref={(m) => {
            if (m) mats.current[i * 2 + 1] = m;
          }}
          color={color}
          transparent
        />
      </mesh>
    </group>
  );
  return (
    <group ref={group}>
      {pin("#f06131", 0, 0)}
      {pin("#4285f4", EXIT_ERROR_M, 1)}
    </group>
  );
}

export function World() {
  const group = useRef<THREE.Group>(null);
  const terrain = useTerrain();
  const tex = useTextures();
  useFade(group, worldOpacity);
  useFrame(() => {
    if (group.current) group.current.position.z = distance(story.s);
  });
  return (
    <group ref={group} position={[-LANE_X, 0, 0]}>
      <mesh geometry={terrain}>
        <meshLambertMaterial vertexColors />
      </mesh>
      <Road asphalt={tex.asphalt} />
      <Rails />
      <StreetLamps />
      <Trees />
      <Tunnel concrete={tex.concrete} lining={tex.lining} />
      <SensorTrail />
      <ExitMarkers />
    </group>
  );
}

// ---- backdrop: sky, stars, moon, far ridges (doesn't slide) ----------------------

function ridgeGeometry(radius: number, base: number, amp: number, seed: number, bottom: string, top: string) {
  const seg = 240;
  const pos: number[] = [];
  const col: number[] = [];
  const cb = new THREE.Color(bottom);
  const ct = new THREE.Color(top);
  for (let i = 0; i <= seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    const h = base + amp * Math.pow(fbm(Math.cos(a) * 2.2 + seed, Math.sin(a) * 2.2 + seed, 5), 1.6);
    const x = Math.cos(a) * radius;
    const z = Math.sin(a) * radius;
    pos.push(x, -30, z, x, h, z);
    col.push(cb.r, cb.g, cb.b, ct.r, ct.g, ct.b);
  }
  const index: number[] = [];
  for (let i = 0; i < seg; i++) {
    const a = i * 2;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(index);
  return g;
}

export function Backdrop() {
  const group = useRef<THREE.Group>(null);
  useFade(group, worldOpacity);

  const sky = useMemo(() => {
    const g = new THREE.SphereGeometry(640, 32, 16);
    const zenith = new THREE.Color("#07090c");
    const horizon = new THREE.Color("#1a2130");
    const below = new THREE.Color("#0c0e11");
    const c = new THREE.Color();
    const p = g.attributes.position;
    const col = new Float32Array(p.count * 3);
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i) / 640;
      if (y >= 0) c.copy(horizon).lerp(zenith, Math.pow(y, 0.55));
      else c.copy(horizon).lerp(below, Math.min(1, -y * 6));
      col.set([c.r, c.g, c.b], i * 3);
    }
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, []);

  const stars = useMemo(() => {
    const r = rng(3);
    const pos: number[] = [];
    for (let i = 0; i < 900; i++) {
      const a = r() * Math.PI * 2;
      const el = Math.asin(0.12 + r() * 0.88);
      pos.push(Math.cos(a) * Math.cos(el) * 600, Math.sin(el) * 600, Math.sin(a) * Math.cos(el) * 600);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, []);

  const far = useMemo(() => ridgeGeometry(560, 30, 150, 11, "#0f141b", "#18202b"), []);
  const moonHalo = useMemo(() => {
    const m = glowSpriteMaterial("#cfd8e8", 0.35);
    m.fog = false;
    return m;
  }, []);
  const near = useMemo(() => ridgeGeometry(470, 10, 110, 4, "#090b0d", "#10151a"), []);

  return (
    <group ref={group}>
      <mesh geometry={sky} renderOrder={-2}>
        <meshBasicMaterial vertexColors side={THREE.BackSide} fog={false} depthWrite={false} />
      </mesh>
      <points geometry={stars} renderOrder={-1}>
        <pointsMaterial color="#cfd6e2" size={1.4} sizeAttenuation={false} fog={false} depthWrite={false} />
      </points>
      <sprite position={[190, 170, -559]} scale={70} renderOrder={-1} material={moonHalo} />
      <mesh position={[190, 170, -560]} renderOrder={-1}>
        <circleGeometry args={[10, 32]} />
        <meshBasicMaterial color={new THREE.Color("#f3eee2").multiplyScalar(1.6)} toneMapped={false} fog={false} />
      </mesh>
      <mesh geometry={far}>
        <meshBasicMaterial vertexColors fog={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh geometry={near}>
        <meshBasicMaterial vertexColors fog={false} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}
