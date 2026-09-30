"use client";

import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { GPS, PHONE } from "@/lib/car";
import { story } from "@/lib/story";

import { glowSpriteMaterial } from "./glow";

// A stylised sedan built in code (metres; nose towards +x, y up). Dark
// clear-coated body with line-art edges — the 3D twin of the app's car
// drawing — plus the parts the story needs: headlights (loading screen,
// tunnel), an IDR-orange tail bar, and the phone on the dash that the
// sensor callouts point at.

export const CAR = {
  length: 4.7,
  width: 1.9,
  wheelBase: 1.45, // wheel centres at ±x
  wheelRadius: 0.36,
  /** Phone in its dash mount (car coordinates). */
  phone: new THREE.Vector3(...PHONE),
  /** Where GPS comes from (the phone's receiver, drawn above the roof). */
  gps: new THREE.Vector3(...GPS),
} as const;

/** Body shell: bevelled extrusion reaches ±(length/2 + bevel) in x. */
const NOSE_X = 2.44;
const TAIL_X = -2.42;

const EDGE = "#f2f2f2";
/** Graphite clear-coat: dark like the app, but light enough to catch reflections. */
const PAINT = {
  color: "#3a3e42",
  metalness: 0.55,
  roughness: 0.28,
  clearcoat: 1,
  clearcoatRoughness: 0.08,
  envMapIntensity: 1.6,
} as const;
const ACCENT = "#f06131";

function bodyShape() {
  const s = new THREE.Shape();
  s.moveTo(-2.3, 0.32);
  s.lineTo(2.2, 0.32);
  s.quadraticCurveTo(2.36, 0.34, 2.36, 0.56);
  s.lineTo(2.32, 0.7);
  s.quadraticCurveTo(2.2, 0.86, 1.55, 0.9);
  s.lineTo(0.95, 0.97);
  s.lineTo(-1.45, 1.0);
  s.quadraticCurveTo(-2.15, 1.0, -2.32, 0.82);
  s.lineTo(-2.34, 0.5);
  s.quadraticCurveTo(-2.34, 0.32, -2.3, 0.32);
  return s;
}

function cabinShape() {
  const s = new THREE.Shape();
  s.moveTo(0.95, 0.97);
  s.quadraticCurveTo(0.42, 1.38, -0.05, 1.44);
  s.lineTo(-0.85, 1.43);
  s.quadraticCurveTo(-1.35, 1.36, -1.58, 1.0);
  s.lineTo(0.95, 0.97);
  return s;
}

/** The glasshouse's roof panel (painted, not glass). */
function roofShape() {
  const s = new THREE.Shape();
  s.moveTo(0.36, 1.3);
  s.quadraticCurveTo(0.18, 1.41, -0.05, 1.445);
  s.lineTo(-0.85, 1.435);
  s.quadraticCurveTo(-1.12, 1.4, -1.28, 1.3);
  s.lineTo(0.36, 1.3);
  return s;
}

const smooth = (a: number, b: number, v: number) => {
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Pulls the extrusion's sides in: rounded nose/tail in plan, tumblehome above. */
function sculpt(g: THREE.BufferGeometry, width: (x: number, y: number) => number) {
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    pos.setZ(i, pos.getZ(i) * width(x, y));
  }
  g.computeVertexNormals();
  return g;
}

const bodyWidth = (x: number, y: number) =>
  1 - 0.13 * smooth(1.55, 2.44, x) - 0.1 * smooth(-1.75, -2.42, x) - 0.06 * smooth(0.72, 1.02, y);
const cabinWidth = (_x: number, y: number) => 1 - 0.26 * smooth(0.98, 1.46, y);

function extrude(shape: THREE.Shape, depth: number, bevel: number) {
  const g = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 4,
    curveSegments: 24,
  });
  g.translate(0, 0, -depth / 2);
  g.computeVertexNormals();
  return g;
}

function Wheel({ x, z, spin }: { x: number; z: number; spin: React.RefObject<number> }) {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    if (ref.current) ref.current.rotation.z = -spin.current / CAR.wheelRadius;
  });
  return (
    <group position={[x, CAR.wheelRadius, z]}>
      <group ref={ref}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[CAR.wheelRadius, CAR.wheelRadius, 0.26, 32]} />
          <meshStandardMaterial color="#0c0d0d" roughness={0.9} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, z > 0 ? 0.131 : -0.131]}>
          <cylinderGeometry args={[0.22, 0.22, 0.01, 24]} />
          <meshStandardMaterial color="#3a3b3b" metalness={0.8} roughness={0.35} />
        </mesh>
        {/* spokes, so spinning reads */}
        {[0, 1, 2, 3, 4].map((i) => (
          <mesh
            key={i}
            position={[0, 0, z > 0 ? 0.138 : -0.138]}
            rotation={[0, 0, (i / 5) * Math.PI * 2]}
          >
            <boxGeometry args={[0.38, 0.035, 0.01]} />
            <meshStandardMaterial color="#8c8d8d" metalness={0.6} roughness={0.4} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

type CarProps = {
  /** Distance driven (m) — spins the wheels. */
  distanceRef: React.RefObject<number>;
};

/**
 * The headlight beam (one spotlight for both lamps — every lit material pays
 * per light) plus, optionally, glowing lamp lenses. Brightness comes from
 * `story.lampL/R`, which flicker on during the intro.
 */
export function Headlights({
  noseX,
  y,
  halfSpread,
  lenses = true,
  materials = [],
  glowAt,
}: {
  noseX: number;
  y: number;
  halfSpread: number;
  lenses?: boolean;
  /** Headlight materials found in a loaded model: [left, right]. */
  materials?: THREE.MeshStandardMaterial[][];
  /** Where each lamp's glow sits (the model's real lamps), left then right. */
  glowAt?: [number, number, number][];
}) {
  const glowPos = glowAt ?? [
    [noseX + 0.08, y, halfSpread],
    [noseX + 0.08, y, -halfSpread],
  ];
  const root = useRef<THREE.Group>(null);
  const lensMats = useRef<THREE.MeshBasicMaterial[]>([]);
  const beam = useRef<THREE.SpotLight>(null);
  const white = useMemo(() => new THREE.Color("#fff1d6"), []);
  // Per lamp: a tight core and a wide halo (stand-ins for bloom).
  const glows = useMemo(
    () => [0, 1].map(() => ({ core: glowSpriteMaterial("#fff4e2"), halo: glowSpriteMaterial("#ffe9c8", 0) })),
    [],
  );
  const v = useMemo(() => ({ fwd: new THREE.Vector3(), to: new THREE.Vector3(), at: new THREE.Vector3() }), []);

  // A spotlight aims at its `target`, which must be in the scene graph.
  useLayoutEffect(() => {
    const b = beam.current;
    if (!b) return;
    b.target.position.set(noseX + 12, -0.8, 0);
    b.parent?.add(b.target);
  }, [noseX]);

  useFrame(({ camera }) => {
    const l = story.lampL;
    const r = story.lampR;
    lensMats.current.forEach((m, i) => {
      if (m) m.color.copy(white).multiplyScalar(0.15 + (i === 0 ? l : r) * 6);
    });
    materials.forEach((list, i) =>
      list.forEach((m) => {
        m.emissive.copy(white);
        m.emissiveIntensity = 0.1 + (i === 0 ? l : r) * 6;
      }),
    );
    if (beam.current) beam.current.intensity = ((l + r) / 2) * 70;

    // Glare is strongest looking into the lamps, faint from the side, none behind.
    const g = root.current;
    if (!g) return;
    v.fwd.set(1, 0, 0).transformDirection(g.matrixWorld);
    v.at.set(noseX, y, 0).applyMatrix4(g.matrixWorld);
    const facing = Math.max(0, v.fwd.dot(v.to.subVectors(camera.position, v.at).normalize()));
    const glare = 0.08 + 0.92 * facing ** 2;
    glows.forEach((gl, i) => {
      const on = i === 0 ? l : r;
      gl.core.opacity = on * (0.2 + 0.6 * glare);
      gl.halo.opacity = on * glare * 0.4;
    });
  });

  return (
    <group ref={root}>
      {lenses &&
        [1, -1].map((side, i) => (
          <mesh key={side} position={[noseX - 0.1, y, side * halfSpread]} rotation={[0, 0, 0.12]}>
            <boxGeometry args={[0.2, 0.07, 0.4]} />
            <meshBasicMaterial
              ref={(m) => {
                if (m) lensMats.current[i] = m;
              }}
              color="#000000"
              toneMapped={false}
            />
          </mesh>
        ))}
      {glowPos.map((p, i) => (
        <group key={i} position={p}>
          <sprite material={glows[i].core} scale={0.34} />
          <sprite material={glows[i].halo} scale={2.6} />
        </group>
      ))}
      <spotLight
        ref={beam}
        color="#fff1d6"
        angle={0.5}
        penumbra={0.8}
        distance={50}
        decay={1.4}
        intensity={0}
        position={[noseX + 0.05, y, 0]}
      />
    </group>
  );
}

/** Soft orange glow on the tail lights, seen from behind. */
export function TailGlow({ points, tailX }: { points: [number, number, number][]; tailX: number }) {
  const root = useRef<THREE.Group>(null);
  const mat = useMemo(() => glowSpriteMaterial("#ff3a22", 0.3), []);
  const v = useMemo(() => ({ back: new THREE.Vector3(), to: new THREE.Vector3(), at: new THREE.Vector3() }), []);
  useFrame(({ camera }) => {
    const g = root.current;
    if (!g) return;
    v.back.set(-1, 0, 0).transformDirection(g.matrixWorld);
    v.at.set(tailX, 0.8, 0).applyMatrix4(g.matrixWorld);
    const facing = Math.max(0, v.back.dot(v.to.subVectors(camera.position, v.at).normalize()));
    mat.setValues({ opacity: 0.35 * facing * facing });
  });
  return (
    <group ref={root}>
      {points.map((p, i) => (
        <sprite key={i} material={mat} position={p} scale={0.6} />
      ))}
    </group>
  );
}

/** The phone in its dash mount — the only hardware IDR needs. */
export function Phone() {
  const screen = useRef<THREE.MeshBasicMaterial>(null);
  const base = useMemo(() => new THREE.Color(ACCENT), []);
  useFrame((state) => {
    if (screen.current) screen.current.color.copy(base).multiplyScalar(1.5 + Math.sin(state.clock.elapsedTime * 3) * 0.6);
  });
  return (
    <group position={CAR.phone.toArray()} rotation={[0, Math.PI / 2, -0.25]}>
      <mesh>
        <boxGeometry args={[0.08, 0.16, 0.01]} />
        <meshStandardMaterial color="#0a0a0a" />
      </mesh>
      <mesh position={[0, 0, 0.006]}>
        <planeGeometry args={[0.07, 0.145]} />
        <meshBasicMaterial ref={screen} color={ACCENT} toneMapped={false} />
      </mesh>
    </group>
  );
}

/** The built-in stylised sedan (used until a model is provided). */
export function ProceduralCar({ distanceRef }: CarProps) {
  const body = useMemo(() => sculpt(extrude(bodyShape(), CAR.width - 0.16, 0.08), bodyWidth), []);
  const cabin = useMemo(() => sculpt(extrude(cabinShape(), CAR.width - 0.46, 0.05), cabinWidth), []);
  const roof = useMemo(() => {
    const g = sculpt(extrude(roofShape(), CAR.width - 0.46, 0.05), cabinWidth);
    g.scale(1, 1, 1.02);
    g.translate(0, 0.008, 0);
    return g;
  }, []);
  const bodyEdges = useMemo(() => new THREE.EdgesGeometry(body, 28), [body]);
  const cabinEdges = useMemo(() => new THREE.EdgesGeometry(cabin, 28), [cabin]);

  return (
    <group>
      {/* body + glasshouse */}
      <mesh geometry={body} castShadow>
        <meshPhysicalMaterial {...PAINT} />
      </mesh>
      <lineSegments geometry={bodyEdges}>
        <lineBasicMaterial color={EDGE} transparent opacity={0.35} />
      </lineSegments>
      <mesh geometry={cabin} castShadow>
        <meshPhysicalMaterial color="#0b0d0f" metalness={0.9} roughness={0.06} clearcoat={1} envMapIntensity={1.4} />
      </mesh>
      <lineSegments geometry={cabinEdges}>
        <lineBasicMaterial color={EDGE} transparent opacity={0.45} />
      </lineSegments>
      <mesh geometry={roof}>
        <meshPhysicalMaterial {...PAINT} />
      </mesh>

      <Headlights noseX={NOSE_X} y={0.68} halfSpread={0.6} />

      {/* IDR-orange tail bar */}
      <mesh position={[TAIL_X + 0.06, 0.8, 0]}>
        <boxGeometry args={[0.14, 0.05, CAR.width - 0.42]} />
        <meshBasicMaterial color={new THREE.Color(ACCENT).multiplyScalar(2.4)} toneMapped={false} />
      </mesh>

      <TailGlow
        tailX={TAIL_X}
        points={[
          [TAIL_X - 0.05, 0.8, 0.55],
          [TAIL_X - 0.05, 0.8, -0.55],
        ]}
      />

      <Phone />

      <Wheel x={CAR.wheelBase} z={0.84} spin={distanceRef} />
      <Wheel x={CAR.wheelBase} z={-0.84} spin={distanceRef} />
      <Wheel x={-CAR.wheelBase} z={0.84} spin={distanceRef} />
      <Wheel x={-CAR.wheelBase} z={-0.84} spin={distanceRef} />
    </group>
  );
}
