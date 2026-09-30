"use client";

import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { CAR_MODEL } from "@/lib/carModel";

import { Headlights, TailGlow } from "./Car";

const HEADLIGHT = /head.?(light|lamp)|front.?light|\bdrl\b/i;
const TAILLIGHT = /tail.?(light|lamp)|rear.?light|brake|stop.?lamp/i;
const WHEEL = /wheel|tire|tyre|\brim\b/i;

type Prepared = {
  root: THREE.Group;
  /** Headlight materials, [left, right] (left = +z side). */
  lamps: THREE.MeshStandardMaterial[][];
  wheels: { pivot: THREE.Object3D; radius: number }[];
  noseX: number;
  tailX: number;
  lampY: number;
  halfSpread: number;
  /** Front face of each headlight, left then right (for its glow). */
  heads: [number, number, number][];
  /** Back face of each tail light (for its glow). */
  tails: [number, number, number][];
};

const named = (o: THREE.Object3D, re: RegExp) => {
  const m = (o as THREE.Mesh).material;
  const matNames = (Array.isArray(m) ? m : m ? [m] : []).map((x) => x.name).join(" ");
  let p: THREE.Object3D | null = o;
  let names = matNames;
  for (let i = 0; p && i < 3; i++, p = p.parent) names += " " + p.name;
  return re.test(names);
};

/**
 * Bounds of the left (z ≥ 0) and right halves of some meshes, vertex by
 * vertex — one mesh often holds both lamps, so its own centre is between them.
 */
function sideBoxes(meshes: THREE.Mesh[]) {
  const boxes = [new THREE.Box3(), new THREE.Box3()];
  const v = new THREE.Vector3();
  for (const m of meshes) {
    const pos = m.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
      boxes[v.z >= 0 ? 0 : 1].expandByPoint(v);
    }
  }
  return boxes;
}

/** Normalises any car model: nose to +x, `length` long, on the ground, centred. */
function prepare(source: THREE.Object3D): Prepared {
  const model = source.clone(true);
  const inner = new THREE.Group();
  inner.add(model);
  const root = new THREE.Group();
  root.add(inner);

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  // Longest horizontal side becomes x.
  let yaw = size.z > size.x ? -Math.PI / 2 : 0;
  inner.rotation.y = yaw;
  root.updateMatrixWorld(true);

  // Headlights tell us which end is the nose.
  const meshes: THREE.Mesh[] = [];
  model.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
  });
  const heads = meshes.filter((m) => named(m, HEADLIGHT));
  const centre = new THREE.Box3().setFromObject(inner).getCenter(new THREE.Vector3());
  if (heads.length) {
    const hc = new THREE.Box3();
    heads.forEach((m) => hc.expandByObject(m));
    if (hc.getCenter(new THREE.Vector3()).x < centre.x) yaw += Math.PI;
  }
  inner.rotation.y = yaw + CAR_MODEL.yaw;
  root.updateMatrixWorld(true);

  // Scale to length, then sit it on the ground at the origin.
  const b1 = new THREE.Box3().setFromObject(inner);
  const s1 = b1.getSize(new THREE.Vector3());
  inner.scale.setScalar(CAR_MODEL.length / s1.x);
  root.updateMatrixWorld(true);
  const b2 = new THREE.Box3().setFromObject(inner);
  const c2 = b2.getCenter(new THREE.Vector3());
  inner.position.set(-c2.x, -b2.min.y, -c2.z);
  root.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(inner);
  const dims = bounds.getSize(new THREE.Vector3());

  // Materials: own copies, tuned for the studio lighting.
  const lamps: THREE.MeshStandardMaterial[][] = [[], []];
  meshes.forEach((m) => {
    const list = Array.isArray(m.material) ? m.material : [m.material];
    m.castShadow = true;
    const copies = list.map((mat) => {
      const c = mat.clone() as THREE.MeshStandardMaterial;
      if ("envMapIntensity" in c) c.envMapIntensity = 1.4;
      for (const t of [c.map, c.normalMap, c.roughnessMap, c.metalnessMap]) if (t) t.anisotropy = 8;
      return c;
    });
    m.material = Array.isArray(m.material) ? copies : copies[0];
    if (named(m, HEADLIGHT)) {
      const side = new THREE.Box3().setFromObject(m).getCenter(new THREE.Vector3()).z >= 0 ? 0 : 1;
      copies.forEach((c) => c.isMeshStandardMaterial && lamps[side].push(c));
    } else if (named(m, TAILLIGHT)) {
      // Lit through their own lens texture (red and amber segments, like the
      // real lamp) rather than painted one flat colour.
      copies.forEach((c) => {
        if (!c.isMeshStandardMaterial) return;
        if (c.map) {
          c.emissive = new THREE.Color("#ffffff");
          c.emissiveMap = c.map;
          c.emissiveIntensity = 1.5;
        } else {
          c.emissive = new THREE.Color("#ff2a1a");
          c.emissiveIntensity = 1.2;
        }
      });
    }
  });

  // Wheels: gather parts per corner under a pivot at the wheel's centre.
  const corners = new Map<string, THREE.Mesh[]>();
  meshes
    .filter((m) => named(m, WHEEL))
    .forEach((m) => {
      const c = new THREE.Box3().setFromObject(m).getCenter(new THREE.Vector3());
      const key = `${c.x > 0 ? "f" : "r"}${c.z > 0 ? "l" : "r"}`;
      corners.set(key, [...(corners.get(key) ?? []), m]);
    });
  const wheels: Prepared["wheels"] = [];
  corners.forEach((parts) => {
    const wb = new THREE.Box3();
    parts.forEach((p) => wb.expandByObject(p));
    const pivot = new THREE.Object3D();
    wb.getCenter(pivot.position);
    root.add(pivot);
    root.updateMatrixWorld(true);
    parts.forEach((p) => pivot.attach(p));
    wheels.push({ pivot, radius: Math.max(0.2, wb.getSize(new THREE.Vector3()).y / 2) });
  });

  root.updateMatrixWorld(true);
  const face = (boxes: THREE.Box3[], front: boolean) =>
    boxes
      .filter((b) => !b.isEmpty())
      .map((b) => {
        const c = b.getCenter(new THREE.Vector3());
        return [front ? b.max.x + 0.02 : b.min.x - 0.02, c.y, c.z] as [number, number, number];
      });

  return {
    root,
    lamps,
    wheels,
    noseX: bounds.max.x,
    tailX: bounds.min.x,
    heads: face(sideBoxes(meshes.filter((m) => named(m, HEADLIGHT))), true),
    tails: face(sideBoxes(meshes.filter((m) => named(m, TAILLIGHT))), false),
    lampY: bounds.min.y + dims.y * 0.46,
    halfSpread: dims.z * 0.32,
  };
}

/** A provided car model (public/models/car.glb) in place of the built-in one. */
export function ModelCar({ distanceRef }: { distanceRef: React.RefObject<number> }) {
  const { scene } = useGLTF(CAR_MODEL.url, false);
  const car = useMemo(() => prepare(scene), [scene]);
  const spun = useRef(0);
  useFrame(() => {
    const d = distanceRef.current;
    if (d === spun.current) return;
    spun.current = d;
    car.wheels.forEach((w) => (w.pivot.rotation.z = -d / w.radius));
  });
  return (
    <group>
      <primitive object={car.root} />
      <Headlights
        noseX={car.noseX}
        y={car.lampY}
        halfSpread={car.halfSpread}
        lenses={car.lamps[0].length + car.lamps[1].length === 0}
        materials={car.lamps}
        glowAt={car.heads.length === 2 ? car.heads : undefined}
      />
      <TailGlow points={car.tails} tailX={car.tailX} />
    </group>
  );
}
