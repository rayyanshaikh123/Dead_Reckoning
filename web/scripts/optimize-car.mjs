// Turns a downloaded car model into the site's `public/models/car.glb`.
//
//   npm run optimize:car -- [input.glb]      (default: model-src/1982_mercedes_w201.glb)
//
// What it does, and why:
//   • drops the interior, engine and invisible helper parts — the camera never
//     sees them, but every part costs a draw call and texture memory;
//   • makes the windows dark tinted glass (opaque: no see-through to the
//     removed cabin, and no transparency sorting);
//   • tints the paint graphite to match the site;
//   • names the lights "headlight_*" / "taillight_*" so the site drives them;
//   • splits each wheel part into four corners ("wheel_fl_*"…) so they spin;
//   • textures → WebP, at most 2048 px; geometry → meshopt-compressed.
// Part names below are from the 1982 Mercedes W201 (Sketchfab) model.

import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";

import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { dedup, meshopt, prune, textureCompress } from "@gltf-transform/functions";
import { MeshoptEncoder } from "meshoptimizer";
import sharp from "sharp";

const input = process.argv[2] ?? "model-src/1982_mercedes_w201.glb";
const output = "public/models/car.glb";

// Interior and engine go, except the glass parts (the rear window is one).
const DROP = /^(INT_(?!Glass)|ENG_)|^Material_109$|^Plate_D$|_Blur$/;
const GLASS = /^(EXT_Glass|INT_Glass|INT_Glass_Refl)$/;
const HEADLIGHT = /^HL_/;
const TAILLIGHT = /^TL_/;
const SPINS = /^EXT_(Rim|Rim_Badge|Rim_Bolts|Tyre|Brake_Disc)$/;

await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ "meshopt.encoder": MeshoptEncoder });
const doc = await io.read(input);
const root = doc.getRoot();
const matName = (node) => node.getMesh()?.listPrimitives()[0]?.getMaterial()?.getName() ?? "";

// ---- drop what the camera never sees -------------------------------------------
let dropped = 0;
for (const node of root.listNodes()) {
  if (node.getMesh() && DROP.test(matName(node))) {
    node.getMesh().dispose();
    node.dispose();
    dropped++;
  }
}

// ---- materials -------------------------------------------------------------------
for (const m of root.listMaterials()) {
  const name = m.getName();
  if (GLASS.test(name)) {
    // Double-sided: some panes face inwards, and culling them would show the
    // bare inside of the body through the window.
    m.setDoubleSided(true).setAlphaMode("OPAQUE").setBaseColorTexture(null).setBaseColorFactor([0.006, 0.007, 0.008, 1]).setMetallicFactor(0.25).setRoughnessFactor(0.12);
  } else if (name === "EXT_Car_Paint") {
    m.setBaseColorFactor([0.045, 0.05, 0.056, 1]); // graphite (linear)
  }
}

// ---- names the site looks for ----------------------------------------------------
for (const node of root.listNodes()) {
  const name = matName(node);
  if (HEADLIGHT.test(name)) node.setName(`headlight_${name}`);
  else if (TAILLIGHT.test(name)) node.setName(`taillight_${name}`);
}

// ---- wheels: split each part into its four corners --------------------------------
const spinning = root.listNodes().filter((n) => n.getMesh() && SPINS.test(matName(n)));

/** World-space centroid of every triangle, via the node's world matrix. */
function transform(m, [x, y, z]) {
  return [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
}

// Centre of all the wheels together, in world space (splits front/rear, left/right).
let lo = [Infinity, Infinity, Infinity];
let hi = [-Infinity, -Infinity, -Infinity];
for (const node of spinning) {
  const wm = node.getWorldMatrix();
  for (const prim of node.getMesh().listPrimitives()) {
    const pos = prim.getAttribute("POSITION");
    for (let i = 0; i < pos.getCount(); i++) {
      const p = transform(wm, pos.getElement(i, []));
      lo = lo.map((v, k) => Math.min(v, p[k]));
      hi = hi.map((v, k) => Math.max(v, p[k]));
    }
  }
}
const mid = lo.map((v, k) => (v + hi[k]) / 2);
// Longest horizontal extent of the wheel set = the car's length axis.
const lengthAxis = hi[0] - lo[0] > hi[2] - lo[2] ? 0 : 2;
const sideAxis = lengthAxis === 0 ? 2 : 0;

let split = 0;
for (const node of spinning) {
  const wm = node.getWorldMatrix();
  const parent = node.getParentNode();
  for (const prim of node.getMesh().listPrimitives()) {
    const index = prim.getIndices().getArray();
    const pos = prim.getAttribute("POSITION");
    const groups = new Map();
    for (let t = 0; t < index.length; t += 3) {
      const c = [0, 0, 0];
      for (let k = 0; k < 3; k++) transform(wm, pos.getElement(index[t + k], [])).forEach((v, a) => (c[a] += v / 3));
      const key = `${c[lengthAxis] > mid[lengthAxis] ? "a" : "b"}${c[sideAxis] > mid[sideAxis] ? "l" : "r"}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(index[t], index[t + 1], index[t + 2]);
    }
    for (const [key, tris] of groups) {
      // Compact: only this corner's vertices (so its bounds are the wheel's).
      const remap = new Map();
      const newIndex = new Uint32Array(tris.length);
      tris.forEach((v, i) => {
        if (!remap.has(v)) remap.set(v, remap.size);
        newIndex[i] = remap.get(v);
      });
      const part = doc.createPrimitive().setMaterial(prim.getMaterial()).setMode(prim.getMode());
      part.setIndices(doc.createAccessor().setType("SCALAR").setArray(newIndex).setBuffer(prim.getIndices().getBuffer()));
      for (const semantic of prim.listSemantics()) {
        const src = prim.getAttribute(semantic);
        const size = src.getElementSize();
        const Arr = src.getArray().constructor;
        const arr = new Arr(remap.size * size);
        const el = [];
        for (const [from, to] of remap) arr.set(src.getElement(from, el), to * size);
        part.setAttribute(
          semantic,
          doc.createAccessor().setType(src.getType()).setArray(arr).setNormalized(src.getNormalized()).setBuffer(src.getBuffer()),
        );
      }
      const mesh = doc.createMesh(`wheel_${key}_${matName(node)}`).addPrimitive(part);
      const corner = doc
        .createNode(`wheel_${key}_${matName(node)}`)
        .setMesh(mesh)
        .setTranslation(node.getTranslation())
        .setRotation(node.getRotation())
        .setScale(node.getScale());
      (parent ?? root.listScenes()[0]).addChild(corner);
      split++;
    }
  }
  node.getMesh().dispose();
  node.dispose();
}

// ---- compress ------------------------------------------------------------------------
await doc.transform(
  prune({ keepAttributes: false }),
  dedup(),
  textureCompress({ encoder: sharp, targetFormat: "webp", resize: [2048, 2048], quality: 85 }),
  meshopt({ encoder: MeshoptEncoder, level: "medium" }),
);

mkdirSync(path.dirname(output), { recursive: true });
await io.write(output, doc);

const tris = root.listMeshes().reduce((n, m) => n + m.listPrimitives().reduce((a, p) => a + (p.getIndices()?.getCount() ?? 0) / 3, 0), 0);
console.log(`dropped ${dropped} parts, wheels → ${split} corner parts`);
console.log(`${root.listMeshes().length} meshes, ${Math.round(tris).toLocaleString()} triangles, ${root.listTextures().length} textures`);
console.log(`wrote ${output}${existsSync(output) ? "" : " (missing?)"}`);
