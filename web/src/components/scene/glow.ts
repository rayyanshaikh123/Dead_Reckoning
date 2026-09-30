import * as THREE from "three";

// Light glow without a post-processing bloom pass: soft additive sprites.
// Bloom re-renders the whole screen several times per frame (costly on the
// tile-based GPUs in Macs and phones); a few sprites cost next to nothing.

let texture: THREE.Texture | null = null;

/** Shared soft round glow (bright core, long falloff). */
export function glowTexture() {
  if (texture) return texture;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.12, "rgba(255,255,255,0.75)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.22)");
  grad.addColorStop(0.7, "rgba(255,255,255,0.05)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** Material for a glow sprite (additive: it only ever adds light). */
export function glowSpriteMaterial(color: THREE.ColorRepresentation, opacity = 1) {
  return new THREE.SpriteMaterial({
    map: glowTexture(),
    color,
    opacity,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });
}

/** Material for many glows in one draw call (a `points` object). */
export function glowPointsMaterial(color: THREE.ColorRepresentation, size: number, opacity = 1) {
  return new THREE.PointsMaterial({
    map: glowTexture(),
    color,
    size,
    opacity,
    sizeAttenuation: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });
}
