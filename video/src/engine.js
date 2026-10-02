// IDR story: three.js scenes (3D phone + real-drive map). Ported from a standalone renderer.
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { Reflector } from 'three/examples/jsm/objects/Reflector.js';


export async function createEngine(canvas, assetUrl) {
const WIDTH = 1920, HEIGHT = 1080;
const ACCENT = new THREE.Color('#f06131');
const PI = Math.PI;

// ---------- easing helpers ----------
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lin = (t, a, b) => clamp((t - a) / (b - a));
const lerp = (a, b, k) => a + (b - a) * k;
const eio = x => (x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const eo = x => 1 - Math.pow(1 - x, 3);
const eo5 = x => 1 - Math.pow(1 - x, 5);
const ei = x => x * x * x;
const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
const vl = (a, b, k) => a.clone().lerp(b, k);
const env = (t, a, b, c, d) => Math.min(lin(t, a, b), 1 - lin(t, c, d)); // trapezoid

// ---------- renderer ----------
// Rendering is driven by Remotion: update(t) poses the scene for time t, draw() renders it.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(WIDTH, HEIGHT);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(30, WIDTH / HEIGHT, 0.05, 200);
scene.add(camera);

// ---------- environment (studio softboxes, orange strip) ----------
{
  const es = new THREE.Scene();
  es.background = new THREE.Color(0x020202);
  const add = (w, h, color, k, pos) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(k), side: THREE.DoubleSide }));
    m.position.copy(pos); m.lookAt(0, 0, 0); es.add(m);
  };
  add(7, 2.2, '#ffffff', 1.6, v3(0, 7, 2));        // top softbox
  add(0.5, 9, '#f06131', 7.0, v3(-6, 0.5, -3));    // orange strip left-back
  add(0.5, 9, '#f06131', 5.0, v3(6.5, 0, -2));     // orange strip right-back
  add(0.25, 9, '#ffffff', 2.2, v3(5, 0, 5));       // thin white strip front-right
  add(3, 5, '#ffffff', 0.6, v3(-6, 1, 5));         // dim fill front-left
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(es, 0.03).texture;
  window.__env = scene.environment;
}

// ---------- background gradient (camera-locked) ----------
const bgMat = new THREE.ShaderMaterial({
  uniforms: { uI: { value: 1 }, uGlow: { value: new THREE.Vector2(0.5, 0.55) }, uGlowI: { value: 0.0 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `varying vec2 vUv; uniform float uI; uniform vec2 uGlow; uniform float uGlowI;
    void main(){ vec2 p=(vUv-0.5)*vec2(1.78,1.); float r=length(p-vec2(0.,0.08));
      vec3 c=mix(vec3(0.03,0.031,0.032), vec3(0.0), smoothstep(0.0,1.0,r));
      vec2 g=(vUv-uGlow)*vec2(1.78,1.); c+= vec3(0.94,0.38,0.19)*0.03*uGlowI*exp(-dot(g,g)*5.0);
      gl_FragColor=vec4(c*uI,1.); }`,
  depthWrite: false, depthTest: false, toneMapped: false,
});
const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat);
bg.position.z = -150; bg.renderOrder = -10; bg.frustumCulled = false;
camera.add(bg);
function fitBg() { const h = 2 * 150 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)); bg.scale.set(h * camera.aspect / 2 * 1.02, h / 2 * 1.02, 1); }

// ---------- lights ----------
scene.add(new THREE.HemisphereLight(0x9aa0a6, 0x050505, 0.15));
const key = new THREE.DirectionalLight(0xfff1e0, 1.2);
key.position.set(3, 6, 6); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); key.shadow.camera.left = -5; key.shadow.camera.right = 5; key.shadow.camera.top = 5; key.shadow.camera.bottom = -5;
key.shadow.camera.near = 1; key.shadow.camera.far = 20; key.shadow.bias = -0.0005; key.shadow.radius = 3;
scene.add(key);
function spot(x, y, z, color) {
  const s = new THREE.SpotLight(color, 0, 30, 0.6, 0.6, 2);
  s.position.set(x, y, z); scene.add(s); scene.add(s.target); return s;
}
const rimL = spot(-4.5, 2.0, -3.5, ACCENT);
const rimR = spot(4.5, 0.5, -3.0, ACCENT);
const rimTop = spot(0, 6, -2.5, 0xffe0c8);
const beamLight = new THREE.PointLight(0xffe6cc, 0, 12, 2);
scene.add(beamLight);
const sweep = new THREE.PointLight(0xffffff, 0, 10, 2);
scene.add(sweep);

// ---------- textures ----------
const loader = new THREE.TextureLoader();
const loadTex = url => new Promise(r => loader.load(url, t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; r(t); }));
const screens = {};
for (const n of ['01_intro', '02_home', '03_tunnel_test', '04_live', '07_drive_report']) screens[n] = await loadTex(assetUrl(`screens/${n}.webp`));

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
// deterministic rng
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

const MARK = 'M70 352 C150 352 175 300 205 270 C240 236 270 250 300 222 C330 194 360 168 440 150';
const MARK_TUNNEL = 'M205 270 C240 236 270 250 300 222';

const pcbTex = canvasTex(1024, 800, (g, w, h) => {
  g.fillStyle = '#0b120e'; g.fillRect(0, 0, w, h);
  g.strokeStyle = 'rgba(190,140,70,0.55)'; g.lineWidth = 3;
  for (let i = 0; i < 140; i++) {
    let x = rnd() * w, y = rnd() * h; g.beginPath(); g.moveTo(x, y);
    for (let k = 0; k < 4; k++) { if (rnd() < .5) x += (rnd() - .5) * 300; else y += (rnd() - .5) * 300; g.lineTo(x, y); }
    g.stroke();
  }
  g.fillStyle = 'rgba(210,170,90,0.8)';
  for (let i = 0; i < 260; i++) g.fillRect(rnd() * w, rnd() * h, 7, 7);
  g.fillStyle = 'rgba(255,255,255,0.35)'; g.font = '22px monospace';
  g.fillText('IDR-IMU', 300, 520); g.fillText('U12', 640, 300); g.fillText('C34', 120, 140);
});
const batTex = canvasTex(1024, 1500, (g, w, h) => {
  g.fillStyle = '#1b1c1d'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#f06131'; g.fillRect(0, 120, w, 26);
  g.fillStyle = 'rgba(242,242,242,0.75)'; g.font = '600 64px monospace'; g.fillText('Li-ion', 80, 330);
  g.font = '36px monospace'; g.fillStyle = 'rgba(242,242,242,0.4)'; g.fillText('3.87 V', 80, 400);
  g.strokeStyle = 'rgba(255,255,255,0.08)'; g.lineWidth = 2; g.strokeRect(40, 40, w - 80, h - 80);
});
const backLogoTex = canvasTex(512, 512, (g) => {
  g.clearRect(0, 0, 512, 512); g.lineCap = 'round';
  g.lineWidth = 34; g.strokeStyle = 'rgba(205,205,205,0.9)'; g.stroke(new Path2D(MARK));
  g.strokeStyle = '#f06131'; g.stroke(new Path2D(MARK_TUNNEL));
  g.fillStyle = 'rgba(205,205,205,0.9)'; g.beginPath(); g.arc(440, 150, 30, 0, 7); g.fill();
});
const imuTopTex = canvasTex(256, 256, (g) => {
  g.fillStyle = '#121314'; g.fillRect(0, 0, 256, 256);
  g.fillStyle = '#f06131'; g.beginPath(); g.arc(46, 46, 16, 0, 7); g.fill();
  g.fillStyle = 'rgba(255,255,255,0.75)'; g.font = '600 46px monospace'; g.fillText('IMU', 70, 150);
  g.font = '26px monospace'; g.fillStyle = 'rgba(255,255,255,0.45)'; g.fillText('9-AXIS', 70, 196);
});

// ---------- phone geometry ----------
const W = 1.5, H = 3.1, R = 0.24;
function rrShape(w, h, r) {
  const s = new THREE.Shape(); const x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.absarc(x + w - r, y + r, r, -PI / 2, 0);
  s.lineTo(x + w, y + h - r); s.absarc(x + w - r, y + h - r, r, 0, PI / 2);
  s.lineTo(x + r, y + h); s.absarc(x + r, y + h - r, r, PI / 2, PI);
  s.lineTo(x, y + r); s.absarc(x + r, y + r, r, PI, PI * 1.5);
  return s;
}
function rrPath(w, h, r) { const s = rrShape(w, h, r); const p = new THREE.Path(); p.curves = s.curves; return p; }
function extrude(shape, depth, bevel, z0) {
  const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: bevel > 0, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 5, curveSegments: 32 });
  g.translate(0, 0, z0); g.computeVertexNormals(); return g;
}
function planeUV(geom, w, h) {
  const p = geom.attributes.position, uv = geom.attributes.uv;
  for (let i = 0; i < p.count; i++) uv.setXY(i, (p.getX(i) + w / 2) / w, (p.getY(i) + h / 2) / h);
  uv.needsUpdate = true; return geom;
}

const mFrame = new THREE.MeshPhysicalMaterial({ color: 0x3a3b3d, metalness: 1, roughness: 0.2, clearcoat: 0.6, clearcoatRoughness: 0.1, envMapIntensity: 1.3 });
const mBack = new THREE.MeshPhysicalMaterial({ color: 0x18191a, metalness: 0.35, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.08, envMapIntensity: 1.0 });
const mBlack = new THREE.MeshStandardMaterial({ color: 0x040404, roughness: 0.4, metalness: 0.2 });
const mGlass = new THREE.ShaderMaterial({
  transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
  vertexShader: `varying vec3 vN; varying vec3 vV; varying vec3 vW; void main(){ vec4 w=modelMatrix*vec4(position,1.); vW=w.xyz; vN=normalize(mat3(modelMatrix)*normal); vV=normalize(cameraPosition-w.xyz); gl_Position=projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `varying vec3 vN; varying vec3 vV; varying vec3 vW; void main(){ vec3 n=normalize(vN); if(!gl_FrontFacing) n=-n; float f=pow(1.-clamp(dot(n,vV),0.,1.),4.);
    vec3 r=reflect(-vV,n); float sky=smoothstep(-0.2,0.9,r.y); float streak=smoothstep(0.035,0.0,abs(r.x*0.8+r.y*0.6-0.15))*0.5;
    vec3 c=vec3(1.0,0.97,0.94)*(0.03+0.5*f)*(0.4+0.6*sky) + vec3(1.)*streak*0.10;
    gl_FragColor=vec4(c,1.); }`,
});
const mScreen = new THREE.MeshBasicMaterial({ map: screens['02_home'], toneMapped: false });
const mLensGlass = new THREE.MeshPhysicalMaterial({ color: 0x05080f, metalness: 0.6, roughness: 0.04, clearcoat: 1, iridescence: 0.8, iridescenceIOR: 1.6, envMapIntensity: 1.6 });
const mPCB = new THREE.MeshStandardMaterial({ map: pcbTex, roughness: 0.55, metalness: 0.3 });
const mChip = new THREE.MeshStandardMaterial({ color: 0x1d1e20, roughness: 0.35, metalness: 0.5 });
const mBat = new THREE.MeshStandardMaterial({ color: 0x1b1c1d, roughness: 0.5, metalness: 0.3 });
const mBatLabel = new THREE.MeshStandardMaterial({ map: batTex, roughness: 0.5, metalness: 0.2 });

const phone = new THREE.Group(); scene.add(phone);
const layers = {};
function layer(name) { const g = new THREE.Group(); phone.add(g); layers[name] = g; return g; }

// frame ring
const frameG = layer('frame');
{
  const s = rrShape(W, H, R); s.holes.push(rrPath(W - 0.1, H - 0.1, R - 0.05));
  const m = new THREE.Mesh(extrude(s, 0.13, 0.012, -0.065), mFrame); frameG.add(m);
  const btn = (y, h) => { const b = new THREE.Mesh(new THREE.BoxGeometry(0.03, h, 0.05), mFrame); b.position.set(W / 2 + 0.014, y, 0); frameG.add(b); };
  btn(0.75, 0.42); btn(0.2, 0.22);
}
// back shell + camera bump
const backG = layer('back');
{
  backG.add(new THREE.Mesh(extrude(rrShape(W - 0.03, H - 0.03, R - 0.015), 0.006, 0.003, -0.0865), mBack));
  const bump = new THREE.Group(); bump.position.set(-0.36, 1.08, 0); backG.add(bump);
  bump.add(new THREE.Mesh(extrude(rrShape(0.6, 0.62, 0.16), 0.032, 0.008, -0.125), mBack));
  const lens = (x, y, r) => {
    const g = new THREE.Group(); g.position.set(x, y, -0.137); bump.add(g);
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(r + 0.016, r + 0.016, 0.02, 48), mFrame); ring.rotation.x = PI / 2; g.add(ring);
    const gl = new THREE.Mesh(new THREE.CircleGeometry(r, 48), mLensGlass); gl.position.z = -0.0105; gl.rotation.y = PI; g.add(gl);
    const inner = new THREE.Mesh(new THREE.RingGeometry(r * 0.35, r * 0.5, 48), new THREE.MeshStandardMaterial({ color: 0x0a0a0a, metalness: 1, roughness: 0.2 }));
    inner.position.z = -0.011; inner.rotation.y = PI; g.add(inner);
  };
  lens(-0.14, 0.15, 0.1); lens(-0.14, -0.15, 0.1); lens(0.15, 0.0, 0.1);
  const flash = new THREE.Mesh(new THREE.CircleGeometry(0.035, 32), new THREE.MeshStandardMaterial({ color: 0xfff1dc, emissive: 0x332b20, roughness: 0.3 }));
  flash.position.set(0.17, 0.2, -0.1305); flash.rotation.y = PI; bump.add(flash);
  const logo = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.62), new THREE.MeshPhysicalMaterial({ map: backLogoTex, transparent: true, metalness: 0.9, roughness: 0.25, clearcoat: 1 }));
  logo.position.set(0, -0.15, -0.0901); logo.rotation.y = PI; backG.add(logo);
}
// battery
const batG = layer('battery');
{
  const b = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.75, 0.04), mBat); b.position.set(0, -0.52, -0.02); batG.add(b);
  const l = new THREE.Mesh(new THREE.PlaneGeometry(1.18, 1.73), mBatLabel); l.position.set(0, -0.52, -0.0405); l.rotation.y = PI; batG.add(l);
  const l2 = new THREE.Mesh(new THREE.PlaneGeometry(1.18, 1.73), mBatLabel); l2.position.set(0, -0.52, 0.0005); batG.add(l2);
}
// logic board + IMU
const boardG = layer('board');
const IMU = v3(-0.22, 0.78, -0.0455);
let imuGlow, axes, imuRing;
{
  const b = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.98, 0.016), [mChip, mChip, mChip, mChip, mPCB, mPCB]); b.position.set(0, 0.88, -0.02); boardG.add(b);
  const chip = (x, y, w, h, d, mat) => { const c = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat || mChip); c.position.set(x, y, -0.028 - d / 2); boardG.add(c); return c; };
  chip(0.28, 1.0, 0.38, 0.38, 0.026); chip(0.3, 0.6, 0.3, 0.16, 0.02); chip(-0.4, 1.17, 0.18, 0.12, 0.018); chip(0.52, 1.24, 0.1, 0.1, 0.012);
  for (let i = 0; i < 14; i++) chip(-0.55 + rnd() * 1.1, 0.45 + rnd() * 0.85, 0.035, 0.02, 0.01, new THREE.MeshStandardMaterial({ color: 0x8a6a3a, metalness: 0.8, roughness: 0.3 }));
  const imu = chip(IMU.x, IMU.y, 0.17, 0.17, 0.022, [mChip, mChip, mChip, mChip, mChip, new THREE.MeshStandardMaterial({ map: imuTopTex, emissive: 0xffffff, emissiveMap: imuTopTex, emissiveIntensity: 0.0, roughness: 0.4 })]);
  imuGlow = imu.material[5];
  // glowing axes from the IMU (X accent, Y positive green, Z location blue)
  axes = new THREE.Group(); axes.position.copy(IMU); axes.position.z -= 0.015; boardG.add(axes);
  const ax = (dir, color) => {
    const g = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(1.4), toneMapped: false, transparent: true });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.2, 12), mat); shaft.position.y = 0.1; g.add(shaft);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.04, 20), mat); cone.position.y = 0.22; g.add(cone);
    g.quaternion.setFromUnitVectors(v3(0, 1, 0), dir); axes.add(g); return g;
  };
  ax(v3(1, 0, 0), '#f06131'); ax(v3(0, 1, 0), '#48fa5d'); ax(v3(0, 0, -1), '#4285f4');
  imuRing = new THREE.Mesh(new THREE.RingGeometry(0.11, 0.116, 64), new THREE.MeshBasicMaterial({ color: ACCENT.clone().multiplyScalar(2), toneMapped: false, transparent: true, side: THREE.DoubleSide }));
  imuRing.position.copy(IMU); imuRing.position.z -= 0.02; boardG.add(imuRing);
}
// display + glass
const dispG = layer('display');
{
  dispG.add(new THREE.Mesh(extrude(rrShape(W - 0.04, H - 0.04, R - 0.02), 0.008, 0, 0.07), mBlack));
  const plate = new THREE.Mesh(new THREE.ShapeGeometry(rrShape(W - 0.04, H - 0.04, R - 0.02), 32), mBlack); plate.position.z = 0.0785; dispG.add(plate);
  const sw = W - 0.1, sh = sw * 1278 / 589;
  const scr = new THREE.Mesh(planeUV(new THREE.ShapeGeometry(rrShape(sw, sh, R - 0.06), 32), sw, sh), mScreen); scr.position.set(0, 0, 0.079); dispG.add(scr);
}
const glassG = layer('glass');
glassG.add(new THREE.Mesh(extrude(rrShape(W - 0.03, H - 0.03, R - 0.015), 0.006, 0.003, 0.0805), mGlass));
phone.traverse(o => { if (o.isMesh) { o.castShadow = true; } });

// screen-space mapping helper: screenshot pixel -> phone local
const SW = W - 0.1, SH = SW * 1278 / 589;
const scrPt = (px, py) => v3(-SW / 2 + SW * px / 589, SH / 2 - SH * py / 1278, 0.08);

// ---------- floor (hero) ----------
const FLOOR_Y = -H / 2 - 0.013;
const floor = new THREE.Group(); scene.add(floor);
{
  const refl = new Reflector(new THREE.PlaneGeometry(60, 60), { textureWidth: WIDTH, textureHeight: HEIGHT, color: 0x8a8a8a });
  refl.rotation.x = -PI / 2; refl.position.y = FLOOR_Y; floor.add(refl);
  const fade = canvasTex(512, 512, (g) => {
    const gr = g.createRadialGradient(256, 256, 0, 256, 256, 256);
    gr.addColorStop(0, 'rgba(0,0,0,0.45)'); gr.addColorStop(0.35, 'rgba(0,0,0,0.75)'); gr.addColorStop(1, 'rgba(0,0,0,1)');
    g.fillStyle = gr; g.fillRect(0, 0, 512, 512);
  });
  fade.colorSpace = THREE.NoColorSpace;
  const over = new THREE.Mesh(new THREE.PlaneGeometry(26, 26), new THREE.MeshBasicMaterial({ color: 0x000000, alphaMap: fade, transparent: true, depthWrite: false }));
  over.rotation.x = -PI / 2; over.position.y = FLOOR_Y + 0.002; floor.add(over);
  const sh = new THREE.Mesh(new THREE.PlaneGeometry(26, 26), new THREE.ShadowMaterial({ opacity: 0.6 }));
  sh.rotation.x = -PI / 2; sh.position.y = FLOOR_Y + 0.004; sh.receiveShadow = true; floor.add(sh);
}

// ---------- haze behind the product ----------
const hazeMat = new THREE.ShaderMaterial({
  uniforms: { uI: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
  vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `varying vec2 vUv; uniform float uI; void main(){ vec2 p=(vUv-.5)*2.; float d=dot(p,p);
    vec3 c=vec3(0.94,0.38,0.19)*exp(-d*3.0)*0.55 + vec3(1.0,0.75,0.6)*exp(-d*18.)*0.12; gl_FragColor=vec4(c*uI*0.2,1.); }`,
});
const haze = new THREE.Mesh(new THREE.PlaneGeometry(14, 10), hazeMat); haze.position.set(0, 0.3, -4.5); scene.add(haze);

// ---------- light beam (opening reveal) ----------
const beamMat = new THREE.ShaderMaterial({
  uniforms: { uX: { value: -10 }, uI: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
  vertexShader: `varying vec3 vP; void main(){ vP=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `varying vec3 vP; uniform float uX; uniform float uI;
    void main(){ float x = vP.x - uX - vP.y*0.18; float core=exp(-x*x*40.); float glow=exp(-x*x*1.2);
      float fy = exp(-vP.y*vP.y*0.035);
      vec3 c = vec3(1.0,0.93,0.85)*core*1.8 + vec3(0.94,0.45,0.22)*glow*0.35;
      gl_FragColor=vec4(c*fy*uI,1.); }`,
});
const beam = new THREE.Mesh(new THREE.PlaneGeometry(40, 16), beamMat); beam.position.set(0, 0, -1.6); scene.add(beam);

// ---------- particles ----------
const NP = 520;
const pGeom = new THREE.BufferGeometry();
{
  const pos = new Float32Array(NP * 3), sd = new Float32Array(NP);
  for (let i = 0; i < NP; i++) { pos[i * 3] = (rnd() - .5) * 18; pos[i * 3 + 1] = (rnd() - .5) * 10; pos[i * 3 + 2] = -9 + rnd() * 14; sd[i] = rnd(); }
  pGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3)); pGeom.setAttribute('seed', new THREE.BufferAttribute(sd, 1));
}
const pMat = new THREE.ShaderMaterial({
  uniforms: { uT: { value: 0 }, uI: { value: 0 }, uFocus: { value: 6 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
  vertexShader: `attribute float seed; uniform float uT; uniform float uFocus; varying float vA; varying float vSoft; varying float vS;
    void main(){ vec3 p=position; p.y = mod(p.y + uT*(0.08+seed*0.12) + 5., 10.) - 5.; p.x += sin(uT*0.4+seed*30.)*0.25; p.z += cos(uT*0.3+seed*17.)*0.2;
      vec4 mv = modelViewMatrix*vec4(p,1.); float dist=-mv.z; float blur=clamp(abs(dist-uFocus)/uFocus,0.,1.5);
      gl_PointSize = (3.0 + seed*4.0 + blur*55.0) * (6.0/max(dist,0.5)) ;
      vA = (0.35+0.65*seed) / (1.0 + blur*blur*14.0); vSoft = clamp(blur,0.05,1.); vS=seed;
      gl_Position = projectionMatrix*mv; }`,
  fragmentShader: `uniform float uI; varying float vA; varying float vSoft; varying float vS;
    void main(){ float d=length(gl_PointCoord-.5); float a=smoothstep(0.5,0.5-0.5*vSoft-0.02,d);
      vec3 c=mix(vec3(0.94,0.38,0.19), vec3(1.0,0.85,0.7), step(0.7,vS));
      gl_FragColor=vec4(c*a*vA*uI*1.6,1.); }`,
});
const points = new THREE.Points(pGeom, pMat); points.frustumCulled = false; scene.add(points);

// ---------- flare (sprite) ----------
const flareTex = canvasTex(1024, 256, (g, w, h) => {
  const gr = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, h / 2); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.2, 'rgba(255,200,160,0.6)'); gr.addColorStop(1, 'rgba(240,97,49,0)');
  g.fillStyle = gr; g.fillRect(w / 2 - h / 2, 0, h, h);
  const lg = g.createLinearGradient(0, 0, w, 0); lg.addColorStop(0, 'rgba(240,97,49,0)'); lg.addColorStop(0.5, 'rgba(255,230,210,1)'); lg.addColorStop(1, 'rgba(240,97,49,0)');
  g.fillStyle = lg; g.fillRect(0, h / 2 - 3, w, 6);
  g.globalAlpha = 0.35; g.fillRect(0, h / 2 - 10, w, 20);
});
const flare = new THREE.Sprite(new THREE.SpriteMaterial({ map: flareTex, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, transparent: true }));
flare.scale.set(8, 2, 1); scene.add(flare);

// ---------- post ----------
const rt = new THREE.WebGLRenderTarget(WIDTH, HEIGHT, { type: THREE.HalfFloatType, samples: 4 });
const composer = new EffectComposer(renderer, rt);
composer.setPixelRatio(1); composer.setSize(WIDTH, HEIGHT);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);
const bloom = new UnrealBloomPass(new THREE.Vector2(WIDTH, HEIGHT), 0.5, 0.5, 0.9);
composer.addPass(bloom);
composer.addPass(new OutputPass());
const finalPass = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uDir: { value: new THREE.Vector2() }, uRadial: { value: 0 }, uFlash: { value: 0 }, uFade: { value: 1 }, uTime: { value: 0 }, uCA: { value: 0.0025 }, uGrain: { value: 0.035 }, uBlur: { value: 0 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform vec2 uDir; uniform float uRadial,uFlash,uFade,uTime,uCA,uGrain,uBlur; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }
    void main(){ vec2 uv=vUv; vec2 c=uv-.5; vec3 col=vec3(0.); float tot=0.;
      int n = uBlur>0.5 ? 28 : 1;
      for(int i=0;i<28;i++){ if(i>=n) break; float f = n>1 ? float(i)/27.-.5 : 0.;
        vec2 suv = uv + uDir*f - c*uRadial*(f+.5); vec2 ca=c*uCA*(1.+uRadial*20.);
        col.r+=texture2D(tDiffuse,suv+ca).r; col.g+=texture2D(tDiffuse,suv).g; col.b+=texture2D(tDiffuse,suv-ca).b; tot+=1.; }
      col/=tot;
      float v = smoothstep(1.05,0.28,length(c*vec2(1.0,0.9))); col*=mix(0.45,1.,v);
      col = mix(col, vec3(1.0,0.96,0.9), uFlash);
      col += (hash(uv*vec2(1920.,1080.)+fract(uTime*7.13))-.5)*uGrain;
      gl_FragColor=vec4(col*uFade,1.); }`,
});
composer.addPass(finalPass);


// =====================================================================
//  MAP WORLD (real drive: IO-VNBD Drive M, "highway tunnel" test)
// =====================================================================
const MAP = await (await fetch(assetUrl('map.json'))).json();
const mapScene = new THREE.Scene();
mapScene.background = new THREE.Color(0x000000);
const mapCam = new THREE.PerspectiveCamera(35, WIDTH / HEIGHT, 1, 20000);
mapScene.add(mapCam);
const P3 = (p, y = 0) => new THREE.Vector3(p[0], y, -p[1]);
const ROUTE = MAP.route.map(p => P3(p));
const CUM = MAP.cum;
const O0 = MAP.o0, O1 = MAP.o1;
function routeAt(d) { // position along the route at distance d
  if (d <= 0) return ROUTE[0].clone();
  let i = 1; while (i < CUM.length - 1 && CUM[i] < d) i++;
  const k = clamp((d - CUM[i - 1]) / Math.max(1e-6, CUM[i] - CUM[i - 1]));
  return ROUTE[i - 1].clone().lerp(ROUTE[i], k);
}
function routeIdx(f) { const i = Math.floor(clamp(f, 0, ROUTE.length - 1.001)); return ROUTE[i].clone().lerp(ROUTE[i + 1], f - i); }
const D_ENTRY = CUM[O0], D_EXIT = CUM[O1];
const ENTRY = ROUTE[O0].clone(), EXIT = ROUTE[O1].clone();
const FWD = EXIT.clone().sub(ENTRY).normalize();
const SIDE = v3(-FWD.z, 0, FWD.x);

// ground with faint grid
const groundMat = new THREE.ShaderMaterial({
  uniforms: { uC: { value: new THREE.Vector3() } }, toneMapped: false,
  vertexShader: `varying vec3 vW; void main(){ vec4 w=modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `varying vec3 vW; uniform vec3 uC;
    void main(){ vec2 g=abs(fract(vW.xz/50.)-.5); float l=1.-smoothstep(0.0,0.012,min(g.x,g.y));
      vec2 g2=abs(fract(vW.xz/250.)-.5); float l2=1.-smoothstep(0.0,0.004,min(g2.x,g2.y));
      float r=length(vW.xz-uC.xz); float fade=exp(-r*r/(1400.*1400.));
      vec3 c=vec3(0.022,0.023,0.024)*fade + vec3(0.03)*l*fade*0.5 + vec3(0.03)*l2*fade;
      gl_FragColor=vec4(c,1.); }`,
});
const ground = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), groundMat); ground.rotation.x = -PI / 2; mapScene.add(ground);

function ribbon(pts, width, y) {
  const pos = [], dist = [], side = [], idx = []; let acc = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    const t = b.clone().sub(a); t.y = 0; t.normalize(); const n = v3(-t.z, 0, t.x).multiplyScalar(width / 2);
    if (i > 0) acc += pts[i].distanceTo(pts[i - 1]);
    pos.push(pts[i].x + n.x, y, pts[i].z + n.z, pts[i].x - n.x, y, pts[i].z - n.z);
    dist.push(acc, acc); side.push(1, -1);
    if (i < pts.length - 1) { const k = i * 2; idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('dist', new THREE.Float32BufferAttribute(dist, 1));
  g.setAttribute('side', new THREE.Float32BufferAttribute(side, 1));
  g.setIndex(idx); return g;
}
const ROADW = { motorway: 16, secondary: 10, unclassified: 7, service: 4.5 };
const ROADC = { motorway: '#6a6b6b', secondary: '#4f5050', unclassified: '#424343', service: '#353636' };
const roadMats = {};
for (const w of MAP.ways) {
  const hw = ROADW[w.hw] ? w.hw : 'service';
  roadMats[hw] = roadMats[hw] || new THREE.MeshBasicMaterial({ color: ROADC[hw], toneMapped: false, depthWrite: false, side: THREE.DoubleSide });
  const m = new THREE.Mesh(ribbon(w.p.map(p => P3(p)), ROADW[hw], hw === 'motorway' ? 0.3 : 0.2), roadMats[hw]); m.renderOrder = 1; mapScene.add(m);
}
// revealable route ribbons: white GPS trail and orange IDR trail
function trailMat(color, dashed = false) {
  return new THREE.ShaderMaterial({
    uniforms: { uA: { value: 0 }, uB: { value: 0 }, uC: { value: new THREE.Color(color) }, uI: { value: 1 } },
    transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
    vertexShader: `attribute float dist; attribute float side; varying float vD; varying float vS; void main(){ vD=dist; vS=side; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uA,uB,uI; uniform vec3 uC; varying float vD; varying float vS;
      void main(){ if(vD<uA||vD>uB) discard; float edge=1.-smoothstep(0.55,1.0,abs(vS));
        float head=smoothstep(uB-140.,uB,vD); float a=(0.55+0.45*head)*mix(0.35,1.,edge);
        ${dashed ? 'if(fract(vD/22.)>0.55) discard;' : ''}
        gl_FragColor=vec4(uC*(1.+head*1.2)*uI, a*uI); }`,
  });
}
const gpsTrailMat = trailMat('#4285f4');
const gpsTrail = new THREE.Mesh(ribbon(ROUTE, 7, 0.6), gpsTrailMat); gpsTrail.renderOrder = 3; mapScene.add(gpsTrail);
const idrTrailMat = trailMat('#f06131');
const idrTrail = new THREE.Mesh(ribbon(ROUTE, 9, 0.8), idrTrailMat); idrTrail.renderOrder = 4; mapScene.add(idrTrail);

// tunnel: roof band + portals
const TUN = ROUTE.slice(O0, O1 + 1);
const roofMat = new THREE.ShaderMaterial({
  uniforms: { uI: { value: 1 } }, transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
  vertexShader: `attribute float dist; attribute float side; varying float vD; varying float vS; void main(){ vD=dist; vS=side; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uI; varying float vD; varying float vS;
    void main(){ float hatch=step(0.5,fract((vD+vS*30.)/18.)); float edge=smoothstep(0.86,0.97,abs(vS));
      vec3 c=mix(vec3(0.06,0.062,0.065), vec3(0.09,0.09,0.095), hatch) + vec3(0.94,0.38,0.19)*edge*0.9;
      gl_FragColor=vec4(c*uI, (0.72+edge*0.28)*uI); }`,
});
const roof = new THREE.Mesh(ribbon(TUN, 64, 26), roofMat); roof.renderOrder = 6; mapScene.add(roof);
const portalMat = new THREE.MeshBasicMaterial({ color: ACCENT.clone().multiplyScalar(1.6), toneMapped: false });
for (const [p, q] of [[ROUTE[O0], ROUTE[O0 + 3]], [ROUTE[O1], ROUTE[O1 - 3]]]) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(66, 26, 3), portalMat);
  b.position.copy(p).setY(13); b.lookAt(q.clone().setY(13)); mapScene.add(b);
  const hole = new THREE.Mesh(new THREE.BoxGeometry(40, 18, 3.4), new THREE.MeshBasicMaterial({ color: 0x050505 }));
  hole.position.copy(b.position).setY(9); hole.quaternion.copy(b.quaternion); mapScene.add(hole);
}
// walls (for 3D read)
for (const s of [1, -1]) {
  const pts = TUN.map((p, i) => { const a = TUN[Math.max(0, i - 1)], b = TUN[Math.min(TUN.length - 1, i + 1)]; const t = b.clone().sub(a).normalize(); return p.clone().add(v3(-t.z, 0, t.x).multiplyScalar(32 * s)); });
  const g = ribbon(pts, 0.01, 0); const pa = g.attributes.position;
  for (let i = 0; i < pa.count; i++) pa.setY(i, i % 2 === 0 ? 26 : 0);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x0c0d0d, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }));
  m.renderOrder = 5; mapScene.add(m);
}

// dots
const glowTex = canvasTex(256, 256, (g) => { const gr = g.createRadialGradient(128, 128, 0, 128, 128, 128); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(255,255,255,0.45)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = gr; g.fillRect(0, 0, 256, 256); });
function makeDot(color, { ring = 0xffffff, hollow = false } = {}) {
  const g = new THREE.Group(); mapScene.add(g);
  const c = new THREE.Color(color);
  const halo = new THREE.Mesh(new THREE.CircleGeometry(1, 48), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.18, depthWrite: false, toneMapped: false }));
  halo.rotation.x = -PI / 2; halo.position.y = 1; halo.renderOrder = 7; g.add(halo);
  const outer = new THREE.Mesh(new THREE.RingGeometry(hollow ? 0.62 : 0.0, 1, 48), new THREE.MeshBasicMaterial({ color: ring, depthWrite: false, toneMapped: false, transparent: true }));
  outer.rotation.x = -PI / 2; outer.position.y = 1.2; outer.renderOrder = 8; g.add(outer);
  const inner = new THREE.Mesh(new THREE.CircleGeometry(0.72, 48), new THREE.MeshBasicMaterial({ color: c.clone().multiplyScalar(1.3), depthWrite: false, toneMapped: false, transparent: true }));
  inner.rotation.x = -PI / 2; inner.position.y = 1.4; inner.renderOrder = 9; inner.visible = !hollow; g.add(inner);
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: c, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, transparent: true }));
  glow.position.y = 2; glow.renderOrder = 10; g.add(glow);
  g.userData = { halo, outer, inner, glow };
  return g;
}
const gpsDot = makeDot('#4285f4');
const idrDot = makeDot('#f06131');
const truthDot = makeDot('#ffffff', { hollow: true });
const insDot = makeDot('#e5484d', { ring: 0xffd6d6 });
function setDot(d, pos, size, { halo = 3, glowI = 1, op = 1 } = {}) {
  d.visible = op > 0.001; d.position.copy(pos);
  const u = d.userData; u.outer.scale.setScalar(size); u.inner.scale.setScalar(size); u.halo.scale.setScalar(size * halo);
  u.glow.scale.setScalar(size * 9); u.glow.material.opacity = 0.55 * glowI * op;
  u.outer.material.opacity = op; u.inner.material.opacity = op; u.halo.material.opacity = 0.16 * op;
}

// classic INS (double integration) path: entry speed, accel bias, slow gyro drift
const V0 = MAP.speed[O0];
function insPath(b, drift) {
  const pts = [ENTRY.clone()]; let p = ENTRY.clone(), hdg = 0;
  for (let i = 1; i <= 300; i++) {
    const tt = i * 0.1, v = V0 + b * tt; hdg += drift * 0.1;
    const dir = FWD.clone().multiplyScalar(Math.cos(hdg)).add(SIDE.clone().multiplyScalar(Math.sin(hdg)));
    p = p.clone().add(dir.multiplyScalar(v * 0.1)); pts.push(p);
  }
  return pts;
}
let INS;
{ // tune the accel bias so the exit error matches the benchmark (792 m, README table 2)
  let lo = 0, hi = 4; const drift = THREE.MathUtils.degToRad(0.9);
  for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; const e = insPath(mid, drift)[300].distanceTo(EXIT); if (e < 792) lo = mid; else hi = mid; }
  INS = insPath((lo + hi) / 2, drift);
}
const insTrailMat = trailMat('#e5484d', true);
const insTrail = new THREE.Mesh(ribbon(INS, 6, 0.7), insTrailMat); insTrail.renderOrder = 4; mapScene.add(insTrail);
const INS_CUM = [0]; for (let i = 1; i < INS.length; i++) INS_CUM.push(INS_CUM[i - 1] + INS[i].distanceTo(INS[i - 1]));
const insAt = f => { const i = Math.floor(clamp(f, 0, 299.999)); return INS[i].clone().lerp(INS[i + 1], f - i); };
const SIDR = MAP.sidr;
const idrAt = f => { const i = Math.floor(clamp(f, 0, 299.999)); return routeAt(lerp(SIDR[i], SIDR[i + 1], f - i)); };

function mapCamFollow(target, h, back = 0.75, yaw = 0, fov = 35) {
  const f = FWD.clone().applyAxisAngle(v3(0, 1, 0), yaw);
  mapCam.position.copy(target).addScaledVector(f, -h * back).setY(h);
  mapCam.up.set(0, 1, 0); mapCam.lookAt(target); mapCam.fov = fov;
}
const toScreen = (p, cam) => { const q = p.clone().project(cam); return [(q.x * .5 + .5) * WIDTH, (-q.y * .5 + .5) * HEIGHT, q.z]; };


const CUTS = [[5.0, 'whip', 1], [14.0, 'zoom'], [17.0, 'whip', -1], [21.3, 'zoom'], [31.0, 'zoom'], [35.0, 'whip', 1]];
function resetPhone() {
  phone.position.set(0, 0, 0); phone.rotation.set(0, 0, 0); phone.visible = true;
  for (const k in layers) { layers[k].position.set(0, 0, 0); layers[k].rotation.set(0, 0, 0); layers[k].visible = true; }
  axes.visible = false; imuRing.visible = false; imuGlow.emissiveIntensity = 0;
}
function camWorld(pos, tgt, fov, roll = 0) { camera.position.copy(pos); camera.up.set(0, 1, 0); camera.lookAt(tgt); camera.rotateZ(roll); camera.fov = fov; }
function camLocal(pos, tgt, fov, roll = 0) {
  phone.updateMatrixWorld(true);
  camera.position.copy(phone.localToWorld(pos.clone())); camera.up.set(0, 1, 0);
  camera.lookAt(phone.localToWorld(tgt.clone())); camera.rotateZ(roll); camera.fov = fov;
}
// replay mapping (use case): outage sample index as a function of time
const RA = 21.5, RB = 31.0;
const repK = t => 300 * eio(lin(t, RA - 0.2, RB)) ;

function renderAt(t) {
  resetPhone();
  let bgI = 1, envI = 1, rim = 1, keyI = 1.2, hazeI = 0.5, partI = 0.7, scrB = 1, fade = 1, flash = 0, beamI = 0, flareI = 0, floorOn = false;
  let useMap = false;
  mScreen.map = screens['02_home'];
  // hide map dots by default
  for (const d of [gpsDot, idrDot, truthDot, insDot]) d.visible = false;
  gpsTrailMat.uniforms.uB.value = -1; idrTrailMat.uniforms.uB.value = -1; insTrailMat.uniforms.uB.value = -1;
  roofMat.uniforms.uI.value = 1;
  const dotSize = () => mapCam.position.distanceTo(mapCam.userData.tgt || ENTRY) * 0.0085;

  if (t < 5) {
    // ---------------- INTRO ----------------
    phone.rotation.set(0.04, -0.55, 0); phone.position.set(0.9, 0.05 * Math.sin(t * 1.3), 0);
    const k = eio(lin(t, 0, 5.2));
    camWorld(v3(0.25 * (1 - k), 0.25 - 0.1 * k, lerp(11, 7.0, k)), v3(0.2, 0.05, 0), 30);
    const bx = lerp(-9, 9, eio(lin(t, 0.4, 2.7)));
    beamMat.uniforms.uX.value = bx; beamI = env(t, 0.3, 0.7, 2.3, 3.1);
    beamLight.position.set(bx * 0.6, 0.5, -1.2); beamLight.intensity = beamI * 30;
    bgI = eo(lin(t, 0.5, 3.5)); envI = 0.15 + 0.85 * eo(lin(t, 1.5, 3.7));
    rim = eo(lin(t, 0.9, 2.5)); keyI = eo(lin(t, 2.3, 3.8)); hazeI = 0.5 * eo(lin(t, 1.4, 3.6)); partI = 0.75 * eo(lin(t, 1, 3.4));
    mScreen.map = screens['01_intro']; scrB = eo(lin(t, 3.2, 3.7));
    phone.rotation.y += 0.25 * eio(lin(t, 3, 5));
  } else if (t < 14) {
    // ---------------- PROBLEM (map) ----------------
    useMap = true;
    // car approaches with GPS until 8.0, then GPS is lost at the tunnel entry
    const fIdx = O0 - 150 + 150 * lin(t, 5.0, 8.0);
    const car = routeIdx(fIdx);
    const gpsOn = t < 8.0;
    let tgt, h;
    if (t < 10.4) { tgt = car.clone().addScaledVector(FWD, 120); h = lerp(420, 380, lin(t, 5, 10.4)); mapCamFollow(tgt, h, 0.8, 0.12 * Math.sin(t * 0.3)); }
    else {
      const k = eio(lin(t, 10.4, 11.4)); const insP = insAt(300 * ei(lin(t, 10.6, 13.6)));
      const mid = ENTRY.clone().lerp(insP, 0.5).addScaledVector(FWD, -60);
      tgt = vl(ENTRY.clone().addScaledVector(FWD, 120), mid, k); h = lerp(380, Math.max(420, ENTRY.distanceTo(insP) * 1.05), k);
      mapCamFollow(tgt, h, 0.8, 0.12 * Math.sin(t * 0.3));
    }
    mapCam.userData.tgt = tgt;
    const ds = mapCam.position.distanceTo(tgt) * 0.0085;
    gpsTrailMat.uniforms.uA.value = CUM[Math.max(0, O0 - 260)]; gpsTrailMat.uniforms.uB.value = CUM[Math.floor(Math.min(fIdx, O0))];
    gpsTrailMat.uniforms.uI.value = gpsOn ? 1 : 0.45;
    const frozenPulse = gpsOn ? 1 : 1 + 0.9 * eo(lin(t, 8.0, 9.2));
    setDot(gpsDot, car, ds, { halo: gpsOn ? 2.4 + 0.4 * Math.sin(t * 6) : 3 * frozenPulse, op: gpsOn ? 1 : 0.75 });
    if (!gpsOn) gpsDot.userData.inner.material.color.set('#7a8597'); else gpsDot.userData.inner.material.color.set('#4285f4').multiplyScalar(1.3);
    if (t > 10.6) {
      const f = 300 * ei(lin(t, 10.6, 13.6));
      setDot(insDot, insAt(f), ds, { halo: 2 });
      insTrailMat.uniforms.uA.value = 0; insTrailMat.uniforms.uB.value = INS_CUM[Math.floor(f)];
    }
    roofMat.uniforms.uI.value = 1;
  } else if (t < 21.3) {
    // ---------------- HOW IT WORKS (phone) ----------------
    if (t < 17.0) { // SENSE: x-ray IMU
      phone.rotation.set(0.08, PI + 0.3, 0); layers.back.visible = false;
      imuGlow.emissiveIntensity = 0.6 + 0.4 * Math.sin(t * 12);
      axes.visible = true; imuRing.visible = true;
      axes.scale.setScalar(Math.max(0.001, eo(lin(t, 14.1, 14.7))));
      axes.rotation.set(0.15 * Math.sin(t * 2), 0.2 * Math.sin(t * 1.5), 0);
      const rp = ((t - 14) * 1.4) % 1; imuRing.scale.setScalar(1 + rp * 1.3); imuRing.material.opacity = 1 - rp;
      const k = eio(lin(t, 14, 17));
      camLocal(vl(v3(-0.4, 0.15, -1.6), v3(-0.5, 0.45, -1.05), k), IMU.clone().add(v3(0.32, 0.12, 0)), 28, lerp(0.12, 0.0, k));
      hazeI = 0.25; partI = 0.6;
    } else { // THINK: phone right, data panel left
      phone.rotation.set(0.02, -0.42 + 0.12 * eio(lin(t, 17, 21.3)), 0.0);
      phone.position.set(1.55, 0.0, 0); phone.position.y += 0.04 * Math.sin(t * 1.4);
      mScreen.map = screens['03_tunnel_test'];
      const k = eio(lin(t, 17, 21.3));
      camWorld(v3(lerp(0.3, 0.0, k), 0.15, lerp(7.6, 7.0, k)), v3(0.25, 0.05, 0), 30);
      hazeI = 0.35; partI = 0.6;
    }
  } else if (t < 35) {
    // ---------------- SNAP + USE CASE (map replay) + RESULT ----------------
    useMap = true;
    const f = repK(t); // outage sample 0..300
    const idrP = idrAt(f), truP = routeIdx(O0 + f), insP = insAt(f);
    let tgt, h, back = 0.8, yaw = 0;
    if (t < 25.6) { // close follow behind IDR dot, roof faded so we can see inside
      tgt = idrP.clone().addScaledVector(FWD, 30).addScaledVector(SIDE, -55); h = lerp(230, 260, lin(t, 21.3, 25.6)); back = 1.0; yaw = 0.35 * (1 - eio(lin(t, 21.3, 25.6)));
      roofMat.uniforms.uI.value = 0.25;
    } else if (t < 31.0) { // crane up to the wide comparison
      const k = eio(lin(t, 25.6, 27.4));
      const wideT = ENTRY.clone().lerp(insAt(300), 0.48);
      tgt = vl(idrP.clone().addScaledVector(FWD, 30).addScaledVector(SIDE, -55), wideT, k); h = lerp(260, 1350, k); back = lerp(1.0, 0.75, k);
      roofMat.uniforms.uI.value = lerp(0.25, 1, k);
    } else { // zoom into the exit for the error callout
      const k = eio(lin(t, 31.0, 32.2));
      const wideT = ENTRY.clone().lerp(insAt(300), 0.48);
      tgt = vl(wideT, EXIT.clone().lerp(idrP, 0.5).addScaledVector(FWD, -55), k); h = lerp(1350, 240, k) - 25 * lin(t, 32.2, 35); back = lerp(0.75, 0.8, k); yaw = -0.5 * k;
      roofMat.uniforms.uI.value = 1;
    }
    mapCamFollow(tgt, h, back, yaw); mapCam.userData.tgt = tgt;
    const ds = Math.max(4, mapCam.position.distanceTo(tgt) * 0.0085);
    setDot(idrDot, idrP, ds, { halo: 2.6 + 0.4 * Math.sin(t * 7) });
    idrTrailMat.uniforms.uA.value = D_ENTRY - 60; idrTrailMat.uniforms.uB.value = lerp(SIDR[0], SIDR[300], f / 300);
    gpsTrailMat.uniforms.uA.value = CUM[Math.max(0, O0 - 260)]; gpsTrailMat.uniforms.uB.value = D_ENTRY; gpsTrailMat.uniforms.uI.value = 0.55;
    const cmp = t >= 25.6 ? eo(lin(t, 25.8, 26.6)) : 0;
    if (cmp > 0) {
      setDot(truthDot, truP.clone().setY(2), ds * 1.15, { halo: 0, glowI: 0.3, op: cmp });
      setDot(insDot, insP, ds, { halo: 2, op: cmp });
      insTrailMat.uniforms.uA.value = 0; insTrailMat.uniforms.uB.value = INS_CUM[Math.floor(Math.min(299, f))]; insTrailMat.uniforms.uI.value = cmp;
    }
    if (t > 31) flash = Math.max(0, 1 - Math.abs(t - 31.02) / 0.12) * 0.3;
  } else {
    // ---------------- OUTRO hero ----------------
    floorOn = true; mScreen.map = screens['03_tunnel_test'];
    const land = 35.62;
    const fall = ei(lin(t, 35.0, land));
    const settle = t > land ? 0.05 * Math.exp(-(t - land) * 9) * Math.sin((t - land) * 34) : 0;
    phone.position.set(-1.35, lerp(3.4, 0, fall) + Math.abs(settle), 0);
    phone.rotation.set(0, lerp(-0.9, 0.42, eo(lin(t, 35.0, 36.0))), 0);
    const k = eio(lin(t, 35.0, 40));
    camWorld(v3(lerp(-0.2, 0.35, k), lerp(-0.25, -0.5, k), lerp(9.4, 8.3, k)), v3(0.35, -0.15, 0), 30);
    phone.updateMatrixWorld(true);
    flareI = Math.max(0, 1 - Math.max(0, t - land) / 0.9) * (t > land - 0.03 ? 1 : 0) * 1.4 + 0.1 * env(t, 36.4, 37.2, 38.6, 39.5);
    flare.position.copy(phone.localToWorld(v3(W / 2, -H / 2 + 0.05, 0.08)));
    if (t > 36.4) flare.position.copy(phone.localToWorld(v3(W / 2 + 0.01, lerp(-H / 2, H / 2, eio(lin(t, 36.4, 39.5))), 0.08)));
    flash = Math.max(0, 1 - Math.abs(t - land - 0.02) / 0.1) * 0.35;
    hazeI = 0.7; partI = 0.75; rim = 1.15;
    fade = 1 - eio(lin(t, 39.35, 40.0));
    sweep.intensity = 6 * env(t, 36.4, 37, 38.6, 39.5); sweep.position.copy(phone.localToWorld(v3(2.2, lerp(-2, 2, lin(t, 36.4, 39.5)), 1.5)));
  }

  // whip / zoom transitions
  const activeCam = useMap ? mapCam : camera;
  let blurDir = new THREE.Vector2(), radial = 0;
  for (const [c, type, dir] of CUTS) {
    const pre = ei(lin(t, c - 0.16, c)) * (t < c ? 1 : 0);
    const post = (1 - eo(lin(t, c, c + 0.24))) * (t >= c ? 1 : 0);
    const kk = Math.max(pre, post); if (kk <= 0) continue;
    if (type === 'whip') { activeCam.rotateY(-(t < c ? 1 : -1) * dir * kk * 0.42); blurDir.x += dir * kk * 0.11; }
    else { activeCam.fov *= (t < c ? 1 - 0.35 * kk : 1 - 0.25 * kk); radial += kk * 0.16; }
    flash = Math.max(flash, (t >= c ? post : 0) * 0.1);
  }
  camera.updateProjectionMatrix(); mapCam.updateProjectionMatrix(); fitBg();
  camera.updateMatrixWorld(true); mapCam.updateMatrixWorld(true);

  // globals
  renderPass.scene = useMap ? mapScene : scene; renderPass.camera = useMap ? mapCam : camera;
  groundMat.uniforms.uC.value.copy(mapCam.userData.tgt || ENTRY);
  bgMat.uniforms.uI.value = bgI; bgMat.uniforms.uGlowI.value = hazeI;
  scene.environmentIntensity = envI;
  rimL.intensity = 260 * rim; rimR.intensity = 200 * rim; rimTop.intensity = 120 * rim;
  rimL.target.position.copy(phone.position); rimR.target.position.copy(phone.position); rimTop.target.position.copy(phone.position);
  key.intensity = keyI;
  hazeMat.uniforms.uI.value = hazeI; haze.position.x = phone.position.x;
  pMat.uniforms.uT.value = t; pMat.uniforms.uI.value = partI; pMat.uniforms.uFocus.value = camera.position.distanceTo(phone.position);
  beamMat.uniforms.uI.value = beamI; beam.visible = beamI > 0;
  if (t >= 5) beamLight.intensity = 0;
  if (t < 35) sweep.intensity = 0;
  mScreen.color.setScalar(scrB * 0.8);
  floor.visible = floorOn;
  flare.visible = flareI > 0.001; flare.material.opacity = Math.min(1, flareI); flare.scale.set(9 * (0.6 + 0.4 * Math.min(1, flareI)), 2.2, 1);
  bloom.strength = useMap ? 0.6 : 0.5 + flash * 0.8;
  const fu = finalPass.uniforms;
  fu.uDir.value.copy(blurDir); fu.uRadial.value = radial; fu.uBlur.value = (Math.abs(blurDir.x) + radial) > 0.002 ? 1 : 0;
  fu.uFlash.value = Math.min(0.9, flash); fu.uFade.value = fade; fu.uTime.value = t;

  // screen-space anchors and live numbers for the React overlays
  const fp = 300 * ei(lin(t, 10.6, 13.6));
  const fr = repK(t);
  return {
    useMap, fade,
    gps: toScreen(gpsDot.position, mapCam), ins: toScreen(insDot.position, mapCam), idr: toScreen(idrDot.position, mapCam),
    tunMid: toScreen(ENTRY.clone().lerp(EXIT, 0.5).setY(30), mapCam),
    insOff: Math.round(insAt(fp).distanceTo(routeIdx(O0 + fp))),
    replay: { seconds: fr / 10, speedKmh: Math.round(MAP.vest[O0 + Math.floor(Math.min(299, fr))]), distanceM: Math.round(lerp(SIDR[0], SIDR[300], fr / 300) - SIDR[0]) },
    sensorCursor: O0 + 40 + (t - 17.0) * 22,
  };
}

  return { MAP, update: renderAt, draw: () => composer.render(), dispose: () => renderer.dispose() };
}
