// Procedural machine models. Each stays inside its exact 2D footprint (w x d), so what
// you see is what the planner treats as solid. Local frame: x along the footprint
// width, z along its depth, y up, origin at the footprint centre on the floor.
import * as THREE from "three";
import { blockTexture, corrugatedTexture, hazardStripes } from "./textures.js";

const M = {
  steel: new THREE.MeshStandardMaterial({ color: "#c9ced3", metalness: 0.7, roughness: 0.32 }),
  dark: new THREE.MeshStandardMaterial({ color: "#3b3f44", metalness: 0.6, roughness: 0.5 }),
  frame: new THREE.MeshStandardMaterial({ color: "#2a2d31", metalness: 0.5, roughness: 0.6 }),
  blue: new THREE.MeshStandardMaterial({ color: "#2f5d8a", metalness: 0.4, roughness: 0.45 }),
  yellow: new THREE.MeshStandardMaterial({ color: "#e3b505", metalness: 0.3, roughness: 0.5 }),
  green: new THREE.MeshStandardMaterial({ color: "#3d5a40", metalness: 0.3, roughness: 0.6 }),
  rubber: new THREE.MeshStandardMaterial({ color: "#151515", roughness: 0.9 }),
  glow: new THREE.MeshStandardMaterial({ color: "#ff7a1a", emissive: "#ff5a00", emissiveIntensity: 2.2 }),
  led: new THREE.MeshStandardMaterial({ color: "#3cff7a", emissive: "#20ff60", emissiveIntensity: 1.5 }),
};
let brick, ribs, stripes;
function lazyTextures() {
  if (brick) return;
  brick = new THREE.MeshStandardMaterial({ map: blockTexture("#5a2c22", true), roughness: 0.9 });
  ribs = new THREE.MeshStandardMaterial({ map: corrugatedTexture("#4b6b4e"), metalness: 0.3, roughness: 0.55 });
  stripes = new THREE.MeshStandardMaterial({ map: hazardStripes(), roughness: 0.7 });
}

function box(w, h, d, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y + h / 2, z);
  return m;
}

function cyl(r, h, mat, x = 0, y = 0, z = 0, axis = "y", seg = 32) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), mat);
  if (axis === "x") m.rotation.z = Math.PI / 2;
  if (axis === "z") m.rotation.x = Math.PI / 2;
  m.position.set(x, y, z);
  return m;
}

function plinth(g, w, d) {
  g.add(box(w, 0.15, d, stripes));
  g.add(box(w - 0.3, 0.2, d - 0.3, M.dark, 0, 0.05));
}

function tank(w, d, h) {
  const g = new THREE.Group();
  g.add(box(w, 0.6, d, new THREE.MeshStandardMaterial({ color: "#7d7f80", roughness: 0.95 })));  // bund
  const nx = Math.max(1, Math.round(w / 3)), nz = Math.max(1, Math.round(d / 3));
  const r = Math.min(w / nx, d / nz) * 0.42;
  const th = h - r - 0.6;
  const tops = [];
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      const x = -w / 2 + (i + 0.5) * (w / nx), z = -d / 2 + (j + 0.5) * (d / nz);
      g.add(cyl(r, th, M.steel, x, 0.6 + th / 2, z));
      const dome = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2), M.steel);
      dome.position.set(x, 0.6 + th, z);
      g.add(dome);
      for (const f of [0.3, 0.65]) {
        const band = new THREE.Mesh(new THREE.TorusGeometry(r + 0.02, 0.04, 8, 40), M.dark);
        band.rotation.x = Math.PI / 2;
        band.position.set(x, 0.6 + th * f, z);
        g.add(band);
      }
      tops.push([x, z]);
    }
  }
  const walkY = 0.6 + th + r * 0.6;  // catwalk over the tank tops
  const long = w >= d;
  g.add(box(long ? w * 0.9 : 0.9, 0.08, long ? 0.9 : d * 0.9, M.frame, 0, walkY));
  const L = (long ? w : d) * 0.9;
  for (const s of [-0.45, 0.45]) {  // railings: posts every 1.5 m, knee and top rails
    for (const h of [0.5, 1.0]) g.add(box(long ? L : 0.04, 0.04, long ? 0.04 : L, M.yellow, long ? 0 : s, walkY + h, long ? s : 0));
    for (let a = -L / 2; a <= L / 2 + 1e-6; a += 1.5) g.add(box(0.04, 1.04, 0.04, M.yellow, long ? a : s, walkY, long ? s : a));
  }
  const [lx, lz] = tops[0];  // ladder
  for (const s of [-0.22, 0.22]) g.add(box(0.05, walkY, 0.05, M.yellow, lx + s, 0, lz - r - 0.15));
  for (let y = 0.9; y < walkY; y += 0.35) g.add(box(0.44, 0.03, 0.03, M.yellow, lx, y, lz - r - 0.15));
  return g;
}

function furnace(w, d, h) {
  const g = new THREE.Group();
  plinth(g, w, d);
  const bw = w * 0.86, bd = d * 0.8, bh = h * 0.78;
  g.add(box(bw, bh, bd, brick, 0, 0.2));
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) g.add(box(0.3, bh + 0.3, 0.3, M.frame, sx * bw / 2, 0.2, sz * bd / 2));
  g.add(box(bw + 0.4, 0.35, bd + 0.4, M.frame, 0, 0.2 + bh));
  const door = box(bw * 0.4, bh * 0.35, 0.08, M.glow, 0, 0.6, bd / 2 + 0.02);
  door.userData.flicker = true;
  g.add(door);
  g.add(cyl(0.55, 9 - bh, M.dark, -bw * 0.25, 0.2 + bh + (9 - bh) / 2, 0));  // chimney to the roof
  const light = new THREE.PointLight("#ff6a1a", 6, 9, 2);
  light.position.set(0, 1.2, bd / 2 + 1);
  light.userData.flicker = true;
  g.add(light);
  return g;
}

function compressor(w, d, h) {
  const g = new THREE.Group();
  plinth(g, w, d);
  const long = w >= d, L = (long ? w : d) * 0.7, S = long ? d : w;
  const r = Math.min(S * 0.18, 0.8);
  for (const s of [-S * 0.22, S * 0.22]) {
    g.add(cyl(r, L, M.blue, long ? 0 : s, 0.35 + r, long ? s : 0, long ? "x" : "z"));
    for (const e of [-L / 2, L / 2]) {
      const cap = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 12), M.blue);
      cap.position.set(long ? e : s, 0.35 + r, long ? s : e);
      g.add(cap);
    }
  }
  g.add(box(long ? w * 0.25 : S * 0.7, h * 0.7, long ? S * 0.7 : d * 0.25, M.dark, long ? w * 0.36 : 0, 0.35, long ? 0 : d * 0.36));
  const fan = cyl(Math.min(S * 0.3, 1.0), 0.25, M.frame, long ? w * 0.36 : 0, 0.35 + h * 0.75, long ? 0 : d * 0.36);
  fan.userData.spin = true;
  g.add(fan);
  g.add(box(0.6, 1.8, 0.4, M.steel, long ? -w * 0.42 : S * 0.4, 0.2, long ? S * 0.4 : -d * 0.42));
  g.add(box(0.08, 0.08, 0.02, M.led, long ? -w * 0.42 : S * 0.4, 1.6, (long ? S * 0.4 : -d * 0.42) + 0.21));
  return g;
}

function press(w, d, h) {
  const g = new THREE.Group();
  plinth(g, w, d);
  const long = w >= d, L = long ? w : d, S = long ? d : w;
  const along = (a, s, y, dims, mat) => {
    const [lx, hy, sz] = dims;
    return box(long ? lx : sz, hy, long ? sz : lx, mat, long ? a : s, y, long ? s : a);
  };
  g.add(along(0, 0, 0.2, [L * 0.95, 0.7, S * 0.35], M.rubber));  // conveyor
  for (let a = -L * 0.45; a <= L * 0.45; a += 0.8) g.add(along(a, 0, 0.9, [0.1, 0.05, S * 0.36], M.steel));
  const n = Math.max(2, Math.floor(L / 4.5));
  for (let k = 0; k < n; k++) {
    const a = -L / 2 + (k + 0.5) * (L / n);
    for (const s of [-S * 0.38, S * 0.38]) g.add(along(a, s, 0.2, [0.6, h - 0.2, 0.5], M.yellow));
    g.add(along(a, 0, h - 0.6, [0.8, 0.6, S * 0.9], M.yellow));
    g.add(cyl(0.22, 1.0, M.steel, long ? a : 0, h - 1.0, long ? 0 : a));
  }
  return g;
}

function generator(w, d, h) {
  const g = new THREE.Group();
  plinth(g, w, d);
  g.add(box(w * 0.94, h * 0.85, d * 0.9, ribs, 0, 0.2));
  const long = w >= d;
  for (const s of [-0.25, 0.25]) {
    const x = long ? s * w : 0, z = long ? 0 : s * d;
    g.add(cyl(0.22, 1.6, M.dark, x, 0.2 + h * 0.85 + 0.8, z));
    g.add(cyl(0.3, 0.06, M.dark, x, 0.2 + h * 0.85 + 1.65, z));
  }
  g.add(box(long ? 0.1 : w * 0.7, h * 0.6, long ? d * 0.7 : 0.1, M.frame, long ? w * 0.47 : 0, 0.4, long ? 0 : d * 0.47));
  return g;
}

function drum(w, d, h) {
  const g = new THREE.Group();
  plinth(g, w, d);
  const long = w >= d, L = (long ? w : d) * 0.78, S = long ? d : w;
  const r = Math.min(S * 0.36, h * 0.4);
  const drumMesh = cyl(r, L, M.steel, 0, 0.3 + r + 0.3, 0, long ? "x" : "z", 40);
  drumMesh.userData.roll = long ? "x" : "z";
  g.add(drumMesh);
  for (const a of [-L * 0.3, L * 0.3]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r + 0.06, 0.1, 10, 48), M.dark);
    if (long) ring.rotation.y = Math.PI / 2;
    ring.position.set(long ? a : 0, 0.3 + r + 0.3, long ? 0 : a);
    g.add(ring);
    g.add(box(long ? 0.5 : S * 0.8, 0.6, long ? S * 0.8 : 0.5, M.frame, long ? a : 0, 0.2, long ? 0 : a));
  }
  const burner = box(long ? (w - L) * 0.45 : S * 0.6, h * 0.7, long ? S * 0.6 : (d - L) * 0.45, brick,
    long ? -(L / 2 + (w - L) * 0.25) : 0, 0.2, long ? 0 : -(L / 2 + (d - L) * 0.25));
  g.add(burner);
  return g;
}

function generic(w, d, h) {
  const g = new THREE.Group();
  plinth(g, w, d);
  g.add(box(w * 0.85, h * 0.8, d * 0.8, M.dark, 0, 0.2));
  g.add(box(0.6, 1.6, 0.3, M.steel, w * 0.3, 0.2, d * 0.42));
  return g;
}

const BUILDERS = { tank, furnace, compressor, press, generator, drum, machine: generic };

// Machine at its footprint; hot ones get an orange beacon. Everything casts shadows.
export function buildMachine(m, P) {
  lazyTextures();
  const [x0, y0, x1, y1] = m.rect;
  const w = x1 - x0, d = y1 - y0;
  const g = (BUILDERS[m.kind] || generic)(w, d, m.height);
  if (m.hot) {
    const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 8), M.glow);
    beacon.position.set(w / 2 - 0.3, m.height + 0.4, d / 2 - 0.3);
    beacon.userData.pulse = true;
    g.add(beacon);
  }
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  g.position.copy(P((x0 + x1) / 2, (y0 + y1) / 2));
  g.userData.machine = m;
  return g;
}

// Animate flicker, spin, roll and pulse markers.
export function animateMachines(root, t) {
  root.traverse((o) => {
    const u = o.userData;
    if (u.flicker && o.isLight) o.intensity = 5 + 2 * Math.sin(t * 13) * Math.sin(t * 7.3);
    if (u.flicker && o.isMesh) o.material.emissiveIntensity = 1.8 + 0.5 * Math.sin(t * 11);
    if (u.spin) o.rotation.y = t * 8;
    if (u.roll === "x") o.rotation.x = t * 0.8;
    if (u.roll === "z") o.rotation.y = t * 0.8;  // spin about its own axis
    if (u.pulse) o.scale.setScalar(1 + 0.25 * Math.max(0, Math.sin(t * 4)));
  });
}
