// Details that make the hall read as a real plant. Nothing here stands on the open floor:
// it hangs from the roof, sits on the walls, or stays outside, so the planner's grid
// still describes every obstacle.
import * as THREE from "three";
import { corrugatedTexture, rollerDoorTexture, textPlate } from "./textures.js";
import { tube } from "./geom.js";

const steel = new THREE.MeshStandardMaterial({ color: "#5f6670", metalness: 0.6, roughness: 0.45 });
const galv = new THREE.MeshStandardMaterial({ color: "#b9c0c7", metalness: 0.8, roughness: 0.3 });
const craneYellow = new THREE.MeshStandardMaterial({ color: "#f0b400", metalness: 0.35, roughness: 0.45 });
const red = new THREE.MeshStandardMaterial({ color: "#c41e24", metalness: 0.2, roughness: 0.45 });
const black = new THREE.MeshStandardMaterial({ color: "#1b1d20", roughness: 0.7 });

function box(w, h, d, mat, p) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.copy(p);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

function plate(tex, w, h, p, lookAt, emissive = 0) {
  const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 });
  if (emissive) { mat.emissive = new THREE.Color("#ffffff"); mat.emissiveMap = tex; mat.emissiveIntensity = emissive; }
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  m.position.copy(p);
  m.lookAt(lookAt);
  return m;
}

export const ridgeOf = (scene) => {
  const H = scene.height, rise = Math.min(2.5, H * 0.06);
  return (y) => scene.hall_height + rise * (1 - Math.abs(y - H / 2) / (H / 2));
};

// Pitched metal roof with a skylight strip on each side, gutters, downpipes, a facade
// sign and closed roller doors (drawn on the wall: no opening, so no change for the model).
export function envelope(scene, F) {
  const roof = new THREE.Group(), walls = new THREE.Group();
  const ridge = ridgeOf(scene), W = F.W, H = F.H, T = 0.4;
  const sheet = new THREE.MeshStandardMaterial({ map: corrugatedTexture("#8f979f"), metalness: 0.5, roughness: 0.5, side: THREE.DoubleSide });
  const sky = new THREE.MeshStandardMaterial({ color: "#e8f2fa", transparent: true, opacity: 0.35, roughness: 0.1, side: THREE.DoubleSide, depthWrite: false });
  const quad = (pts, mat, uvScale) => {
    const g = new THREE.BufferGeometry();
    const [a, b, c, d] = pts;
    g.setAttribute("position", new THREE.Float32BufferAttribute([a, b, c, a, c, d].flatMap((v) => [v.x, v.y, v.z]), 3));
    const [u, v] = uvScale;
    g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, u, 0, u, v, 0, 0, u, v, 0, v], 2));
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, mat);
    m.receiveShadow = true;
    return m;
  };
  for (const [y0, y1] of [[-T, H / 2], [H + T, H / 2]]) {
    const ys = [y0, y0 + (y1 - y0) * 0.55, y0 + (y1 - y0) * 0.8, y1];  // metal, skylight, metal
    ys.slice(0, -1).forEach((ya, k) => {
      const yb = ys[k + 1];
      const pts = [[-T, ya], [W + T, ya], [W + T, yb], [-T, yb]].map(([x, y]) => F.P(x, y, ridge(Math.min(Math.max(y, 0), H)) + 0.05));
      const len = Math.abs(yb - ya);
      const m = quad(pts, k === 1 ? sky : sheet.clone(), [(W + 2 * T) / 4, len / 4]);
      if (k !== 1) { m.material.map = m.material.map.clone(); m.material.map.repeat.set(1, 1); m.material.map.needsUpdate = true; }
      roof.add(m);
    });
  }
  for (const y of [-T - 0.15, H + T + 0.15]) {  // gutters and downpipes
    roof.add(box(W + 2 * T, 0.18, 0.22, galv, F.P(W / 2, y, scene.hall_height - 0.05)));
    for (let x = 3; x < W; x += 12) walls.add(tube(F.P(x, y, 0.2), F.P(x, y, scene.hall_height - 0.1), 0.06, galv, 10));
  }
  const sign = plate(textPlate("HALL A", "#13202d", "#f2c418", 512, 128, 84), 7, 1.75,
    F.P(W / 2, -T - 0.05, scene.hall_height - 1.4), F.P(W / 2, -50, scene.hall_height - 1.4), 0.25);
  walls.add(sign);
  const roller = new THREE.MeshStandardMaterial({ map: rollerDoorTexture(), metalness: 0.55, roughness: 0.4 });
  for (const side of ["south", "north"]) {  // one loading door per long side, away from exits
    const y = side === "south" ? -T / 2 : H + T / 2;
    const exits = scene.exits.filter((e) => e.side === side).map((e) => e.xy[0]);
    const x = [W * 0.3, W * 0.7, W * 0.5, W * 0.15, W * 0.85].find((c) => exits.every((ex) => Math.abs(ex - c) > 6));
    if (x === undefined) continue;
    for (const off of [-T / 2 - 0.04, T / 2 + 0.04]) walls.add(box(4.2, 4.6, 0.06, roller, F.P(x, y + off, 2.3)));
    walls.add(box(4.6, 0.6, T + 0.3, steel, F.P(x, y, 4.9)));
  }
  roof.traverse((o) => { if (o.isMesh) o.castShadow = false; });
  return { roof, walls };
}

// Overhead travelling crane on runway beams, ventilation ducts, sprinkler mains, cable
// trays, wall cabinets and fire extinguishers.
export function interior(scene, F) {
  const g = new THREE.Group(), W = F.W, H = F.H;
  const rail = 7.2;
  for (const y of [0.7, H - 0.7]) {
    g.add(box(W, 0.45, 0.3, steel, F.P(W / 2, y, rail)));
    for (let x = 0; x <= W + 0.01; x += 6) g.add(box(0.25, 0.25, 0.9, steel, F.P(Math.min(x, W), y < H / 2 ? 0.35 : H - 0.35, rail - 0.2)));
  }
  const crane = new THREE.Group();
  crane.add(box(1.0, 0.9, H - 1.2, craneYellow, new THREE.Vector3(0, rail + 0.65, 0)));
  for (const s of [-1, 1]) crane.add(box(1.8, 0.6, 0.6, craneYellow, new THREE.Vector3(0, rail + 0.45, s * (H / 2 - 0.7))));
  const trolley = new THREE.Group();
  trolley.add(box(1.6, 0.7, 1.4, black, new THREE.Vector3(0, rail + 1.35, 0)));
  trolley.add(tube(new THREE.Vector3(0, rail + 1.0, 0), new THREE.Vector3(0, rail - 0.4, 0), 0.03, black, 6));
  trolley.add(box(0.5, 0.6, 0.35, craneYellow, new THREE.Vector3(0, rail - 0.7, 0)));
  crane.add(trolley);
  crane.add(plate(textPlate("SWL 10 t", "#f0b400", "#111", 256, 96, 52), 1.2, 0.45,
    new THREE.Vector3(0, rail + 0.65, -0.52), new THREE.Vector3(0, rail + 0.65, -5)));
  crane.userData.crane = { W, H, trolley };
  g.add(crane);
  for (const y of [H * 0.32, H * 0.68]) {  // round ducts with diffusers
    g.add(tube(F.P(0.5, y, 8.1), F.P(W - 0.5, y, 8.1), 0.32, galv, 20));
    for (let x = 4; x < W; x += 8) g.add(box(0.6, 0.25, 0.6, galv, F.P(x, y, 7.7)));
  }
  for (let y = 4; y < H; y += 8) {  // sprinkler mains
    g.add(tube(F.P(0.3, y, 8.7), F.P(W - 0.3, y, 8.7), 0.05, red, 8));
    for (let x = 3; x < W; x += 6) g.add(tube(F.P(x, y, 8.7), F.P(x, y, 8.45), 0.03, red, 6));
  }
  for (const y of [0.25, H - 0.25]) {  // cable trays with cables
    g.add(box(W - 1, 0.08, 0.35, galv, F.P(W / 2, y, 5.0)));
    g.add(box(W - 1, 0.07, 0.25, black, F.P(W / 2, y, 5.07)));
  }
  const cabinet = new THREE.MeshStandardMaterial({ color: "#c7ccd1", metalness: 0.4, roughness: 0.45 });
  const led = new THREE.MeshStandardMaterial({ color: "#3cff7a", emissive: "#20ff60", emissiveIntensity: 1.4 });
  const farFromExits = (x, y) => scene.exits.every((e) => Math.hypot(e.xy[0] - x, e.xy[1] - y) > 4);
  for (const [x, y, nx] of [[0.2, H * 0.3, 1], [0.2, H * 0.75, 1], [W - 0.2, H * 0.3, -1], [W - 0.2, H * 0.7, -1]]) {
    if (!farFromExits(x, y)) continue;
    g.add(box(0.35, 1.9, 1.3, cabinet, F.P(x, y, 1.25)));
    g.add(box(0.02, 0.06, 0.06, led, F.P(x + nx * 0.18, y - 0.4, 2.0)));
  }
  const sign = textPlate("FIRE", "#c41e24", "#ffffff", 256, 128, 64);
  for (let x = 6; x < W; x += 12) {  // extinguishers on the long walls
    for (const [y, ny] of [[0.2, 1], [H - 0.2, -1]]) {
      if (!farFromExits(x, y)) continue;
      g.add(tube(F.P(x, y + ny * 0.1, 0.5), F.P(x, y + ny * 0.1, 1.1), 0.09, red, 14));
      g.add(plate(sign, 0.4, 0.2, F.P(x, y + ny * 0.03, 1.6), F.P(x, y + ny * 5, 1.6)));
    }
  }
  return g;
}

// Asphalt apron with parking lines, grass beyond, trees and an assembly point.
export function exterior(scene, F) {
  const g = new THREE.Group(), W = F.W, H = F.H, A = 16;
  const asphalt = new THREE.Mesh(new THREE.PlaneGeometry(W + 2 * A, H + 2 * A).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: "#3f4246", roughness: 0.95 }));
  asphalt.position.y = -0.02;
  asphalt.receiveShadow = true;
  g.add(asphalt);
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(W + 400, H + 400).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: "#5f7a45", roughness: 1 }));
  grass.position.y = -0.05;
  grass.receiveShadow = true;
  g.add(grass);
  const line = new THREE.MeshBasicMaterial({ color: "#e9e9e9" });
  for (let x = 4; x < W - 4; x += 3) {  // parking bays along the south apron
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.12, 5).rotateX(-Math.PI / 2), line);
    m.position.copy(F.P(x, -A + 4, 0.0));
    g.add(m);
  }
  const e = scene.exits[0];
  const out = { west: [-1, 0], east: [1, 0], south: [0, -1], north: [0, 1] }[e.side];
  const ap = [e.xy[0] + out[0] * 11, e.xy[1] + out[1] * 11];
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.2, 2.6, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: "#1aa34a" }));
  ring.position.copy(F.P(ap[0], ap[1], 0.02));
  g.add(ring);
  g.add(tube(F.P(ap[0], ap[1], 0), F.P(ap[0], ap[1], 2.6), 0.05, steel, 8));
  const s = plate(textPlate("ASSEMBLY POINT", "#13843f", "#ffffff", 512, 160, 52), 1.6, 0.5, F.P(ap[0], ap[1], 2.6), F.P(W / 2, H / 2, 2.6));
  g.add(s);
  const trunk = new THREE.MeshStandardMaterial({ color: "#5a4030", roughness: 1 });
  const leaf = new THREE.MeshStandardMaterial({ color: "#3e6b35", roughness: 0.9, flatShading: true });
  const rnd = (() => { let k = 11; return () => ((k = (k * 16807) % 2147483647) / 2147483647); })();
  for (let i = 0; i < 26; i++) {
    const side = i % 4, t = rnd();
    const [x, y] = [[-A - 6, t * H], [W + A + 6, t * H], [t * W, -A - 6], [t * W, H + A + 6]][side];
    const h = 4 + rnd() * 4;
    g.add(tube(F.P(x, y, 0), F.P(x, y, h * 0.45), 0.18, trunk, 8));
    const crown = new THREE.Mesh(new THREE.IcosahedronGeometry(h * 0.33, 1), leaf);
    crown.position.copy(F.P(x, y, h * 0.7));
    crown.castShadow = true;
    g.add(crown);
  }
  return g;
}

export function animateDetails(root, t) {
  root.traverse((o) => {
    const c = o.userData.crane;
    if (!c) return;
    o.position.x = (c.W / 2 - 4) * Math.sin(t * 0.05);
    c.trolley.position.z = (c.H / 2 - 4) * Math.sin(t * 0.11);
  });
}

// Workers: hard hat, hi-vis vest, dark trousers; instanced so hundreds stay cheap.
export function people(scene, F) {
  const n = scene.people.length;
  const g = new THREE.Group();
  if (!n) return g;
  const parts = [
    [new THREE.CylinderGeometry(0.13, 0.15, 0.85, 10).translate(0, 0.43, 0), "#2b3440"],  // legs
    [new THREE.CapsuleGeometry(0.2, 0.42, 4, 10).translate(0, 1.18, 0), "#ff7a00"],  // vest
    [new THREE.SphereGeometry(0.12, 14, 10).translate(0, 1.62, 0), "#d9a07a"],  // head
    [new THREE.SphereGeometry(0.15, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2).translate(0, 1.66, 0), "#f4f4f0"],  // helmet
  ];
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1);
  const rnd = (() => { let k = 3; return () => ((k = (k * 16807) % 2147483647) / 2147483647); })();
  const poses = scene.people.map(([x, y]) => {
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rnd() * Math.PI * 2);
    return m.clone().compose(F.P(x, y, 0), q.clone(), s.clone().setScalar(0.92 + rnd() * 0.16));
  });
  for (const [geo, color] of parts) {
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.7 }), n);
    poses.forEach((p, i) => mesh.setMatrixAt(i, p));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    g.add(mesh);
  }
  g.userData.count = n;
  return g;
}
