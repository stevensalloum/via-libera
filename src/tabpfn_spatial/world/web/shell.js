// Building shell of a shaped plant (L, U, T halls, corridors, side rooms): floor per wing,
// outer walls along the exported outline with door openings at the exits, a portal-frame
// roof structure per wing. Like the rectangular hall, nothing here stands on the open floor,
// so the planner's grid still describes every obstacle.
import * as THREE from "three";
import { tube } from "./geom.js";
import { blockTexture, corrugatedTexture, floorTexture, signTexture } from "./textures.js";

const WALL_T = 0.4, PLINTH = 1.2, WIN0 = 5.6, WIN1 = 7.0, DOOR_W = 2.4, DOOR_H = 3.2;

function shadows(obj) {
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return obj;
}

function textured(mat, len, h) {
  const m = mat.clone();
  if (m.map) { m.map = m.map.clone(); m.map.repeat.set(len / 4, h / 4); m.map.needsUpdate = true; }
  return m;
}

export function floorAndYard(scene, F) {
  const g = new THREE.Group();
  const yard = new THREE.Mesh(new THREE.PlaneGeometry(F.W + 60, F.H + 60).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: "#5b5e61", roughness: 0.95 }));
  yard.position.copy(F.P(F.W / 2, F.H / 2, -0.03));
  yard.receiveShadow = true;
  g.add(yard);
  const tex = floorTexture(scene);
  for (const [x0, y0, x1, y1] of scene.wings) {
    const mat = new THREE.MeshStandardMaterial({ map: tex.clone(), roughness: 0.82, metalness: 0.05 });
    mat.map.repeat.set((x1 - x0) / F.W, (y1 - y0) / F.H);
    mat.map.needsUpdate = true;
    const slab = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0).rotateX(-Math.PI / 2), mat);
    slab.position.copy(F.P((x0 + x1) / 2, (y0 + y1) / 2, 0));
    slab.receiveShadow = true;
    g.add(slab);
  }
  return g;
}

// Exits on a run: the exit cell's outer edge lies on the run's line, within its span.
function gapsOn(run, scene) {
  const c = scene.cell, horizontal = run.from[1] === run.to[1];
  const lo = Math.min(horizontal ? run.from[0] : run.from[1], horizontal ? run.to[0] : run.to[1]);
  const hi = Math.max(horizontal ? run.from[0] : run.from[1], horizontal ? run.to[0] : run.to[1]);
  const line = horizontal ? run.from[1] : run.from[0];
  return scene.exits.filter((e) => e.normal[0] === run.normal[0] && e.normal[1] === run.normal[1])
    .map((e) => [e.xy[0] + e.normal[0] * c / 2, e.xy[1] + e.normal[1] * c / 2])
    .filter(([x, y]) => Math.abs((horizontal ? y : x) - line) < 1e-6 && (horizontal ? x : y) > lo && (horizontal ? x : y) < hi)
    .map(([x, y]) => (horizontal ? x : y)).sort((a, b) => a - b);
}

export function outerWalls(scene, F) {
  const concrete = new THREE.MeshStandardMaterial({ map: blockTexture("#9b9b97"), roughness: 0.9 });
  const clad = new THREE.MeshStandardMaterial({ map: corrugatedTexture(), metalness: 0.45, roughness: 0.45, side: THREE.DoubleSide });
  const glass = new THREE.MeshStandardMaterial({ color: "#9cc2d8", metalness: 0.2, roughness: 0.05, transparent: true, opacity: 0.38, depthWrite: false });
  const frameMat = new THREE.MeshStandardMaterial({ color: "#13803c", metalness: 0.4, roughness: 0.5 });
  const sign = new THREE.MeshStandardMaterial({ map: signTexture("EXIT"), emissive: "#ffffff", emissiveMap: signTexture("EXIT"), emissiveIntensity: 0.9 });
  const plinth = new THREE.Group(), cladding = new THREE.Group(), doors = new THREE.Group();
  const hall = scene.hall_height;
  for (const run of scene.outline) {
    const horizontal = run.from[1] === run.to[1];
    const a = horizontal ? Math.min(run.from[0], run.to[0]) : Math.min(run.from[1], run.to[1]);
    const b = horizontal ? Math.max(run.from[0], run.to[0]) : Math.max(run.from[1], run.to[1]);
    const line = (horizontal ? run.from[1] : run.from[0]) + (horizontal ? run.normal[1] : run.normal[0]) * WALL_T / 2;
    const at = (s) => (horizontal ? [s, line] : [line, s]);
    const addBox = (group, s0, s1, z0, z1, mat) => {
      const len = s1 - s0;
      if (len <= 0.01) return;
      const geo = horizontal ? new THREE.BoxGeometry(len, z1 - z0, WALL_T) : new THREE.BoxGeometry(WALL_T, z1 - z0, len);
      const m = new THREE.Mesh(geo, textured(mat, len, z1 - z0));
      m.position.copy(F.P(...at((s0 + s1) / 2), (z0 + z1) / 2));
      group.add(shadows(m));
    };
    const gaps = gapsOn(run, scene);
    let start = a - WALL_T / 2;
    const pieces = [];
    for (const gp of gaps) { pieces.push([start, gp - DOOR_W / 2]); start = gp + DOOR_W / 2; }
    pieces.push([start, b + WALL_T / 2]);
    for (const [s0, s1] of pieces) { addBox(plinth, s0, s1, 0, PLINTH, concrete); addBox(cladding, s0, s1, PLINTH, WIN0, clad); }
    addBox(cladding, a - WALL_T / 2, b + WALL_T / 2, WIN1, hall, clad);
    addBox(cladding, a - WALL_T / 2, b + WALL_T / 2, WIN0, WIN1, glass);
    for (const gp of gaps) {
      addBox(cladding, gp - DOOR_W / 2, gp + DOOR_W / 2, DOOR_H, WIN0, clad);
      for (const s of [gp - DOOR_W / 2, gp + DOOR_W / 2]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.16, DOOR_H, 0.16), frameMat);
        post.position.copy(F.P(...at(s), DOOR_H / 2));
        doors.add(shadows(post));
      }
      const inward = horizontal ? [gp, line - run.normal[1] * (WALL_T / 2 + 0.06)] : [line - run.normal[0] * (WALL_T / 2 + 0.06), gp];
      const plate = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 0.44), sign);
      plate.position.copy(F.P(...inward, DOOR_H + 0.45));
      plate.lookAt(F.P(inward[0] - run.normal[0], inward[1] - run.normal[1], DOOR_H + 0.45));
      doors.add(plate);
    }
  }
  return { plinth, cladding, doors };
}

// Columns along each wing's long sides every 6 m, roof beams across, high-bay lamps; flat
// roof panels per wing (hidden by default, like the rectangular hall).
export function roofs(scene, F) {
  const steel = new THREE.MeshStandardMaterial({ color: "#5f6670", metalness: 0.6, roughness: 0.45 });
  const lampMat = new THREE.MeshStandardMaterial({ color: "#fff8e8", emissive: "#fff4dc", emissiveIntensity: 2.5 });
  const panelMat = new THREE.MeshStandardMaterial({ map: corrugatedTexture("#8d949b"), metalness: 0.5, roughness: 0.45, side: THREE.DoubleSide });
  const frame = new THREE.Group(), panels = new THREE.Group();
  const hall = scene.hall_height, c = scene.cell, eps = 0.05;
  const outsideCell = (x, y) => {
    const ix = Math.floor(x / c), iy = Math.floor(y / c);
    if (ix < 0 || iy < 0 || iy >= scene.outside.length || ix >= scene.outside[0].length) return true;
    return Boolean(scene.outside[iy][ix]);
  };
  // a column may only stand on the outline: some cell touching the point is outside
  const onOutline = (x, y) => [[-eps, -eps], [eps, -eps], [-eps, eps], [eps, eps]].some(([dx, dy]) => outsideCell(x + dx, y + dy));
  for (const [x0, y0, x1, y1] of scene.wings) {
    const alongX = x1 - x0 >= y1 - y0;
    const L0 = alongX ? x0 : y0, L1 = alongX ? x1 : y1, S0 = alongX ? y0 : x0, S1 = alongX ? y1 : x1;
    const P = (l, s, z) => (alongX ? F.P(l, s, z) : F.P(s, l, z));
    for (let l = L0; l <= L1 + 0.01; l += 6) {
      const ll = Math.min(l, L1);
      for (const s of [S0, S1]) {
        const [px, py] = alongX ? [ll, s] : [s, ll];
        if (onOutline(px, py)) frame.add(tube(P(ll, s, 0), P(ll, s, hall), 0.2, steel, 6));
      }
      frame.add(tube(P(ll, S0, hall), P(ll, S1, hall), 0.16, steel, 6));
    }
    for (let s = S0 + 4; s < S1; s += 4) frame.add(tube(P(L0, s, hall + 0.3), P(L1, s, hall + 0.3), 0.06, steel, 6));
    for (let l = L0 + 3; l < L1; l += 6) {
      for (let s = S0 + 5; s < S1 - 2; s += 10) {
        const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.45, 0.35, 20, 1, true), steel);
        shade.position.copy(P(l, s, hall - 1.2));
        frame.add(shade);
        const bulb = new THREE.Mesh(new THREE.CircleGeometry(0.4, 20).rotateX(Math.PI / 2), lampMat);
        bulb.position.copy(P(l, s, hall - 1.37));
        frame.add(bulb);
        frame.add(tube(P(l, s, hall - 1.2), P(l, s, hall), 0.02, steel, 4));
      }
    }
    const mat = panelMat.clone();
    mat.map = mat.map.clone(); mat.map.repeat.set((x1 - x0) / 4, (y1 - y0) / 4); mat.map.needsUpdate = true;
    const roof = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0 + 0.4, 0.15, y1 - y0 + 0.4), mat);
    roof.position.copy(F.P((x0 + x1) / 2, (y0 + y1) / 2, hall + 0.45));
    panels.add(shadows(roof));
  }
  frame.traverse((o) => { if (o.isMesh) o.castShadow = false; });
  return { frame, panels };
}
