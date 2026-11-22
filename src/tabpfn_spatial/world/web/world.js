// Build the 3D factory from the exported scene. Every obstacle comes from the planner's
// factory: machine footprints, inner wall cells, outer walls with exit gaps.
import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { envelope, exterior, interior, people } from "./details.js";
import { tube } from "./geom.js";
import { buildMachine } from "./machines.js";
import { Agvs, HeightZone } from "./robots.js";
import * as shell from "./shell.js";
import { blockTexture, corrugatedTexture, floorTexture, signTexture } from "./textures.js";

export { tube };
export const DOOR_W = 2.4;  // m, clear width of an exit door opening
const DOOR_H = 3.2;
const WALL_T = 0.4;
const PLINTH = 1.2;
const WIN0 = 5.6, WIN1 = 7.0;  // window band

export function makeFrame(scene) {
  const W = scene.width, H = scene.height;
  const P = (x, y, z = 0) => new THREE.Vector3(x - W / 2, z, H / 2 - y);
  const toFactory = (v) => [v.x + W / 2, H / 2 - v.z];
  return { W, H, P, toFactory };
}

function label(text, cls) {
  const el = document.createElement("div");
  el.className = `label ${cls}`;
  el.textContent = text;
  return new CSS2DObject(el);
}

function shadows(obj, cast = true) {
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = cast; o.receiveShadow = true; } });
  return obj;
}

function floor(scene, F) {
  const g = new THREE.Group();
  const slab = new THREE.Mesh(
    new THREE.PlaneGeometry(F.W, F.H).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ map: floorTexture(scene), roughness: 0.82, metalness: 0.05 }),
  );
  slab.receiveShadow = true;
  g.add(slab);
  return g;
}

// Outer walls in segments around the exit gaps; returns {plinth, cladding, doors}.
function outerWalls(scene, F) {
  const concrete = new THREE.MeshStandardMaterial({ map: blockTexture("#9b9b97"), roughness: 0.9 });
  const clad = new THREE.MeshStandardMaterial({ map: corrugatedTexture(), metalness: 0.45, roughness: 0.45, side: THREE.DoubleSide });
  const frameMat = new THREE.MeshStandardMaterial({ color: "#13803c", metalness: 0.4, roughness: 0.5 });
  const sign = new THREE.MeshStandardMaterial({ map: signTexture("EXIT"), emissive: "#ffffff", emissiveMap: signTexture("EXIT"), emissiveIntensity: 0.9 });
  const glass = new THREE.MeshStandardMaterial({ color: "#9cc2d8", metalness: 0.2, roughness: 0.05, transparent: true, opacity: 0.38, depthWrite: false });
  const frameMetal = new THREE.MeshStandardMaterial({ color: "#3d444c", metalness: 0.6, roughness: 0.4 });
  const plinth = new THREE.Group(), cladding = new THREE.Group(), doors = new THREE.Group();
  const hall = scene.hall_height;
  const sides = {
    south: { len: F.W, at: (s, off) => [s, -WALL_T / 2 + off], along: "x" },
    north: { len: F.W, at: (s, off) => [s, F.H + WALL_T / 2 - off], along: "x" },
    west: { len: F.H, at: (s, off) => [-WALL_T / 2 + off, s], along: "y" },
    east: { len: F.H, at: (s, off) => [F.W + WALL_T / 2 - off, s], along: "y" },
  };
  for (const [side, S] of Object.entries(sides)) {
    const gaps = scene.exits.filter((e) => e.side === side)
      .map((e) => (S.along === "x" ? e.xy[0] : e.xy[1])).sort((a, b) => a - b);
    const pieces = [];
    let start = -WALL_T;
    for (const gp of gaps) { pieces.push([start, gp - DOOR_W / 2]); start = gp + DOOR_W / 2; }
    pieces.push([start, S.len + WALL_T]);
    const addBox = (group, s0, s1, y0, y1, mat) => {
      const len = s1 - s0;
      if (len <= 0.01) return;
      const [x, y] = S.at((s0 + s1) / 2, 0);
      const geo = S.along === "x" ? new THREE.BoxGeometry(len, y1 - y0, WALL_T) : new THREE.BoxGeometry(WALL_T, y1 - y0, len);
      const m = new THREE.Mesh(geo, mat.clone());
      if (m.material.map) { m.material.map = m.material.map.clone(); m.material.map.repeat.set(len / 4, (y1 - y0) / 4); m.material.map.needsUpdate = true; }
      m.position.copy(F.P(x, y, (y0 + y1) / 2));
      group.add(shadows(m));
    };
    for (const [a, b] of pieces) { addBox(plinth, a, b, 0, PLINTH, concrete); addBox(cladding, a, b, PLINTH, WIN0, clad); }
    addBox(cladding, -WALL_T, S.len + WALL_T, WIN1, hall, clad);
    addBox(cladding, -WALL_T, S.len + WALL_T, WIN0, WIN1, glass);
    for (let s = 0; s <= S.len; s += 1.5) addBox(cladding, s - 0.04, s + 0.04, WIN0, WIN1, frameMetal);
    for (const gp of gaps) {  // lintel over the door, green frame, lit EXIT sign inside
      addBox(cladding, gp - DOOR_W / 2, gp + DOOR_W / 2, DOOR_H, WIN0, clad);
      for (const s of [gp - DOOR_W / 2, gp + DOOR_W / 2]) {
        const [x, y] = S.at(s, 0);
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.16, DOOR_H, 0.16), frameMat);
        post.position.copy(F.P(x, y, DOOR_H / 2));
        doors.add(shadows(post));
      }
      const [hx, hy] = S.at(gp, 0);
      const head = new THREE.Mesh(S.along === "x" ? new THREE.BoxGeometry(DOOR_W + 0.3, 0.16, 0.2) : new THREE.BoxGeometry(0.2, 0.16, DOOR_W + 0.3), frameMat);
      head.position.copy(F.P(hx, hy, DOOR_H));
      doors.add(head);
      const [sx, sy] = S.at(gp, WALL_T / 2 + 0.06);
      const plate = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 0.44), sign);
      plate.position.copy(F.P(sx, sy, DOOR_H + 0.45));
      plate.lookAt(F.P(F.W / 2, F.H / 2, DOOR_H + 0.45));
      doors.add(plate);
    }
  }
  return { plinth, cladding, doors };
}

// Portal frames every 6 m with a pitched roof, purlins, high-bay lamps; roof panels apart.
function roofFrame(scene, F) {
  const steel = new THREE.MeshStandardMaterial({ color: "#5f6670", metalness: 0.6, roughness: 0.45 });
  const lampMat = new THREE.MeshStandardMaterial({ color: "#fff8e8", emissive: "#fff4dc", emissiveIntensity: 2.5 });
  const frame = new THREE.Group();
  const hall = scene.hall_height, rise = Math.min(2.5, F.H * 0.06);
  const ridge = (y) => hall + rise * (1 - Math.abs(y - F.H / 2) / (F.H / 2));
  const xs = [];
  for (let x = 0; x <= F.W + 0.01; x += 6) xs.push(Math.min(x, F.W));
  if (xs.at(-1) < F.W - 0.5) xs.push(F.W);
  for (const x of xs) {
    for (const y of [-WALL_T / 2, F.H + WALL_T / 2]) frame.add(tube(F.P(x, y, 0), F.P(x, y, hall), 0.22, steel, 6));
    frame.add(tube(F.P(x, -WALL_T / 2, hall), F.P(x, F.H / 2, hall + rise), 0.2, steel, 6));
    frame.add(tube(F.P(x, F.H + WALL_T / 2, hall), F.P(x, F.H / 2, hall + rise), 0.2, steel, 6));
  }
  for (let y = 0; y <= F.H + 0.01; y += 4) frame.add(tube(F.P(0, y, ridge(y)), F.P(F.W, y, ridge(y)), 0.07, steel, 6));
  for (let x = 3; x < F.W; x += 6) {
    for (const y of [F.H * 0.25, F.H * 0.75]) {
      const z = ridge(y) - 1.4;
      frame.add(tube(F.P(x, y, z), F.P(x, y, ridge(y)), 0.02, steel, 4));
      const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.45, 0.35, 20, 1, true), steel);
      shade.position.copy(F.P(x, y, z));
      frame.add(shade);
      const bulb = new THREE.Mesh(new THREE.CircleGeometry(0.4, 20).rotateX(Math.PI / 2), lampMat);
      bulb.position.copy(F.P(x, y, z - 0.17));
      frame.add(bulb);
    }
  }
  frame.traverse((o) => { if (o.isMesh) o.castShadow = false; });
  return { frame };
}

function innerWalls(scene, F) {
  const mat = new THREE.MeshStandardMaterial({ map: blockTexture("#a3a29c"), roughness: 0.92 });
  const g = new THREE.Group(), c = scene.cell, h = 6;
  for (const run of scene.walls) {
    const [x0, y0] = run.from, [x1, y1] = run.to;
    const xa = Math.min(x0, x1) - c / 2, xb = Math.max(x0, x1) + c / 2;
    const ya = Math.min(y0, y1) - c / 2, yb = Math.max(y0, y1) + c / 2;
    const m = new THREE.Mesh(new THREE.BoxGeometry(xb - xa, h, yb - ya), mat.clone());
    m.material.map = m.material.map.clone();
    m.material.map.repeat.set(Math.max(xb - xa, yb - ya) / 4, h / 4);
    m.position.copy(F.P((xa + xb) / 2, (ya + yb) / 2, h / 2));
    g.add(shadows(m));
  }
  return g;
}

// Yellow overhead gas pipes (DIN 2403 colour for gas), valves at fittings, drops into
// the machines they feed, hangers to the roof.
function pipes(scene, F) {
  const g = new THREE.Group();
  const gas = new THREE.MeshStandardMaterial({ color: "#f2c418", metalness: 0.35, roughness: 0.4 });
  const valveMat = new THREE.MeshStandardMaterial({ color: "#2b2e33", metalness: 0.6, roughness: 0.4 });
  const wheel = new THREE.MeshStandardMaterial({ color: "#c8202a", roughness: 0.5 });
  const rod = new THREE.MeshStandardMaterial({ color: "#555b63", metalness: 0.6, roughness: 0.5 });
  const z = scene.pipe_height, r = 0.11;
  const joints = new Set();
  for (const run of scene.pipes) {
    const a = F.P(...run.from, z), b = F.P(...run.to, z);
    if (a.distanceTo(b) > 0.01) g.add(tube(a, b, r, gas));
    for (const p of [run.from, run.to]) joints.add(p.join(","));
    const len = Math.hypot(run.to[0] - run.from[0], run.to[1] - run.from[1]);
    for (let s = 1.5; s < len; s += 3) {
      const t = s / len;
      const x = run.from[0] + t * (run.to[0] - run.from[0]), y = run.from[1] + t * (run.to[1] - run.from[1]);
      g.add(tube(F.P(x, y, z), F.P(x, y, scene.hall_height), 0.018, rod, 4));
    }
  }
  for (const j of joints) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(r * 1.15, 16, 10), gas);
    s.position.copy(F.P(...j.split(",").map(Number), z));
    g.add(s);
  }
  const c = scene.cell;
  const outsideAt = (x, y) => {  // off the grid or an outside cell: the building's edge
    const ix = Math.floor(x / c), iy = Math.floor(y / c);
    if (ix < 0 || iy < 0 || iy >= scene.solid.length || ix >= scene.solid[0].length) return true;
    return Boolean(scene.outside && scene.outside[iy][ix]);
  };
  for (const [x, y] of scene.fittings) {
    if ([[c, 0], [-c, 0], [0, c], [0, -c]].some(([dx, dy]) => outsideAt(x + dx, y + dy))) {
      g.add(tube(F.P(x, y, 0.15), F.P(x, y, z), r * 1.15, gas));  // supply riser from the ground
      const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.12, 16), valveMat);
      flange.position.copy(F.P(x, y, 1.2));
      g.add(flange);
    }
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.34, 16), valveMat);
    body.position.copy(F.P(x, y, z));
    g.add(body);
    const hw = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.025, 8, 24), wheel);
    hw.rotation.x = Math.PI / 2;
    hw.position.copy(F.P(x, y, z + 0.38));
    g.add(hw);
    const m = scene.machines.find((mm) => x > mm.rect[0] - 2.2 && x < mm.rect[2] + 2.2 && y > mm.rect[1] - 2.2 && y < mm.rect[3] + 2.2);
    if (m) {  // drop down and feed the machine
      const tx = Math.min(Math.max(x, m.rect[0] + 0.3), m.rect[2] - 0.3);
      const ty = Math.min(Math.max(y, m.rect[1] + 0.3), m.rect[3] - 0.3);
      const low = Math.min(1.6, m.height * 0.6);
      g.add(tube(F.P(x, y, z), F.P(x, y, low), r * 0.8, gas));
      g.add(tube(F.P(x, y, low), F.P(tx, ty, low), r * 0.8, gas));
      const elbow = new THREE.Mesh(new THREE.SphereGeometry(r * 0.95, 12, 8), gas);
      elbow.position.copy(F.P(x, y, low));
      g.add(elbow);
    }
  }
  return shadows(g);
}

// Gas detectors as installed in plants: bracket-mounted on the face of the wall or machine
// next to them, or hung from the roof on a drop rod; mounted high for light gases, low for
// heavy ones (scene.sensor_mounts, from the exporter). No floor poles.
function sensors(scene, F) {
  const g = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: "#3a3f45", metalness: 0.5, roughness: 0.5 });
  const head = new THREE.MeshStandardMaterial({ color: "#f5d000", roughness: 0.5 });
  const c = scene.cell;
  const mounts = scene.sensor_mounts || scene.sensors.map(() => ({ kind: "ceiling", dir: [0, 0], z: 4.0 }));
  scene.sensors.forEach(([x, y], i) => {
    const m = mounts[i], s = new THREE.Group();
    let px = x, py = y;
    if (m.kind === "wall") {  // on the wall face: half a cell towards the solid, minus the bracket
      px = x + m.dir[0] * (c / 2 - 0.12);
      py = y + m.dir[1] * (c / 2 - 0.12);
      const plate = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.42, 0.3), metal);
      plate.position.copy(F.P(x + m.dir[0] * (c / 2 - 0.02), y + m.dir[1] * (c / 2 - 0.02), m.z));
      plate.scale.set(m.dir[0] ? 0.15 : 1, 1, m.dir[1] ? 0.15 : 1);
      s.add(plate);
    } else {  // hung from the roof structure
      s.add(tube(F.P(x, y, scene.hall_height), F.P(x, y, m.z + 0.2), 0.02, metal, 6));
    }
    const h = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.3, 0.16), head);
    h.position.copy(F.P(px, py, m.z));
    s.add(h);
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.08, 0.12, 12), metal);
    cone.position.copy(F.P(px, py, m.z - 0.2));  // sensing head points down
    s.add(cone);
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 6),
      new THREE.MeshStandardMaterial({ color: "#2bff6a", emissive: "#14ff55", emissiveIntensity: 1.6 }));
    led.position.copy(F.P(px, py, m.z + 0.2));
    led.userData.sensor = i;
    s.add(led);
    const l = label(`S${i}`, "sensor");
    l.position.copy(F.P(px, py, m.z + 0.5));
    s.add(l);
    g.add(shadows(s));
  });
  return g;
}

// The planner's solid cells as a translucent red overlay (proof that 3D = model).
function plannerGrid(scene, F) {
  const out = scene.outside;  // outside the building: solid too, but not drawn
  const solid = scene.solid.flatMap((row, iy) => row.map((v, ix) => (v && !(out && out[iy][ix]) ? [ix, iy] : null))).filter(Boolean);
  const c = scene.cell;
  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(c * 0.96, 0.12, c * 0.96),
    new THREE.MeshBasicMaterial({ color: "#ff2d55", transparent: true, opacity: 0.4, depthWrite: false, depthTest: false }),
    solid.length,
  );
  mesh.renderOrder = 10;  // drawn over the machines: what the planner sees as solid
  const m = new THREE.Matrix4();
  solid.forEach(([ix, iy], k) => { m.setPosition(F.P((ix + 0.5) * c, (iy + 0.5) * c, 0.08)); mesh.setMatrixAt(k, m); });
  const g = new THREE.Group();
  g.add(mesh);
  const lines = new THREE.GridHelper(Math.max(F.W, F.H), Math.max(F.W, F.H) / c, "#ff2d55", "#ff8fa6");
  lines.position.set((Math.max(F.W, F.H) - F.W) / 2, 0.02, -(Math.max(F.W, F.H) - F.H) / 2);
  lines.material.transparent = true;
  lines.material.opacity = 0.25;
  g.add(lines);
  return g;
}

export function buildWorld(scene) {
  const F = makeFrame(scene);
  const root = new THREE.Group();
  const labels = new THREE.Group();
  const shaped = Boolean(scene.shaped);
  const empty = () => new THREE.Group();
  const parts = {
    floor: shaped ? shell.floorAndYard(scene, F) : floor(scene, F),
    outer: shaped ? shell.outerWalls(scene, F) : outerWalls(scene, F),
    roof: shaped ? shell.roofs(scene, F) : roofFrame(scene, F),
    shell: shaped ? { walls: empty(), roof: empty() } : envelope(scene, F),
    overhead: shaped ? empty() : interior(scene, F),
    yard: shaped ? empty() : exterior(scene, F),
    people: people(scene, F),
    inner: innerWalls(scene, F),
    machines: new THREE.Group(),
    pipes: pipes(scene, F),
    sensors: sensors(scene, F),
    grid: plannerGrid(scene, F),
    labels,
  };
  for (const m of scene.machines) {
    parts.machines.add(buildMachine(m, F.P));
    const l = label(`${m.name}${m.hot ? "  🔥" : ""}`, m.hot ? "machine hot" : "machine");
    const [x0, y0, x1, y1] = m.rect;
    l.position.copy(F.P((x0 + x1) / 2, (y0 + y1) / 2, m.height + 1.2));
    labels.add(l);
  }
  scene.exits.forEach((e, i) => {
    const l = label(`E${i} exit`, "exit");
    l.position.copy(F.P(...e.xy, 4.2));
    labels.add(l);
  });
  parts.outer.cladding.add(parts.shell.walls);
  if (!shaped) parts.roof.panels = parts.shell.roof;
  parts.agvs = new Agvs(scene, F);
  parts.zone = new HeightZone(F);
  parts.grid.visible = false;
  parts.roof.panels.visible = false;
  root.add(parts.floor, parts.yard, parts.outer.plinth, parts.outer.cladding, parts.outer.doors,
    parts.roof.frame, parts.roof.panels, parts.overhead, parts.inner, parts.machines, parts.pipes,
    parts.sensors, parts.people, parts.grid, parts.agvs.group, parts.zone.group, labels);
  return { root, parts, F };
}
