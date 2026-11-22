// Play a closed-loop run in the 3D world: true gas as layered translucent sheets, the
// planners' danger map, the locator's guesses, sensors changing colour, people moving
// (one planner at a time) and a timeline. Data: data/replay.json (uv run world --replay).
import * as THREE from "three";

const STATE_COLOURS = ["#ff7a00", "#ffd000", "#ff00d0"];  // walking, queueing, in gas
// The selected person, three ways (the 2D milestone's colours): [track, colour, meaning]
const POLICIES = [["ccrp", "#1baf7a", "replanned at every model update"],
  ["once", "#eda100", "planned once at the alarm"], ["nearest", "#e34948", "nearest exit, gas ignored"]];

function b64(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function loadReplay(base = "data/") {
  try {
    const r = await fetch(`${base}replay.json`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

// value 0..255 on the log scale -> RGBA; 128 is the toxic limit
function gasColour(v, out, i, alphaScale) {
  if (v < 64) { out[i + 3] = 0; return; }  // below ~0.01: not drawn (keeps the hall readable)
  const t = (v - 64) / 191;  // 0 at 0.01, ~0.33 at the toxic limit 0.1, 1 at 10
  const [r, g, b] = t < 0.33
    ? [150 + 300 * t, 225 - 30 * t, 120 - 300 * t]  // pale green -> yellow
    : [255, 215 - 300 * (t - 0.33), 25 + 40 * (t - 0.33)];  // yellow -> orange -> red
  out[i] = r; out[i + 1] = Math.max(0, g); out[i + 2] = Math.max(0, b);
  out[i + 3] = Math.min(225, 40 + 380 * t) * alphaScale;
}

export class Replay {
  constructor(rep, world3d, F, parts) {
    this.rep = rep;
    this.F = F;
    this.parts = parts;
    const s = rep.scene;
    this.ny = s.solid.length;
    this.nx = s.solid[0].length;
    this.cell = s.cell;
    this.n = rep.people;
    this.t = 0;
    this.playing = false;
    this.speed = 3;
    this.planner = "ccrp";
    this.group = new THREE.Group();
    world3d.add(this.group);
    this.gasFrames = rep.gas.frames.map(b64);
    this.dangers = rep.updates.map((u) => b64(u.danger));
    this.tracks = {};
    for (const [p, d] of Object.entries(rep.planners)) {
      const raw = b64(d.cells);
      this.tracks[p] = { cells: new Int16Array(raw.buffer), state: b64(d.state), metrics: d.metrics,
        routes: (d.routes || []).map((u) => ({ t: u.t, people: new Int32Array(b64(u.people).buffer),
          offsets: new Int32Array(b64(u.offsets).buffer), cells: new Int16Array(b64(u.cells).buffer) })) };
    }
    this.selected = null;
    this.follow = false;
    this.personView = false;
    this.lastPathKey = null;
    this.pathGroup = new THREE.Group();
    world3d.add(this.pathGroup);
    this.ghosts = POLICIES.map(([, colour]) => {  // the selected person under the other policies
      const g = new THREE.Mesh(new THREE.CapsuleGeometry(0.3, 1.0, 4, 10).translate(0, 0.8, 0),
        new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.75, depthTest: false }));
      g.renderOrder = 21; g.visible = false;
      world3d.add(g);
      return g;
    });
    this.buildGas();
    this.buildDanger();
    this.buildMarkers();
    this.buildPeople();
    this.sensorLeds = [];
    parts.sensors.traverse((o) => { if (o.userData.sensor !== undefined) this.sensorLeds.push(o); });
    this.lastGas = -1;
    this.lastUpdate = -2;
  }

  sheet(y, opacity) {
    const data = new Uint8Array(this.nx * this.ny * 4);
    const tex = new THREE.DataTexture(data, this.nx, this.ny, THREE.RGBAFormat);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, opacity, side: THREE.DoubleSide });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(this.F.W, this.F.H).rotateX(-Math.PI / 2), mat);
    m.position.y = y;
    m.renderOrder = 5;
    this.group.add(m);
    return { tex, data, mesh: m };
  }

  buildGas() {  // three stacked sheets read as a low-lying cloud
    this.gas = [this.sheet(0.06, 1.0), this.sheet(0.9, 0.55), this.sheet(1.8, 0.3)];
  }

  buildDanger() {
    this.danger = this.sheet(0.1, 1.0);
    this.danger.mesh.visible = false;
  }

  buildMarkers() {
    const [lx, ly] = this.rep.leak.xy;
    const z = this.rep.leak.z ?? this.rep.scene.pipe_height;
    const n = 90;
    this.puff = { n, born: new Float32Array(n).map(() => Math.random() * 3), base: this.F.P(lx, ly, z) };
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    this.puffPoints = new THREE.Points(geo, new THREE.PointsMaterial({ color: "#e8f6ff", size: 0.35, transparent: true, opacity: 0.7, depthWrite: false }));
    this.group.add(this.puffPoints);
    const ringMat = new THREE.MeshBasicMaterial({ color: "#19e3ff", transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false });
    this.rings = [0, 1, 2].map(() => {
      const r = new THREE.Mesh(new THREE.RingGeometry(0.7, 0.95, 40).rotateX(-Math.PI / 2), ringMat.clone());
      r.position.y = 0.15;
      r.visible = false;
      this.group.add(r);
      return r;
    });
  }

  buildPeople() {
    const n = this.n;
    const body = new THREE.CapsuleGeometry(0.22, 0.9, 4, 10).translate(0, 0.75, 0);
    const head = new THREE.SphereGeometry(0.15, 12, 8).translate(0, 1.62, 0);
    this.bodies = new THREE.InstancedMesh(body, new THREE.MeshStandardMaterial({ roughness: 0.7 }), n);
    this.heads = new THREE.InstancedMesh(head, new THREE.MeshStandardMaterial({ color: "#f4f4f0", roughness: 0.5 }), n);
    for (const m of [this.bodies, this.heads]) { m.castShadow = true; m.frustumCulled = false; this.group.add(m); }
    const c = new THREE.Color("#ff7a00");
    for (let i = 0; i < n; i++) this.bodies.setColorAt(i, c);
    let k = 7;
    const rnd = () => ((k = (k * 16807) % 2147483647) / 2147483647);
    this.jitter = Array.from({ length: n }, () => [(rnd() - 0.5) * 1.3, (rnd() - 0.5) * 1.3]);
    this.colours = STATE_COLOURS.map((h) => new THREE.Color(h));
  }

  // Planned route of person i at update u of planner p (array of [iy, ix]) or null.
  routeOf(p, u, i) {
    const r = this.tracks[p].routes[u];
    if (!r) return null;
    const j = r.people.indexOf(i);
    if (j < 0) return null;
    const out = [];
    for (let k = r.offsets[j]; k < r.offsets[j + 1]; k++) out.push([r.cells[2 * k], r.cells[2 * k + 1]]);
    return out;
  }

  // Seconds person i spends in harmful gas under each planner (state 2 per second).
  gasSeconds(i) {
    const out = {};
    for (const [p, tr] of Object.entries(this.tracks)) {
      let s = 0;
      for (let k = 0; k * this.n + i < tr.state.length; k++) if (tr.state[k * this.n + i] === 2) s++;
      out[p] = s;
    }
    return out;
  }

  select(i) {
    this.selected = i;
    this.lastPathKey = "";
    this.yaw = undefined;
    if (i === null) { this.follow = false; this.personView = false; }
  }

  // The selected person under three policies, drawn live as they walk (as in the 2D
  // milestone): green = CCRP replanned at every model update, orange = CCRP planned once at
  // the alarm, red = nearest exit (gas ignored). Thick trail = walked so far; thin green =
  // the route the latest plan still asks for. Ghosts mark where the others are now.
  drawPaths(t) {
    const i = this.selected;
    const clear = () => { this.pathGroup.children.forEach((m) => m.geometry.dispose()); this.pathGroup.clear(); };
    if (i === null || i === undefined) { if (this.lastPathKey !== null) clear(); this.lastPathKey = null; this.ghosts.forEach((g) => { g.visible = false; }); return; }
    const k = this.frameIndex(t);
    const key = `${i}|${k}`;
    this.moveGhosts(t, i);
    if (key === this.lastPathKey) return;
    this.lastPathKey = key;
    clear();
    const c = this.cell, F = this.F;
    const line = (pts, colour, radius, opacity) => {
      if (pts.length < 2) return;
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, false, "catmullrom", 0), pts.length * 3, radius, 6, false);
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity, depthTest: false }));
      mesh.renderOrder = 20;
      this.pathGroup.add(mesh);
    };
    POLICIES.forEach(([p, colour], j) => {
      const tr = this.tracks[p];
      if (!tr) return;
      const lift = 0.3 + 0.12 * j, pts = [];
      for (let q = 0; q <= k; q++) {
        const [x, y] = this.personXY(tr, q, i, false);
        const last = pts.at(-1);
        if (!last || Math.hypot(last.x - F.P(x, y, lift).x, last.z - F.P(x, y, lift).z) > 0.4) pts.push(F.P(x, y, lift));
        if (tr.state[q * this.n + i] === 3) break;
      }
      line(pts, colour, 0.17, 0.95);
    });
    const live = this.tracks.ccrp;
    if (live && live.routes.length && live.state[k * this.n + i] !== 3) {  // plan ahead
      let u = -1;
      live.routes.forEach((r, q) => { if (r.t <= t + 1e-6) u = q; });
      const route = u >= 0 ? this.routeOf("ccrp", u, i) : null;
      if (route) {
        const [x, y] = this.personXY(live, k, i, false);
        let start = 0, best = 1e9;
        route.forEach(([iy, ix], q) => { const d = Math.hypot((ix + 0.5) * c - x, (iy + 0.5) * c - y); if (d < best) { best = d; start = q; } });
        const pts = [F.P(x, y, 0.2), ...route.slice(start + 1).map(([iy, ix]) => F.P((ix + 0.5) * c, (iy + 0.5) * c, 0.2))];
        line(pts, "#7dffc4", 0.08, 0.7);
      }
    }
  }

  moveGhosts(t, i) {
    const k0 = this.frameIndex(t), k1 = this.frameIndex(t + 1), a = t - Math.floor(t);
    POLICIES.forEach(([p], j) => {
      const g = this.ghosts[j], tr = this.tracks[p];
      g.visible = !!tr && p !== this.planner && tr.state[k0 * this.n + i] !== 3;
      if (!g.visible) return;
      const [x0, y0] = this.personXY(tr, k0, i), [x1, y1] = this.personXY(tr, k1, i);
      g.position.copy(this.F.P(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, 0));
    });
  }

  // Where person i is now (interpolated) and where they head (smoothed yaw).
  pose(i) {
    const track = this.tracks[this.planner];
    const k0 = this.frameIndex(this.t), k1 = this.frameIndex(this.t + 1), a = this.t - Math.floor(this.t);
    const [x0, y0] = this.personXY(track, k0, i), [x1, y1] = this.personXY(track, k1, i);
    const x = x0 + (x1 - x0) * a, y = y0 + (y1 - y0) * a;
    const [xa, ya] = this.personXY(track, this.frameIndex(this.t + 3), i);
    if (Math.hypot(xa - x, ya - y) > 0.3) {
      const target = Math.atan2(ya - y, xa - x);
      if (this.yaw === undefined) this.yaw = target;
      let d = target - this.yaw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      this.yaw += d * 0.08;
    } else if (this.yaw === undefined) this.yaw = Math.PI / 2;
    return [x, y, this.yaw];
  }

  // Follow: behind and above the person. Person view: their own eyes, looking where they go.
  followCamera(camera) {
    const i = this.selected;
    if (i === null || i === undefined) return false;
    const [x, y, yaw] = this.pose(i);
    const dx = Math.cos(yaw), dy = Math.sin(yaw);
    if (this.personView) {
      camera.position.copy(this.F.P(x, y, 1.65));
      camera.lookAt(this.F.P(x + 6 * dx, y + 6 * dy, 1.2));
    } else {
      camera.position.copy(this.F.P(x - 7 * dx, y - 7 * dy, 4.5));
      camera.lookAt(this.F.P(x + 4 * dx, y + 4 * dy, 1.0));
    }
    return true;
  }

  // Mini map overlay: true gas, predicted danger, leak guesses, detectors, people, paths.
  drawMini(g, ms, H) {
    const nx = this.nx, ny = this.ny;
    if (!this.miniCv) {
      this.miniCv = document.createElement("canvas");
      this.miniCv.width = nx; this.miniCv.height = ny;
      this.miniImg = this.miniCv.getContext("2d").createImageData(nx, ny);
      this.miniKey = "";
    }
    const gi = Math.min(this.gasFrames.length - 1, Math.floor(this.t / this.rep.gas.every));
    const key = `${gi}|${this.lastUpdate}|${this.showDangerMini}`;
    if (key !== this.miniKey) {
      this.miniKey = key;
      const d = this.miniImg.data, frame = this.gasFrames[gi];
      const dz = this.lastUpdate >= 0 && this.showDangerMini !== false ? this.dangers[this.lastUpdate] : null;
      for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
        const c = iy * nx + ix, o = ((ny - 1 - iy) * nx + ix) * 4, v = frame[c], p = dz ? dz[c] / 255 : 0;
        let r = 0, gg = 0, b = 0, al = 0;
        if (v >= 128) { r = 214; gg = 40; b = 210; al = 210; }  // toxic now
        else if (v >= 94) { r = 255; gg = 170; b = 0; al = 170; }  // above the alarm level
        else if (v >= 40) { r = 90; gg = 170; b = 255; al = 40 + v; }  // traces
        if (p >= 0.2 && al < 150) { r = 230; gg = 30; b = 50; al = Math.max(al, 60 + 140 * p); }  // predicted
        d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = al;
      }
      this.miniCv.getContext("2d").putImageData(this.miniImg, 0, 0);
    }
    g.imageSmoothingEnabled = false;
    g.drawImage(this.miniCv, 0, 0, nx * this.cell * ms, ny * this.cell * ms);
    const P = (x, y) => [x * ms, H - y * ms];
    const frame = this.gasFrames[gi];
    this.rep.scene.sensors.forEach(([x, y]) => {  // detectors: amber/red when reading high
      const v = frame[Math.floor(y / this.cell) * nx + Math.floor(x / this.cell)];
      if (v < 94) return;
      const [px, py] = P(x, y);
      g.fillStyle = v >= 128 ? "#ff1a1a" : "#ffb000";
      g.fillRect(px - 3, py - 3, 6, 6);
    });
    const alarm = this.rep.t_alarm !== null && this.t >= this.rep.t_alarm;
    if (alarm && this.lastUpdate >= 0) {
      this.rep.updates[this.lastUpdate].source_top.forEach((s, j) => {
        const [px, py] = P(...s.xy);
        g.strokeStyle = "#19e3ff"; g.lineWidth = j ? 1 : 2.5;
        g.beginPath(); g.arc(px, py, 4 + 10 * Math.sqrt(s.p), 0, 7); g.stroke();
      });
    }
    if (alarm) {  // true leak
      const [px, py] = P(...this.rep.leak.xy);
      g.strokeStyle = "#fff"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(px - 4, py - 4); g.lineTo(px + 4, py + 4); g.moveTo(px + 4, py - 4); g.lineTo(px - 4, py + 4); g.stroke();
    }
    const track = this.tracks[this.planner], k = this.frameIndex(this.t);
    for (let i = 0; i < this.n; i++) {
      const st = track.state[k * this.n + i];
      if (st === 3) continue;
      const [px, py] = P(...this.personXY(track, k, i));
      g.fillStyle = STATE_COLOURS[st];
      g.fillRect(px - 1.5, py - 1.5, 3, 3);
    }
    const i = this.selected;
    if (i === null || i === undefined) return;
    POLICIES.forEach(([p, colour]) => {
      const tr = this.tracks[p];
      if (!tr) return;
      g.strokeStyle = colour; g.lineWidth = 2.5; g.beginPath();
      for (let q = 0; q <= k; q++) {
        const [px, py] = P(...this.personXY(tr, q, i, false));
        q ? g.lineTo(px, py) : g.moveTo(px, py);
        if (tr.state[q * this.n + i] === 3) break;
      }
      g.stroke();
      const [px, py] = P(...this.personXY(tr, k, i, false));
      g.fillStyle = colour; g.beginPath(); g.arc(px, py, 4, 0, 7); g.fill();
    });
  }

  // Gas byte where person i stands under policy p, per second up to frame k.
  exposure(p, i, k) {
    const tr = this.tracks[p], out = [];
    if (!tr) return out;
    for (let q = 0; q <= k; q++) {
      const o = (q * this.n + i) * 2, gi = Math.min(this.gasFrames.length - 1, Math.floor(q / this.rep.gas.every));
      out.push(tr.state[q * this.n + i] === 3 ? null : this.gasFrames[gi][tr.cells[o] * this.nx + tr.cells[o + 1]]);
    }
    return out;
  }

  frameIndex(t) { return Math.max(0, Math.min(this.rep.times.length - 1, Math.floor(t))); }

  personXY(track, k, i, jitter = true) {
    const o = (k * this.n + i) * 2;
    const iy = track.cells[o], ix = track.cells[o + 1], j = jitter ? this.jitter[i] : [0, 0];
    return [(ix + 0.5) * this.cell + j[0], (iy + 0.5) * this.cell + j[1]];
  }

  applyGas(t) {
    const g = Math.min(this.gasFrames.length - 1, Math.floor(t / this.rep.gas.every));
    if (g === this.lastGas) return;
    this.lastGas = g;
    const frame = this.gasFrames[g];
    this.gas.forEach((layer, li) => {
      const scale = [1, 0.8, 0.6][li];
      for (let c = 0; c < frame.length; c++) gasColour(frame[c] * (li ? 0.92 : 1), layer.data, c * 4, scale);
      layer.tex.needsUpdate = true;
    });
    for (const led of this.sensorLeds) {  // detectors: green, amber above the alarm level, red above toxic
      const [x, y] = this.rep.scene.sensors[led.userData.sensor];
      const v = frame[Math.floor(y / this.cell) * this.nx + Math.floor(x / this.cell)];
      const col = v >= 128 ? "#ff1a1a" : v >= 94 ? "#ffb000" : "#14ff55";
      led.material.color.set(col);
      led.material.emissive.set(col);
    }
  }

  applyUpdate(t) {
    let u = -1;
    this.rep.updates.forEach((up, i) => { if (up.t <= t + 1e-6) u = i; });
    if (u === this.lastUpdate) return;
    this.lastUpdate = u;
    for (const r of this.rings) r.visible = false;
    const d = this.danger;
    if (u < 0) { d.data.fill(0); d.tex.needsUpdate = true; return; }
    const map = this.dangers[u];
    for (let c = 0; c < map.length; c++) {
      const p = map[c] / 255;
      d.data[c * 4] = 255; d.data[c * 4 + 1] = 20; d.data[c * 4 + 2] = 60;
      d.data[c * 4 + 3] = p >= 0.5 ? 150 : p * 160;
    }
    d.tex.needsUpdate = true;
    this.rep.updates[u].source_top.forEach((g, i) => {
      const r = this.rings[i];
      r.visible = this.showSource;
      r.position.copy(this.F.P(g.xy[0], g.xy[1], 0.15));
      r.scale.setScalar(0.6 + 1.8 * Math.sqrt(g.p));
      r.userData.p = g.p;
    });
  }

  applyPeople(t) {
    const track = this.tracks[this.planner];
    const k0 = this.frameIndex(t), k1 = this.frameIndex(t + 1), a = t - Math.floor(t);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
    const size = this.cinematic ? 2.2 : 1;  // seen from far away in videos: drawn larger
    const scale = new THREE.Vector3(size, size, size);
    for (let i = 0; i < this.n; i++) {
      const st = track.state[k0 * this.n + i];
      if (st === 3 || (this.personView && i === this.selected)) { this.bodies.setMatrixAt(i, hidden); this.heads.setMatrixAt(i, hidden); continue; }
      const [x0, y0] = this.personXY(track, k0, i), [x1, y1] = this.personXY(track, k1, i);
      const x = x0 + (x1 - x0) * a, y = y0 + (y1 - y0) * a;
      if (Math.hypot(x1 - x0, y1 - y0) > 0.05) q.setFromAxisAngle(up, Math.atan2(x1 - x0, -(y1 - y0)) + Math.PI);
      m.compose(this.F.P(x, y, 0), q, scale);
      this.bodies.setMatrixAt(i, m);
      this.heads.setMatrixAt(i, m);
      this.bodies.setColorAt(i, this.colours[st]);
    }
    this.bodies.instanceMatrix.needsUpdate = true;
    this.heads.instanceMatrix.needsUpdate = true;
    this.bodies.instanceColor.needsUpdate = true;
  }

  animatePuff(dt) {
    const pos = this.puffPoints.geometry.attributes.position, p = this.puff;
    const on = this.t > 0.5;
    this.puffPoints.visible = on;
    if (!on) return;
    for (let i = 0; i < p.n; i++) {
      p.born[i] += dt;
      if (p.born[i] > 3) p.born[i] = 0;
      const s = p.born[i], ang = i * 2.399;
      pos.setXYZ(i, p.base.x + Math.cos(ang) * s * 0.9, p.base.y - s * 0.9 + Math.sin(i) * 0.2, p.base.z + Math.sin(ang) * s * 0.9);
    }
    pos.needsUpdate = true;
  }

  stats() {
    const track = this.tracks[this.planner], k = this.frameIndex(this.t);
    let out = 0, gas = 0, queue = 0;
    for (let i = 0; i < this.n; i++) {
      const st = track.state[k * this.n + i];
      if (st === 3) out++; else if (st === 2) gas++; else if (st === 1) queue++;
    }
    return { out, gas, queue };
  }

  update(dt) {
    if (this.playing) {
      this.t = Math.min(this.rep.t_end, this.t + dt * this.speed);
      if (this.t >= this.rep.t_end) this.playing = false;
    }
    this.applyGas(this.t);
    this.applyUpdate(this.t);
    this.applyPeople(this.t);
    this.animatePuff(dt);
    const wall = performance.now() / 1000;
    if (this.rep.agvs) this.parts.agvs.show(this.rep.agvs, this.t, wall);
    const alarm = this.rep.t_alarm !== null && this.t >= this.rep.t_alarm;
    this.parts.machines.userData.stopped = alarm;  // robot arms and conveyors e-stop too
    this.parts.zone.show(this.rep.height_zone, alarm ? this.t : null);
    this.drawPaths(this.t);
    const pulse = 1 + 0.15 * Math.sin(performance.now() / 180);
    for (const r of this.rings) if (r.visible) r.material.opacity = 0.55 + 0.35 * Math.sin(performance.now() / 250);
    return pulse;
  }
}

const PLAN_COLOURS = { nearest: "#e34948", once: "#eda100", ccrp: "#1baf7a", risk: "#2a78d6" };
const PLAN_NAMES = { nearest: "nearest exit", once: "CCRP planned once", ccrp: "CCRP replanned live", risk: "risk-aware" };

// Per planner and second: % of people out, people standing in harmful gas (state 3 / 2).
function planCurves(replay) {
  const out = {}, n = replay.n;
  for (const [p, tr] of Object.entries(replay.tracks)) {
    const T = Math.floor(tr.state.length / n), o = new Float32Array(T), g = new Float32Array(T);
    for (let k = 0; k < T; k++) {
      let a = 0, b = 0;
      for (let i = 0; i < n; i++) { const st = tr.state[k * n + i]; if (st === 3) a++; else if (st === 2) b++; }
      o[k] = (100 * a) / n; g[k] = b;
    }
    out[p] = { out: o, gas: g };
  }
  return out;
}

function drawCurves(cv, curves, key, title, t, tEnd, yMax) {
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, L = 34, B = 18, T0 = 16;
  ctx.clearRect(0, 0, W, H);
  const top = yMax ?? Math.max(1, ...Object.values(curves).map((c) => Math.max(...c[key])));
  ctx.fillStyle = "#c9d1d9"; ctx.font = "11px system-ui"; ctx.fillText(title, L, 11);
  ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 1;
  for (const f of [0, 0.5, 1]) {
    const y = T0 + (1 - f) * (H - T0 - B);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - 4, y); ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.fillText(String(Math.round(f * top)), 4, y + 4);
  }
  ctx.fillText("0 s", L, H - 4); ctx.fillText(`${Math.round(tEnd)} s`, W - 40, H - 4);
  const X = (k) => L + (k / tEnd) * (W - 4 - L), Y = (v) => T0 + (1 - v / top) * (H - T0 - B);
  for (const [p, c] of Object.entries(curves)) {
    ctx.strokeStyle = PLAN_COLOURS[p] || "#ccc"; ctx.lineWidth = 2; ctx.beginPath();
    c[key].forEach((v, k) => (k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v))));
    ctx.stroke();
  }
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(t), T0); ctx.lineTo(X(t), H - B); ctx.stroke();
}

// The gas this person breathes under each policy (log scale), with the toxic limit.
function drawExposure(cv, replay, i, tEnd) {
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, L = 34, B = 18, T0 = 16;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#c9d1d9"; ctx.font = "11px system-ui"; ctx.fillText("gas where this person is (log)", L, 11);
  const X = (k) => L + (k / tEnd) * (W - 4 - L), Y = (v) => T0 + (1 - v / 255) * (H - T0 - B);
  for (const [lab, v] of [["toxic", 127.5], ["alarm", 94]]) {
    ctx.strokeStyle = lab === "toxic" ? "rgba(255,0,208,0.6)" : "rgba(255,176,0,0.5)"; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - 4, Y(v)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#8b949e"; ctx.fillText(lab, 2, Y(v) + 4);
  }
  ctx.fillStyle = "#8b949e"; ctx.fillText("0 s", L, H - 4); ctx.fillText(`${Math.round(tEnd)} s`, W - 40, H - 4);
  const k = replay.frameIndex(replay.t);
  for (const [p, c] of POLICIES) {
    const e = replay.exposure(p, i, k);
    ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath();
    let on = false;
    e.forEach((v, q) => { if (v === null) { on = false; return; } on ? ctx.lineTo(X(q), Y(v)) : ctx.moveTo(X(q), Y(v)); on = true; });
    ctx.stroke();
  }
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(replay.t), T0); ctx.lineTo(X(replay.t), H - B); ctx.stroke();
}

// Timeline panel wired to a Replay.
export function replayPanel(replay) {
  const rep = replay.rep;
  const el = document.createElement("div");
  el.id = "replay";
  const updates = rep.updates.map((u) => `<i style="left:${(100 * u.t) / rep.t_end}%"></i>`).join("");
  el.innerHTML = `
    <div class="rhead"><b>Replay</b> <span>${rep.rung} &middot; leak seed ${rep.seed} &middot; ${rep.physics === "3d" ? "3D gas + 3D models" : "2D gas"}: ${rep.models}</span></div>
    <div class="hint">Click a person to see their three paths live.</div>
    <div class="rrow">
      <button id="r-play">&#9654; Play</button>
      <select id="r-speed"><option value="1">1x</option><option value="3" selected>3x</option><option value="6">6x</option><option value="12">12x</option></select>
      <select id="r-planner" title="whose crowd is shown">${Object.keys(rep.planners).map((p) => `<option value="${p}" ${p === "ccrp" ? "selected" : ""}>${PLAN_NAMES[p] || p}</option>`).join("")}</select>
      <label><input type="checkbox" id="r-gas" checked> true gas</label>
      <label><input type="checkbox" id="r-danger"> predicted danger</label>
      <label><input type="checkbox" id="r-source" checked> leak guesses</label>
      <label title="slow orbit around the plant, for videos"><input type="checkbox" id="r-cine"> cinematic camera</label>
      <button id="r-record" title="Plays from the start and saves the 3D view as a video file">&#9679; Record video</button>
    </div>
    <div class="rtrack"><input type="range" id="r-time" min="0" max="${rep.t_end}" step="0.1" value="0">
      <div class="ticks">${updates}${rep.t_alarm !== null ? `<b style="left:${(100 * rep.t_alarm) / rep.t_end}%"></b>` : ""}</div></div>
    <div id="r-text"></div>
    <div class="rcharts"><canvas id="r-out" width="420" height="110"></canvas><canvas id="r-ingas" width="420" height="110"></canvas>
      <div id="r-legend"></div></div>
    <div id="r-person" hidden><div id="r-pinfo"></div>
      <div class="rrow"><button id="r-follow">Follow</button><button id="r-pview" title="See through this person's eyes, with the green route to follow">&#128065; Person view</button><button id="r-clear">Clear</button></div>
      <canvas id="r-pchart" width="420" height="120"></canvas></div>`;
  document.body.appendChild(el);
  const $ = (id) => document.getElementById(id);
  $("r-play").onclick = () => {
    if (replay.t >= rep.t_end) replay.t = 0;
    replay.playing = !replay.playing;
  };
  $("r-speed").onchange = (e) => { replay.speed = +e.target.value; };
  $("r-planner").onchange = (e) => { replay.planner = e.target.value; };
  $("r-time").oninput = (e) => { replay.t = +e.target.value; replay.playing = false; };
  const gasOn = (on) => replay.gas.forEach((l) => { l.mesh.visible = on; });
  $("r-gas").onchange = (e) => gasOn(e.target.checked);
  $("r-danger").onchange = (e) => { replay.danger.mesh.visible = e.target.checked; };
  replay.showSource = true;
  $("r-source").onchange = (e) => { replay.showSource = e.target.checked; replay.lastUpdate = -2; };
  $("r-record").onclick = () => recordVideo(replay, $("r-record"));
  $("r-cine").onchange = (e) => { replay.cinematic = e.target.checked; };
  $("r-follow").onclick = () => { replay.personView = false; replay.follow = !replay.follow; };
  $("r-pview").onclick = () => { replay.personView = !replay.personView; replay.follow = replay.personView; };
  $("r-clear").onclick = () => replay.select(null);
  const curves = planCurves(replay);
  $("r-legend").innerHTML = Object.keys(curves).map((p) => `<span style="color:${PLAN_COLOURS[p] || "#ccc"}">&#9644; ${PLAN_NAMES[p] || p}</span>`).join(" ");
  return () => {
    const card = $("r-person");
    if (replay.selected !== null && replay.selected !== undefined) {
      const i = replay.selected, g = replay.gasSeconds(i);
      card.hidden = false;
      $("r-pinfo").innerHTML = `<b>Person ${i}</b> &middot; seconds in harmful gas<br>` + POLICIES.filter(([p]) => replay.tracks[p])
        .map(([p, c, what]) => `<span style="color:${c}">&#9644;</span> ${what}: <b>${g[p]} s</b>`).join("<br>");
      $("r-follow").textContent = replay.follow && !replay.personView ? "Stop following" : "Follow";
      $("r-pview").innerHTML = replay.personView ? "Exit person view" : "&#128065; Person view";
      drawExposure($("r-pchart"), replay, i, rep.t_end);
    } else card.hidden = true;
    drawCurves($("r-out"), curves, "out", "% of people out", replay.t, rep.t_end, 100);
    drawCurves($("r-ingas"), curves, "gas", "people in harmful gas now", replay.t, rep.t_end, null);
    $("r-play").innerHTML = replay.playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
    $("r-time").value = replay.t;
    const s = replay.stats(), m = rep.planners[replay.planner].metrics;
    const alarm = rep.t_alarm !== null && replay.t >= rep.t_alarm;
    let u = null;
    for (const up of rep.updates) if (up.t <= replay.t + 1e-6) u = up;
    const model = u
      ? `update at ${u.t.toFixed(0)} s: leak guess ${u.source_error_m.toFixed(1)} m off, size ${u.rate_median.toFixed(2)} (true ${u.rate_true.toFixed(2)}); locate ${u.locate_s.toFixed(2)} s, hazard ${u.predict_s.toFixed(2)} s`
      : "no model output yet";
    $("r-text").innerHTML =
      `<b>t = ${replay.t.toFixed(0)} s</b> &middot; ` +
      (alarm ? `<span class="alarm">ALARM at ${rep.t_alarm} s</span>` : `<span class="calm">leak not detected yet</span>`) +
      ` &middot; ${model}<br>${replay.planner}: <b>${((100 * s.out) / replay.n).toFixed(0)} % out</b>, ` +
      `${s.queue} queueing, <span class="ingas">${s.gas} in gas now</span> &middot; final: ${m.harm_s_mean.toFixed(2)} s in gas per person, ` +
      `${m.exposed_pct.toFixed(1)} % exposed, 95 % out at ${Number.isFinite(m.out_95_s) ? m.out_95_s.toFixed(0) + " s" : "-"}`;
  };
}

// Record the 3D canvas while the replay plays from the start. Served by `uv run world`, the
// video is saved into the run's folder (videos/); otherwise it downloads as a WebM.
// (Labels are HTML on top of the canvas and are not in the video.) Returns a promise.
export function recordVideo(replay, button, name) {
  const canvas = document.querySelector("#app canvas");
  if (!canvas.captureStream || typeof MediaRecorder === "undefined") {
    alert("This browser cannot record the canvas; use screen recording (Cmd+Shift+5).");
    return Promise.resolve(null);
  }
  const type = ["video/webm;codecs=vp9", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
  const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 8e6 });
  const chunks = [];
  const file = name || `replay-${replay.rep.seed}-${replay.planner}.webm`;
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const done = new Promise((resolve) => {
    rec.onstop = async () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      let saved = null;
      try {
        const r = await fetch(`save/${encodeURIComponent(file)}${location.search}`, { method: "POST", body: blob });
        if (r.ok) saved = (await r.json()).path;
      } catch { /* not served by uv run world */ }
      if (!saved) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = file;
        a.click();
      }
      if (button) { button.disabled = false; button.innerHTML = saved ? "&#10003; Saved" : "&#9679; Record video"; }
      resolve(saved || file);
    };
  });
  replay.t = 0;
  replay.playing = true;
  if (button) { button.disabled = true; button.textContent = "Recording..."; }
  rec.start(1000);
  const check = setInterval(() => {
    if (!replay.playing) { clearInterval(check); rec.stop(); }
  }, 250);
  return done;
}
