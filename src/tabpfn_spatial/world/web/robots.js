// AGVs on their lanes and the height zone around a leak.
// AGVs: without a replay they drive their loops (same speed and spacing as sim.robots);
// in a replay they follow the exported positions and stop at the alarm, as real AGVs do on
// an emergency stop. A stopped AGV that blocks the planner gets a red ring.
import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const SPEED = 1.0, PER_LANE = 2;  // sim.robots.AGV_SPEED, AGV_PER_LANE

function agvMesh() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.35, 0.9),
    new THREE.MeshStandardMaterial({ color: "#e8e9eb", metalness: 0.3, roughness: 0.45 }));
  body.position.y = 0.3;
  g.add(body);
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(1.42, 0.08, 0.92),
    new THREE.MeshStandardMaterial({ color: "#f2b705", roughness: 0.5 }));
  stripe.position.y = 0.18;
  g.add(stripe);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.5, 0.7),
    new THREE.MeshStandardMaterial({ color: "#b58b55", roughness: 0.9 }));  // a load
  deck.position.y = 0.73;
  g.add(deck);
  const lidar = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.12, 16),
    new THREE.MeshStandardMaterial({ color: "#1b1d20", roughness: 0.6 }));
  lidar.position.set(0.6, 0.54, 0);
  g.add(lidar);
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8),
    new THREE.MeshStandardMaterial({ color: "#ffb000", emissive: "#ff9a00", emissiveIntensity: 0.4 }));
  beacon.position.set(-0.55, 0.58, 0.3);
  g.add(beacon);
  const ring = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.2, 32).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: "#ff2d55", transparent: true, opacity: 0.85 }));
  ring.position.y = 0.04;
  ring.visible = false;
  g.add(ring);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  g.userData = { beacon, ring };
  return g;
}

function laneGeometry(lane) {
  const pts = [...lane, lane[0]];
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return { pts, cum, total: cum.at(-1) };
}

function alongLane(g, s) {
  s = ((s % g.total) + g.total) % g.total;
  let i = 0;
  while (i < g.cum.length - 2 && g.cum[i + 1] <= s) i++;
  const seg = g.cum[i + 1] - g.cum[i] || 1;
  const f = (s - g.cum[i]) / seg, a = g.pts[i], b = g.pts[i + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), Math.atan2(b[1] - a[1], b[0] - a[0])];
}

export class Agvs {
  constructor(scene, F) {
    this.F = F;
    this.group = new THREE.Group();
    this.lanes = scene.agv_lanes.map(laneGeometry);
    this.vehicles = [];
    const laneMat = new THREE.MeshBasicMaterial({ color: "#f2b705", transparent: true, opacity: 0.55 });
    for (const lane of scene.agv_lanes) {  // painted floor lines along each loop
      const pts = [...lane, lane[0]];
      for (let i = 1; i < pts.length; i++) {
        const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
        const len = Math.hypot(x1 - x0, y1 - y0);
        if (len < 1e-6) continue;
        const strip = new THREE.Mesh(new THREE.PlaneGeometry(len, 0.12).rotateX(-Math.PI / 2), laneMat);
        strip.position.copy(F.P((x0 + x1) / 2, (y0 + y1) / 2, 0.015));
        strip.rotation.y = Math.atan2(y1 - y0, x1 - x0);
        this.group.add(strip);
      }
    }
    this.lanes.forEach((g, l) => {
      for (let k = 0; k < PER_LANE; k++) {
        const m = agvMesh();
        m.userData.lane = l;
        m.userData.offset = (k * g.total) / PER_LANE;
        this.group.add(m);
        this.vehicles.push(m);
      }
    });
    this.replay = null;
  }

  // free running (no replay): drive the loops
  drive(t) {
    for (const m of this.vehicles) {
      const [x, y, h] = alongLane(this.lanes[m.userData.lane], m.userData.offset + SPEED * t);
      this.place(m, x, y, h);
      m.userData.beacon.material.emissiveIntensity = 0.4;
    }
  }

  // replay: positions per second from the run, stopped from t_stop on
  show(agvs, t, wall) {
    const k = Math.min(Math.floor(t), agvs.xy[0].length - 2), f = t - k;
    const stopped = agvs.t_stop !== null && t >= agvs.t_stop;
    this.vehicles.forEach((m, i) => {
      const a = agvs.xy[i][Math.max(0, k)], b = agvs.xy[i][Math.max(0, k) + 1];
      const x = a[0] + f * (b[0] - a[0]), y = a[1] + f * (b[1] - a[1]);
      const moving = Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6;
      this.place(m, x, y, moving ? Math.atan2(b[1] - a[1], b[0] - a[0]) : m.userData.heading ?? 0);
      m.userData.beacon.material.emissiveIntensity = stopped ? 1.5 + 1.5 * Math.sin(wall * 9) : 0.4;
      m.userData.ring.visible = stopped && Boolean(agvs.stops[i]?.blocks);
    });
  }

  place(m, x, y, heading) {
    m.position.copy(this.F.P(x, y, 0));
    m.rotation.y = heading;
    m.userData.heading = heading;
  }
}

// Translucent dome over the height zone + a label with the instruction.
export class HeightZone {
  constructor(F) {
    this.F = F;
    this.group = new THREE.Group();
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 16, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: "#3d8bfd", transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide }));
    this.group.add(this.dome);
    this.truth = new THREE.Mesh(new THREE.RingGeometry(0.97, 1.0, 64).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.7 }));
    this.truth.position.y = 0.05;
    this.group.add(this.truth);
    const el = document.createElement("div");
    el.className = "label zone";
    this.el = el;
    this.label = new CSS2DObject(el);
    this.group.add(this.label);
    this.group.visible = false;
  }

  show(zone, t) {
    if (!zone || t === null) { this.group.visible = false; return; }
    const ups = zone.updates.filter((u) => u.t <= t);
    const z = ups.at(-1);
    if (!z || z.radius_m <= 0 || z.advice === "none") { this.group.visible = false; return; }
    this.group.visible = true;
    const r = Math.max(z.radius_m, 1.0), h = Math.min(r, 3.0);
    this.dome.scale.set(r, h, r);
    this.dome.position.copy(this.F.P(...z.xy, 0));
    this.dome.material.color.set(z.advice === "stay low" ? "#3d8bfd" : "#ff8a1f");
    const tz = zone.truth;
    this.truth.scale.setScalar(Math.max(tz.radius_m, 1.0));
    this.truth.position.copy(this.F.P(...tz.xy, 0.05));
    this.label.position.copy(this.F.P(...z.xy, h + 0.8));
    this.el.textContent = `${z.advice.toUpperCase()} within ${z.radius_m.toFixed(1)} m (${z.gas}, ~${z.litres_per_s.toFixed(0)} L/s)`;
  }
}
