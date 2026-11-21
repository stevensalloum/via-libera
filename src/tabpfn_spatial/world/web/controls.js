// Orbit around the plant, or walk through it at eye height. Walking collides with the
// planner's solid grid and the outer walls (doors open), the same obstacles the
// evacuation planner sees.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { DOOR_W } from "./world.js";

const EYE = 1.7, RADIUS = 0.3, WALK = 3.0, RUN = 7.0;

export function makeSolidTest(scene) {
  const { width: W, height: H, cell, solid } = scene;
  const doors = scene.exits.map((e) => e);
  const inDoor = (x, y) => doors.some((e) => {
    const [ex, ey] = e.xy;
    if (e.side === "west" || e.side === "east") return Math.abs(y - ey) < DOOR_W / 2 - RADIUS;
    return Math.abs(x - ex) < DOOR_W / 2 - RADIUS;
  });
  return (x, y) => {
    const inside = x >= 0 && x <= W && y >= 0 && y <= H;
    const nearWall = x > -0.5 && x < W + 0.5 && y > -0.5 && y < H + 0.5 &&
      (x < 0.05 || x > W - 0.05 || y < 0.05 || y > H - 0.05);
    if (nearWall && !inDoor(x, y)) return true;
    if (!inside) return false;
    const ix = Math.min(Math.floor(x / cell), solid[0].length - 1);
    const iy = Math.min(Math.floor(y / cell), solid.length - 1);
    return solid[iy][ix] === 1;
  };
}

export class Navigator {
  constructor(camera, dom, scene, F) {
    this.camera = camera;
    this.F = F;
    this.solidAt = makeSolidTest(scene);
    this.orbit = new OrbitControls(camera, dom);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.maxPolarAngle = Math.PI * 0.495;
    this.orbit.minDistance = 3;
    this.orbit.maxDistance = Math.max(F.W, F.H) * 3;
    this.walk = new PointerLockControls(camera, document.body);
    this.mode = "orbit";
    this.keys = new Set();
    this.fly = null;
    addEventListener("keydown", (e) => this.keys.add(e.code));
    addEventListener("keyup", (e) => this.keys.delete(e.code));
  }

  blocked(x, y) {
    for (const [dx, dy] of [[0, 0], [RADIUS, 0], [-RADIUS, 0], [0, RADIUS], [0, -RADIUS]]) {
      if (this.solidAt(x + dx, y + dy)) return true;
    }
    return false;
  }

  // Start walking at factory point (x, y), looking towards (lx, ly).
  startWalk(x, y, lx, ly) {
    this.mode = "walk";
    this.orbit.enabled = false;
    this.camera.position.copy(this.F.P(x, y, EYE));
    this.camera.lookAt(this.F.P(lx, ly, EYE));
    this.walk.lock();
  }

  startOrbit() {
    this.mode = "orbit";
    this.walk.unlock();
    this.orbit.enabled = true;
    const [x, y] = this.F.toFactory(this.camera.position);
    this.orbit.target.copy(this.F.P(x, y, 0));
    this.camera.position.add(new THREE.Vector3(0, 18, 18));
  }

  flyTo(position, target, seconds = 1.2) {
    if (this.mode === "walk") this.startOrbit();
    this.fly = { p0: this.camera.position.clone(), t0: this.orbit.target.clone(), p1: position, t1: target, s: 0, T: seconds };
  }

  update(dt) {
    if (this.fly) {
      const f = this.fly;
      f.s = Math.min(1, f.s + dt / f.T);
      const e = f.s < 0.5 ? 2 * f.s * f.s : 1 - (-2 * f.s + 2) ** 2 / 2;
      this.camera.position.lerpVectors(f.p0, f.p1, e);
      this.orbit.target.lerpVectors(f.t0, f.t1, e);
      if (f.s >= 1) this.fly = null;
    }
    if (this.mode === "orbit") { this.orbit.update(); return; }
    if (!this.walk.isLocked) return;
    const k = this.keys, speed = (k.has("ShiftLeft") || k.has("ShiftRight") ? RUN : WALK) * dt;
    const fwd = (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0) - (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0);
    const side = (k.has("KeyD") || k.has("ArrowRight") ? 1 : 0) - (k.has("KeyA") || k.has("ArrowLeft") ? 1 : 0);
    if (!fwd && !side) return;
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    dir.y = 0;
    dir.normalize();
    const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0));
    const move = dir.multiplyScalar(fwd).add(right.multiplyScalar(side)).normalize().multiplyScalar(speed);
    const p = this.camera.position;
    const [x, y] = this.F.toFactory(p);
    const [nx] = this.F.toFactory(p.clone().add(new THREE.Vector3(move.x, 0, 0)));
    if (!this.blocked(nx, y)) p.x += move.x;  // slide along walls: x and z separately
    const [x2, ny] = this.F.toFactory(p.clone().add(new THREE.Vector3(0, 0, move.z)));
    if (!this.blocked(x2, ny)) p.z += move.z;
    p.y = EYE;
  }
}
