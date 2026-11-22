// Play a closed-loop run in the 3D world: true gas as layered translucent sheets, the
// planners' danger map, the locator's guesses, sensors changing colour, people moving
// (one planner at a time) and a timeline. Data: data/replay.json (uv run world --replay).
import * as THREE from "three";

const STATE_COLOURS = ["#ff7a00", "#ffd000", "#ff00d0"];  // walking, queueing, in gas

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
      this.tracks[p] = { cells: new Int16Array(raw.buffer), state: b64(d.state), metrics: d.metrics };
    }
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
    const z = this.rep.scene.pipe_height;
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

  frameIndex(t) { return Math.max(0, Math.min(this.rep.times.length - 1, Math.floor(t))); }

  personXY(track, k, i) {
    const o = (k * this.n + i) * 2;
    const iy = track.cells[o], ix = track.cells[o + 1];
    return [(ix + 0.5) * this.cell + this.jitter[i][0], (iy + 0.5) * this.cell + this.jitter[i][1]];
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
      if (st === 3) { this.bodies.setMatrixAt(i, hidden); this.heads.setMatrixAt(i, hidden); continue; }
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
    const pulse = 1 + 0.15 * Math.sin(performance.now() / 180);
    for (const r of this.rings) if (r.visible) r.material.opacity = 0.55 + 0.35 * Math.sin(performance.now() / 250);
    return pulse;
  }
}

const PLAN_COLOURS = { nearest: "#2a78d6", risk: "#eb6834", ccrp: "#1baf7a" };

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

// Timeline panel wired to a Replay.
export function replayPanel(replay) {
  const rep = replay.rep;
  const el = document.createElement("div");
  el.id = "replay";
  const updates = rep.updates.map((u) => `<i style="left:${(100 * u.t) / rep.t_end}%"></i>`).join("");
  el.innerHTML = `
    <div class="rhead"><b>Replay</b> <span>${rep.rung} &middot; leak seed ${rep.seed} &middot; models: ${rep.models}</span></div>
    <div class="rrow">
      <button id="r-play">&#9654; Play</button>
      <select id="r-speed"><option value="1">1x</option><option value="3" selected>3x</option><option value="6">6x</option><option value="12">12x</option></select>
      <select id="r-planner">${Object.keys(rep.planners).map((p) => `<option ${p === "ccrp" ? "selected" : ""}>${p}</option>`).join("")}</select>
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
      <div id="r-legend"></div></div>`;
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
  const curves = planCurves(replay);
  $("r-legend").innerHTML = Object.keys(curves).map((p) => `<span style="color:${PLAN_COLOURS[p] || "#ccc"}">&#9644; ${p}</span>`).join(" ");
  return () => {
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
