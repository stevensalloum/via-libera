// Via Libera story page: chapter rail, scroll progress, counters, hero video, and the live
// 2D map of the saved run (the same replay.json the 3D factory plays).
const $ = (id) => document.getElementById(id);
const RUN = location.search;  // ?run=<folder> is passed on to the factory and the data
const DATA = new URLSearchParams(RUN).get("run") ? `data/${new URLSearchParams(RUN).get("run")}/` : "data/";
for (const id of ["enter-top", "enter-hero", "enter-live", "enter-3d"]) if ($(id)) $(id).href = `factory.html${RUN}`;

// ---- rail + progress ----------------------------------------------------------------
const sections = [...document.querySelectorAll("section[data-title]")];
$("rail").innerHTML = sections.map((s) => `<a href="#${s.id}" title="${s.dataset.title}"></a>`).join("");
const dots = [...$("rail").children];
addEventListener("scroll", () => {
  const h = document.documentElement;
  $("progress").style.width = `${(100 * h.scrollTop) / (h.scrollHeight - h.clientHeight)}%`;
  let on = 0;
  sections.forEach((s, i) => { if (s.getBoundingClientRect().top < innerHeight * 0.4) on = i; });
  dots.forEach((d, i) => d.classList.toggle("on", i === on));
}, { passive: true });

// ---- counters: count up when visible ----------------------------------------------------
const count = (el) => {
  const to = +el.dataset.count, dec = +(el.dataset.dec || 0), unit = el.dataset.unit || "";
  const t0 = performance.now();
  const tick = (now) => {
    const f = Math.min(1, (now - t0) / 1200), e = 1 - (1 - f) ** 3;
    el.textContent = `${to > 0 && el.dataset.sign ? "+" : ""}${(to * e).toFixed(dec)}${unit}`;
    if (f < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
};
const io = new IntersectionObserver((es) => es.forEach((e) => {
  if (e.isIntersecting) { count(e.target); io.unobserve(e.target); }
}), { threshold: 0.6 });
document.querySelectorAll("[data-count]").forEach((el) => io.observe(el));

// ---- hero video: the newest recording of this run, if any ---------------------------------
(async () => {
  const v = $("hero-video");
  for (const name of ["1-overview-via-libera.webm", "overview.webm"]) {
    const r = await fetch(`${DATA}videos/${name}`, { method: "HEAD" }).catch(() => null);
    if (r && r.ok) { v.src = `${DATA}videos/${name}`; return; }
  }
  v.remove();
})();

// ---- live 2D map ----------------------------------------------------------------------
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const PATHS = [["ccrp", "#1baf7a"], ["once", "#eda100"], ["nearest", "#e34948"]];
const STATE = ["#ff9a3c", "#ffd000", "#ff00d0"];
const rep = await fetch(`${DATA}replay.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
if (!rep) {
  $("status").textContent = "No saved run found. Start with: uv run via-libera";
} else {
  const s = rep.scene, ny = s.solid.length, nx = s.solid[0].length, cell = s.cell, n = rep.people;
  const gas = rep.gas.frames.map(b64), danger = rep.updates.map((u) => b64(u.danger));
  const tracks = {};
  for (const [p, d] of Object.entries(rep.planners)) {
    tracks[p] = { cells: new Int16Array(b64(d.cells).buffer), state: b64(d.state) };
  }
  const T = Math.floor(tracks.ccrp.state.length / n) - 1;
  const cv = $("map"), g = cv.getContext("2d");
  const W = 1000, scale = W / (nx * cell);
  cv.width = W; cv.height = Math.round(ny * cell * scale);
  const X = (x) => x * scale, Y = (y) => cv.height - y * scale;
  const base = document.createElement("canvas");
  base.width = cv.width; base.height = cv.height;
  const bg = base.getContext("2d");
  bg.fillStyle = "#13283b"; bg.fillRect(0, 0, W, cv.height);
  s.solid.forEach((row, iy) => row.forEach((v, ix) => {
    if (v) { bg.fillStyle = "#4b5d6f"; bg.fillRect(X(ix * cell), Y((iy + 1) * cell), cell * scale + 0.5, cell * scale + 0.5); }
  }));
  bg.strokeStyle = "rgba(243,146,0,0.55)"; bg.lineWidth = 2;
  for (const r of s.pipes || []) { bg.beginPath(); bg.moveTo(X(r.from[0]), Y(r.from[1])); bg.lineTo(X(r.to[0]), Y(r.to[1])); bg.stroke(); }
  for (const e of s.exits) { bg.fillStyle = "#1baf7a"; bg.beginPath(); bg.arc(X(e.xy[0]), Y(e.xy[1]), 9, 0, 7); bg.fill(); }
  const layer = document.createElement("canvas");
  layer.width = nx; layer.height = ny;
  const lg = layer.getContext("2d"), img = lg.createImageData(nx, ny);
  let t = 0, playing = false, sel = null, lastKey = "";
  // the person the shortest path hurts most (vs Via Libera) is selected first
  const gasS = (p, i) => { let c = 0; const st = tracks[p].state; for (let k = 0; k <= T; k++) if (st[k * n + i] === 2) c++; return c; };
  let best = 0, gain = -1;
  for (let i = 0; i < n; i++) { const d = gasS("nearest", i) - gasS("ccrp", i); if (d > gain) { gain = d; best = i; } }
  sel = best;
  const xy = (p, k, i) => { const o = (k * n + i) * 2, tr = tracks[p].cells; return [(tr[o + 1] + 0.5) * cell, (tr[o] + 0.5) * cell]; };
  const draw = () => {
    const k = Math.min(T, Math.floor(t)), gi = Math.min(gas.length - 1, Math.floor(t / rep.gas.every));
    let u = -1;
    rep.updates.forEach((up, j) => { if (up.t <= t) u = j; });
    const key = `${gi}|${u}`;
    if (key !== lastKey) {
      lastKey = key;
      const fr = gas[gi], dz = u >= 0 ? danger[u] : null, d = img.data;
      for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
        const c = iy * nx + ix, o = ((ny - 1 - iy) * nx + ix) * 4, v = fr[c], p = dz ? dz[c] / 255 : 0;
        let rgba = [0, 0, 0, 0];
        if (v >= 128) rgba = [214, 40, 210, 220];
        else if (v >= 94) rgba = [255, 170, 0, 170];
        else if (v >= 40) rgba = [90, 170, 255, 40 + v];
        if (p >= 0.2 && rgba[3] < 150) rgba = [230, 30, 50, 60 + 140 * p];
        d.set(rgba, o);
      }
      lg.putImageData(img, 0, 0);
    }
    g.drawImage(base, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(layer, 0, 0, W, cv.height);
    const alarm = rep.t_alarm !== null && t >= rep.t_alarm;
    if (alarm && u >= 0) rep.updates[u].source_top.forEach((q, j) => {
      g.strokeStyle = "#3fc1e8"; g.lineWidth = j ? 1.5 : 3;
      g.beginPath(); g.arc(X(q.xy[0]), Y(q.xy[1]), 8 + 22 * Math.sqrt(q.p), 0, 7); g.stroke();
    });
    for (let i = 0; i < n; i++) {
      const st = tracks.ccrp.state[k * n + i];
      if (st === 3) continue;
      const [x, y] = xy("ccrp", k, i);
      g.fillStyle = STATE[st]; g.fillRect(X(x) - 2.5, Y(y) - 2.5, 5, 5);
    }
    if (sel !== null) for (const [p, col] of PATHS) {
      if (!tracks[p]) continue;
      g.strokeStyle = col; g.lineWidth = 4; g.lineJoin = "round"; g.beginPath();
      for (let q = 0; q <= k; q++) {
        const [x, y] = xy(p, q, sel);
        q ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y));
        if (tracks[p].state[q * n + sel] === 3) break;
      }
      g.stroke();
      const [x, y] = xy(p, k, sel);
      g.fillStyle = col; g.beginPath(); g.arc(X(x), Y(y), 7, 0, 7); g.fill();
      g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.stroke();
    }
    $("clock").textContent = `${t.toFixed(0)} s`;
    $("time").value = t;
    const out = [...Array(n).keys()].filter((i) => tracks.ccrp.state[k * n + i] === 3).length;
    $("status").innerHTML = (alarm ? `<b style="color:#e34948">ALARM</b> at ${rep.t_alarm} s` : "leak not detected yet") +
      ` · ${Math.round((100 * out) / n)} % out` + (u >= 0 ? ` · leak guess ${rep.updates[u].source_error_m.toFixed(1)} m off` : "");
    drawPerson(k);
  };
  const drawPerson = (k) => {
    if (sel === null) return;
    const secs = (p) => { let c = 0; for (let q = 0; q <= k; q++) if (tracks[p].state[q * n + sel] === 2) c++; return c; };
    $("person").innerHTML = `<b>Person ${sel}</b> · seconds in harmful gas so far<br>` +
      PATHS.map(([p, col]) => `<span style="color:${col}">●</span> ${p === "ccrp" ? "Via Libera (live)" : p === "once" ? "planned once" : "shortest path"}: <b>${secs(p)} s</b>`).join("<br>");
    const pc = $("pchart"), c = pc.getContext("2d"), H = pc.height, L = 30, B = 18;
    c.clearRect(0, 0, pc.width, H);
    c.fillStyle = "#8ba3b8"; c.font = "11px system-ui"; c.fillText("gas this person breathes (log scale)", L, 12);
    const PX = (q) => L + (q / T) * (pc.width - L - 6), PY = (v) => 18 + (1 - v / 255) * (H - 18 - B);
    c.strokeStyle = "rgba(214,40,210,0.7)"; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(L, PY(127.5)); c.lineTo(pc.width, PY(127.5)); c.stroke(); c.setLineDash([]);
    c.fillText("toxic", 0, PY(127.5) + 4);
    for (const [p, col] of PATHS) {
      if (!tracks[p]) continue;
      c.strokeStyle = col; c.lineWidth = 2; c.beginPath();
      let on = false;
      for (let q = 0; q <= k; q++) {
        if (tracks[p].state[q * n + sel] === 3) { on = false; continue; }
        const o = (q * n + sel) * 2, gi = Math.min(gas.length - 1, Math.floor(q / rep.gas.every));
        const v = gas[gi][tracks[p].cells[o] * nx + tracks[p].cells[o + 1]];
        on ? c.lineTo(PX(q), PY(v)) : c.moveTo(PX(q), PY(v)); on = true;
      }
      c.stroke();
    }
  };
  $("time").max = T;
  $("time").oninput = (e) => { t = +e.target.value; playing = false; $("play").textContent = "▶ Play"; draw(); };
  $("play").onclick = () => { if (t >= T) t = 0; playing = !playing; $("play").textContent = playing ? "❚❚ Pause" : "▶ Play"; };
  cv.onclick = (e) => {  // nearest person to the click
    const r = cv.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * W, my = ((e.clientY - r.top) / r.height) * cv.height;
    const k = Math.min(T, Math.floor(t));
    let bestI = null, bestD = 30;
    for (let i = 0; i < n; i++) {
      if (tracks.ccrp.state[k * n + i] === 3) continue;
      const [x, y] = xy("ccrp", k, i), d = Math.hypot(X(x) - mx, Y(y) - my);
      if (d < bestD) { bestD = d; bestI = i; }
    }
    if (bestI !== null) { sel = bestI; draw(); }
  };
  new IntersectionObserver((es) => { if (es[0].isIntersecting && t === 0 && !playing) { playing = true; $("play").textContent = "❚❚ Pause"; } }, { threshold: 0.5 }).observe(cv);
  let last = performance.now();
  const loop = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) { t = Math.min(T, t + dt * 4); if (t >= T) { playing = false; $("play").textContent = "▶ Play"; } }
    draw();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
