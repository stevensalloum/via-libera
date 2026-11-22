// Via Libera: full-screen scenes in the order of the video. Keys: ← → (or space) scene,
// C face-cam frame, H hide the navigation. Charts are drawn from story-data.json.
const $ = (id) => document.getElementById(id);
const RUN = location.search;
const RUNQ = new URLSearchParams(RUN).get("run");
const DATA = RUNQ ? `data/${RUNQ}/` : "data/";
$("enter-top").href = `factory.html${RUN}`;
const C = { tabpfn: "#f39200", knn: "#8ba3b8", base: "#3fc1e8", prior: "#5b6f82", perfect: "#e6eef5",
  nearest: "#e34948", once: "#eda100", live: "#1baf7a" };

// ---- scenes: rail, keys, counters, lazy factory ------------------------------------------
const scenes = [...document.querySelectorAll("section.scene")];
$("rail").innerHTML = scenes.map((s) => `<a href="#${s.id}" title="${s.dataset.title}"></a>`).join("");
const dots = [...$("rail").children];
let cur = 0;
const seen = new IntersectionObserver((es) => es.forEach((e) => {
  if (!e.isIntersecting) return;
  cur = scenes.indexOf(e.target);
  dots.forEach((d, i) => d.classList.toggle("on", i === cur));
  $("scene-no").textContent = `${cur + 1} / ${scenes.length} · ${e.target.dataset.title}`;
  e.target.querySelectorAll("[data-count]").forEach(count);
  const f = e.target.querySelector("iframe[data-src]");
  if (f && !f.src) f.src = f.dataset.src;
}), { threshold: 0.55 });
scenes.forEach((s) => seen.observe(s));
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const go = (k) => { cur = Math.max(0, Math.min(scenes.length - 1, k)); scenes[cur].scrollIntoView(); e.preventDefault(); };
  if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(e.key)) go(cur + 1);
  if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) go(cur - 1);
  if (e.key === "c" || e.key === "C") $("camframe").hidden = !$("camframe").hidden;
  if (e.key === "h" || e.key === "H") document.body.classList.toggle("clean");
});
function count(el) {
  if (el.dataset.done) return;
  el.dataset.done = 1;
  const to = +el.dataset.count, unit = el.dataset.unit || "", t0 = performance.now();
  const tick = (now) => {
    const f = Math.min(1, (now - t0) / 1100), e = 1 - (1 - f) ** 3;
    el.textContent = `${Math.round(to * e)}${unit}`;
    if (f < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ---- tooltip ----------------------------------------------------------------------------
const tip = document.createElement("div");
tip.className = "tip";
document.body.appendChild(tip);
const hover = (node, html) => {
  node.addEventListener("mousemove", (e) => { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = `${e.clientX + 12}px`; tip.style.top = `${e.clientY + 12}px`; });
  node.addEventListener("mouseleave", () => { tip.style.display = "none"; });
};

// ---- SVG charts (one y axis each, units on the axes, values on hover) ---------------------
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const fmt = (v, d = 2) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(d));
function frame(box, { title, note, legend, W = 600, H = 330, yLabel, yMin = 0, yMax, xLabel }) {
  box.innerHTML = `<div class="title">${title}</div>` + (legend ? `<div class="legend-row">${legend.map(([n, c, dash]) =>
    `<span><i style="background:${dash ? `repeating-linear-gradient(90deg,${c} 0 5px,transparent 5px 8px)` : c}"></i>${n}</span>`).join("")}</div>` : "");
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  if (note) box.insertAdjacentHTML("beforeend", `<div class="note">${note}</div>`);
  const L = 62, R = 14, T = 12, B = xLabel ? 50 : 34;
  const Y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B);
  for (let k = 0; k <= 4; k++) {
    const v = yMin + ((yMax - yMin) * k) / 4, y = Y(v);
    el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: "#22405c", "stroke-width": 1 }, svg);
    el("text", { x: L - 8, y: y + 4, "text-anchor": "end", fill: "#8ba3b8", "font-size": 12 }, svg).textContent = fmt(v, yMax - yMin < 0.5 ? 2 : 1);
  }
  const yl = el("text", { x: 14, y: (T + H - B) / 2, fill: "#8ba3b8", "font-size": 12.5, "text-anchor": "middle",
    transform: `rotate(-90 14 ${(T + H - B) / 2})` }, svg);
  yl.textContent = yLabel;
  if (xLabel) el("text", { x: (L + W - R) / 2, y: H - 6, fill: "#8ba3b8", "font-size": 12.5, "text-anchor": "middle" }, svg).textContent = xLabel;
  return { svg, L, R, T, B, W, H, Y };
}
function lineChart(box, { xs, xTicks, series, ...o }) {
  const f = frame(box, { ...o, legend: series.map((s) => [s.name, s.color, s.dash]) });
  const X = (i) => f.L + 16 + (i / (xs.length - 1)) * (f.W - f.L - f.R - 32);
  xs.forEach((x, i) => { el("text", { x: X(i), y: f.H - f.B + 18, fill: "#8ba3b8", "font-size": 12, "text-anchor": "middle" }, f.svg).textContent = (xTicks || xs)[i]; });
  for (const s of series) {
    el("path", { d: s.ys.map((v, i) => `${i ? "L" : "M"}${X(i)},${f.Y(v)}`).join(""), stroke: s.color, "stroke-width": s.bold ? 3.5 : 2.2,
      fill: "none", "stroke-dasharray": s.dash ? "6 5" : "none" }, f.svg);
    s.ys.forEach((v, i) => {
      const c = el("circle", { cx: X(i), cy: f.Y(v), r: 5, fill: s.color, stroke: "#102234", "stroke-width": 2 }, f.svg);
      hover(c, `<b>${s.name}</b><br>${o.xLabel}: ${(xTicks || xs)[i]}<br>${o.yLabel}: ${fmt(v, 3)}`);
    });
  }
}
function barChart(box, { groups, series, ref, ...o }) {
  const f = frame(box, { ...o, legend: series.map((s) => [s.name, s.color]).concat(ref ? [[ref.name, ref.color, true]] : []) });
  const gw = (f.W - f.L - f.R) / groups.length, bw = Math.min(46, (gw * 0.78) / series.length);
  groups.forEach((g, gi) => {
    const x0 = f.L + gi * gw + (gw - bw * series.length) / 2;
    el("text", { x: f.L + gi * gw + gw / 2, y: f.H - f.B + 18, fill: "#cfdbe6", "font-size": 12.5, "text-anchor": "middle" }, f.svg).textContent = g;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (v === null || v === undefined) return;
      const x = x0 + si * bw, y = f.Y(v), y0 = f.Y(Math.max(o.yMin || 0, 0));
      const r = el("rect", { x: x + 2, y: Math.min(y, y0), width: bw - 4, height: Math.abs(y0 - y), rx: 3, fill: s.color }, f.svg);
      el("text", { x: x + bw / 2, y: y - 6 - (s.err ? f.Y(0) - f.Y(s.err[gi]) : 0), fill: "#e6eef5", "font-size": 11.5, "text-anchor": "middle" }, f.svg).textContent = fmt(v, o.dec ?? 2);
      if (s.err) {
        const e = s.err[gi];
        el("line", { x1: x + bw / 2, x2: x + bw / 2, y1: f.Y(v - e), y2: f.Y(v + e), stroke: "#e6eef5", "stroke-width": 1.4 }, f.svg);
      }
      hover(r, `<b>${s.name}</b> · ${g}<br>${o.yLabel}: ${fmt(v, 3)}${s.err ? ` ± ${fmt(s.err[gi], 3)} (s.e.)` : ""}`);
    });
  });
  if (ref) {
    const y = f.Y(ref.value);
    el("line", { x1: f.L, x2: f.W - f.R, y1: y, y2: y, stroke: ref.color, "stroke-width": 2, "stroke-dasharray": "7 5" }, f.svg);
    el("text", { x: f.W - f.R - 4, y: y - 6, fill: ref.color, "font-size": 12, "text-anchor": "end" }, f.svg).textContent = `${ref.name}: ${fmt(ref.value)}`;
  }
}
function scatterChart(box, { points, xMax, ...o }) {
  const f = frame(box, { ...o, legend: points.map((p) => [p.name, p.color]) });
  const X = (v) => f.L + (v / xMax) * (f.W - f.L - f.R);
  for (let k = 0; k <= 4; k++) el("text", { x: X((xMax * k) / 4), y: f.H - f.B + 18, fill: "#8ba3b8", "font-size": 12, "text-anchor": "middle" }, f.svg).textContent = (xMax * k) / 4;
  for (const p of points) {
    const c = el("circle", { cx: X(p.x), cy: f.Y(p.y), r: 9, fill: p.color, stroke: "#102234", "stroke-width": 2 }, f.svg);
    const right = p.x > 0.7 * xMax;  // labels near the edge go to the left of the dot
    el("text", { x: X(p.x) + (right ? -14 : 14), y: f.Y(p.y) + 4, fill: "#cfdbe6", "font-size": 12,
      "text-anchor": right ? "end" : "start" }, f.svg).textContent = p.label;
    hover(c, `<b>${p.name}</b><br>recall ${p.y} % · false-positive rate ${p.x} %`);
  }
}

// ---- the charts ---------------------------------------------------------------------------
const D = await fetch("story-data.json").then((r) => r.json());
const s2 = D.source_2d;
lineChart($("c-source-err"), { title: "Source position error after the alarm", xs: s2.t, xLabel: "time after alarm (s)",
  yLabel: "median error (m)", yMax: 18, note: `n = ${s2.n}; held-out leaks · calibration (ECE): TabPFN ${s2.ece.tabpfn}, centroid ${s2.ece.centroid}, k-NN ${s2.ece.knn}`,
  series: [{ name: "prior", color: C.prior, ys: s2.error_m.prior, dash: true }, { name: "centroid", color: C.base, ys: s2.error_m.centroid },
    { name: "k-NN", color: C.knn, ys: s2.error_m.knn }, { name: "TabPFN", color: C.tabpfn, ys: s2.error_m.tabpfn, bold: true }] });
lineChart($("c-source-size"), { title: "Release-rate error", xs: s2.size_t, xLabel: "time after alarm (s)", yLabel: "median error (%)", yMax: 40,
  series: [{ name: "prior", color: C.prior, ys: s2.size_error_pct.prior, dash: true }, { name: "k-NN", color: C.knn, ys: s2.size_error_pct.knn },
    { name: "TabPFN", color: C.tabpfn, ys: s2.size_error_pct.tabpfn, bold: true }] });
const h2 = D.hazard_2d;
lineChart($("c-hazard-brier"), { title: "Forecast error across mixing strengths", xs: h2.diffusivity, xTicks: h2.diffusivity.map((v) => `${v}`),
  xLabel: "turbulent diffusivity (m²/s)", yLabel: "Brier score (lower is better)", yMin: 0.03, yMax: 0.07, note: `n = ${h2.n}`,
  series: [{ name: "forward physics", color: C.base, ys: h2.brier.forward }, { name: "TabPFN", color: C.tabpfn, ys: h2.brier.tabpfn, bold: true }] });
scatterChart($("c-hazard-roc"), { title: "Operating points (harmful cells, test set)", xLabel: "false-positive rate (%)", yLabel: "recall (%)",
  yMin: 50, yMax: 100, xMax: 10, points: [
    { name: "forward physics", color: C.base, x: h2.operating.forward[1], y: h2.operating.forward[0], label: "physics" },
    { name: "TabPFN, P ≥ 0.5", color: C.tabpfn, x: h2.operating.tabpfn_p50[1], y: h2.operating.tabpfn_p50[0], label: "TabPFN P≥0.5" },
    { name: "TabPFN, P ≥ 0.19", color: "#ffc15e", x: h2.operating.tabpfn_p19[1], y: h2.operating.tabpfn_p19[0], label: "TabPFN P≥0.19" }] });
const p2 = D.planner_2d, pv = (m, p) => p2[`${m}/${p}`];
barChart($("c-planner"), { title: "Time in harmful gas per person", groups: ["risk-aware planner", "CCRP planner"], yLabel: "seconds per person", W: 1000,
  yMax: 0.8, dec: 3, ref: { name: "nearest exit (no model)", color: C.nearest, value: pv("baseline", "nearest")[0] },
  note: `${p2.plants} unseen 2D plants, one leak each; mean ± s.e. · source error of the baseline models ${fmt(p2["loc/baseline"], 1)} m, TabPFN ${fmt(p2["loc/tabpfn"], 1)} m (median)`,
  series: [["baseline models", C.base, "baseline"], ["TabPFN models", C.tabpfn, "tabpfn"], ["perfect knowledge", C.perfect, "perfect"]].map(([name, color, m]) => ({
    name, color, values: ["risk", "ccrp"].map((p) => pv(m, p)[0]), err: ["risk", "ccrp"].map((p) => pv(m, p)[1]) })) });
const f3 = D.factory_3d;
barChart($("c-3d-skill"), { title: "Source localisation in 3D", groups: ["position error", "height error"], yLabel: "median error (m)", yMax: 2.5,
  note: `${f3.plants} plants · hazard at P ≥ 0.5: recall ${fmt(f3.tabpfn.hz_recall)} vs ${fmt(f3.knn.hz_recall)} (k-NN), false-alarm rate ${fmt(f3.tabpfn.hz_false, 3)} vs ${fmt(f3.knn.hz_false, 3)}`,
  series: [{ name: "k-NN", color: C.knn, values: [f3.knn.loc_err_median_m, f3.knn.height_err_m] },
    { name: "TabPFN", color: C.tabpfn, values: [f3.tabpfn.loc_err_median_m, f3.tabpfn.height_err_m] }] });
const ex = f3.exposure;
barChart($("c-3d-exposure"), { title: "Time in harmful gas per person (3D)", groups: ["nearest exit", "CCRP + k-NN", "CCRP + TabPFN", "CCRP + perfect"],
  yLabel: "seconds per person", yMax: 0.8, dec: 2, note: `${f3.plants} plants, mean ± s.e.`,
  series: [{ name: "mean", color: C.tabpfn, values: [ex.nearest[0], ex.knn[0], ex.tabpfn[0], ex.perfect[0]], err: [ex.nearest[1], ex.knn[1], ex.tabpfn[1], ex.perfect[1]] }] });
const re = D.real;
barChart($("c-real-source"), { title: "Source error on real measurements", groups: ["wind tunnel, 1:50 site (m)", "sensor grid (cells)"],
  yLabel: "median error (dataset units)", yMax: 1.6, note: `n = ${re.source_redvapor_m.n.join(" / ")} and ${re.source_srgdm_cells.n.join(" / ")} cases (TabPFN / k-NN)`,
  series: [{ name: "k-NN", color: C.knn, values: [re.source_redvapor_m.knn, re.source_srgdm_cells.knn] },
    { name: "TabPFN", color: C.tabpfn, values: [re.source_redvapor_m.tabpfn, re.source_srgdm_cells.tabpfn] }] });
const ds = [["hazard_redvapor", "1:50 site"], ["hazard_orebro", "office, HVAC"], ["hazard_uci", "MOX tunnel"]];
barChart($("c-real-auc"), { title: "Hot-zone detection on real measurements", groups: ds.map((d) => d[1]), yLabel: "AUC", yMin: 0.5, yMax: 1,
  series: [{ name: "k-NN", color: C.knn, values: ds.map((d) => re[d[0]].auc_hot.knn) }, { name: "TabPFN", color: C.tabpfn, values: ds.map((d) => re[d[0]].auc_hot.tabpfn) }] });

// ---- flow chart ---------------------------------------------------------------------------
(() => {
  const svg = $("flowchart");
  const blocks = [
    [20, "Detectors", "readings every 5 s", "Wall and mast detectors, noisy; the alarm fires on the first reading above 0.03."],
    [255, "Source", "where · how strong · how high", "TabPFN classifier + regressors on the sensor history: source map and release-rate quantiles (3D: height too).", 1],
    [490, "Hazard", "P(toxic), P(flammable)", "TabPFN regressor on distance to the source, time, wind, rate: probability per cell 0–120 s ahead.", 1],
    [725, "Routing", "CCRP on P(harm)", "Graph search with exit capacities; each step costs length + w · P(harm). Not a learned model."],
    [960, "People & robots", "a route each", "Every person still inside gets a route; robots stop or reroute."]];
  blocks.forEach(([x, name, sub, info, tab], i) => {
    const g = el("g", { class: `blk${tab ? " tab" : ""}` }, svg);
    el("rect", { x, y: 110, width: 210, height: 120 }, g);
    if (tab) el("text", { x: x + 105, y: 136, "text-anchor": "middle", class: "badge" }, g).textContent = "TabPFN";
    el("text", { x: x + 105, y: 170, "text-anchor": "middle", "font-weight": 700 }, g).textContent = name;
    el("text", { x: x + 105, y: 196, "text-anchor": "middle", class: "s" }, g).textContent = sub;
    g.addEventListener("mouseenter", () => { $("flow-cap").textContent = info; svg.querySelectorAll(".blk").forEach((b) => b.classList.remove("on")); g.classList.add("on"); });
    if (i < blocks.length - 1) el("path", { d: `M${x + 212},170 L${x + 233},170 m-8,-6 l8,6 l-8,6` }, svg);
  });
  el("path", { class: "loop", d: "M1065,232 C1065,330 125,330 125,232" }, svg);
  el("text", { x: 600, y: 322, "text-anchor": "middle", class: "s" }, svg).textContent = "replanned at every model update (15 s)";
  el("text", { x: 600, y: 70, "text-anchor": "middle", class: "s" }, svg).textContent = "context = simulated leaks of the same plant · no gradient training";
})();

// ---- demo: one worker, three plans (the saved cloud run) ----------------------------------
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const PATHS = [["ccrp", C.live, "Via Libera, replanned"], ["once", C.once, "planned once"], ["nearest", C.nearest, "nearest exit"]];
const rep = await fetch(`${DATA}replay.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
if (!rep) $("status").textContent = "No saved run. Start with: uv run via-libera";
else {
  const s = rep.scene, ny = s.solid.length, nx = s.solid[0].length, cell = s.cell, n = rep.people;
  const gas = rep.gas.frames.map(b64), danger = rep.updates.map((u) => b64(u.danger));
  const tr = {};
  for (const [p, d] of Object.entries(rep.planners)) tr[p] = { cells: new Int16Array(b64(d.cells).buffer), state: b64(d.state) };
  const T = Math.floor(tr.ccrp.state.length / n) - 1;
  const cv = $("map"), g = cv.getContext("2d"), W = 1100, sc = W / (nx * cell);
  cv.width = W; cv.height = Math.round(ny * cell * sc);
  const X = (x) => x * sc, Y = (y) => cv.height - y * sc;
  const base = document.createElement("canvas");
  base.width = W; base.height = cv.height;
  const bg = base.getContext("2d");
  bg.fillStyle = "#13283b"; bg.fillRect(0, 0, W, cv.height);
  s.solid.forEach((row, iy) => row.forEach((v, ix) => { if (v) { bg.fillStyle = "#4b5d6f"; bg.fillRect(X(ix * cell), Y((iy + 1) * cell), cell * sc + 0.5, cell * sc + 0.5); } }));
  bg.strokeStyle = "rgba(243,146,0,0.5)"; bg.lineWidth = 2;
  for (const r of s.pipes || []) { bg.beginPath(); bg.moveTo(X(r.from[0]), Y(r.from[1])); bg.lineTo(X(r.to[0]), Y(r.to[1])); bg.stroke(); }
  for (const e of s.exits) { bg.fillStyle = C.live; bg.beginPath(); bg.arc(X(e.xy[0]), Y(e.xy[1]), 9, 0, 7); bg.fill(); }
  const layer = document.createElement("canvas");
  layer.width = nx; layer.height = ny;
  const lg = layer.getContext("2d"), img = lg.createImageData(nx, ny);
  const gasS = (p, i) => { let c = 0; for (let k = 0; k <= T; k++) if (tr[p].state[k * n + i] === 2) c++; return c; };
  let sel = 0, gain = -1;
  for (let i = 0; i < n; i++) { const d = gasS("nearest", i) - gasS("ccrp", i); if (d > gain) { gain = d; sel = i; } }
  const setPerson = (i) => { sel = i; $("enter-person").href = `factory.html?person=${i}&pview=1&solo=1${RUNQ ? `&run=${RUNQ}` : ""}`; };
  setPerson(sel);
  const xy = (p, k, i) => { const o = (k * n + i) * 2, c = tr[p].cells; return [(c[o + 1] + 0.5) * cell, (c[o] + 0.5) * cell]; };
  let t = 0, playing = false, lastKey = "";
  const draw = () => {
    const k = Math.min(T, Math.floor(t)), gi = Math.min(gas.length - 1, Math.floor(t / rep.gas.every));
    let u = -1;
    rep.updates.forEach((up, j) => { if (up.t <= t) u = j; });
    if (`${gi}|${u}` !== lastKey) {
      lastKey = `${gi}|${u}`;
      const fr = gas[gi], dz = u >= 0 ? danger[u] : null, d = img.data;
      for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
        const c = iy * nx + ix, o = ((ny - 1 - iy) * nx + ix) * 4, v = fr[c], p = dz ? dz[c] / 255 : 0;
        let rgba = [0, 0, 0, 0];
        if (v >= 128) rgba = [214, 40, 210, 220]; else if (v >= 94) rgba = [255, 170, 0, 160]; else if (v >= 40) rgba = [90, 170, 255, 40 + v];
        if (p >= 0.2 && rgba[3] < 150) rgba = [230, 30, 50, 60 + 140 * p];
        d.set(rgba, o);
      }
      lg.putImageData(img, 0, 0);
    }
    g.drawImage(base, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(layer, 0, 0, W, cv.height);
    const alarm = rep.t_alarm !== null && t >= rep.t_alarm;
    if (alarm && u >= 0) rep.updates[u].source_top.slice(0, 1).forEach((q) => { g.strokeStyle = C.base; g.lineWidth = 3; g.beginPath(); g.arc(X(q.xy[0]), Y(q.xy[1]), 16, 0, 7); g.stroke(); });
    if ($("crowd").checked) for (let i = 0; i < n; i++) {
      const st = tr.ccrp.state[k * n + i];
      if (st === 3) continue;
      const [x, y] = xy("ccrp", k, i);
      g.fillStyle = "rgba(230,238,245,0.55)"; g.fillRect(X(x) - 2, Y(y) - 2, 4, 4);
    }
    for (const [p, col] of PATHS) {
      if (!tr[p]) continue;
      g.strokeStyle = col; g.lineWidth = 5; g.lineJoin = "round"; g.beginPath();
      for (let q = 0; q <= k; q++) { const [x, y] = xy(p, q, sel); q ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); if (tr[p].state[q * n + sel] === 3) break; }
      g.stroke();
      const [x, y] = xy(p, k, sel);
      g.fillStyle = col; g.beginPath(); g.arc(X(x), Y(y), 9, 0, 7); g.fill(); g.strokeStyle = "#fff"; g.lineWidth = 2; g.stroke();
    }
    $("clock").textContent = `${t.toFixed(0)} s`;
    $("time").value = t;
    $("status").innerHTML = (alarm ? `<b style="color:${C.nearest}">Alarm</b> at ${rep.t_alarm} s` : "no alarm yet") +
      (u >= 0 ? ` · source estimate ${rep.updates[u].source_error_m.toFixed(1)} m off` : "");
    const secs = (p) => { let c = 0; for (let q = 0; q <= k; q++) if (tr[p].state[q * n + sel] === 2) c++; return c; };
    $("person").innerHTML = `<b>Worker ${sel}</b> · seconds in harmful gas<br>` + PATHS.map(([p, col, name]) => `<span style="color:${col}">●</span> ${name}: <b>${secs(p)} s</b>`).join("<br>");
    const pc = $("pchart"), c = pc.getContext("2d"), H = pc.height, L = 34, B = 18;
    c.clearRect(0, 0, pc.width, H);
    c.fillStyle = "#8ba3b8"; c.font = "11px system-ui"; c.fillText("concentration at the worker (log scale)", L, 12);
    const PX = (q) => L + (q / T) * (pc.width - L - 6), PY = (v) => 18 + (1 - v / 255) * (H - 18 - B);
    c.strokeStyle = "rgba(214,40,210,0.7)"; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(L, PY(127.5)); c.lineTo(pc.width, PY(127.5)); c.stroke(); c.setLineDash([]);
    c.fillText("toxic", 2, PY(127.5) + 4); c.fillText("time (s)", pc.width - 50, H - 4);
    for (const [p, col] of PATHS) {
      c.strokeStyle = col; c.lineWidth = 2; c.beginPath();
      let on = false;
      for (let q = 0; q <= k; q++) {
        if (tr[p].state[q * n + sel] === 3) { on = false; continue; }
        const o = (q * n + sel) * 2, v = gas[Math.min(gas.length - 1, Math.floor(q / rep.gas.every))][tr[p].cells[o] * nx + tr[p].cells[o + 1]];
        on ? c.lineTo(PX(q), PY(v)) : c.moveTo(PX(q), PY(v)); on = true;
      }
      c.stroke();
    }
  };
  $("time").max = T;
  $("time").oninput = (e) => { t = +e.target.value; playing = false; $("play").textContent = "▶ Play"; };
  $("play").onclick = () => { if (t >= T) t = 0; playing = !playing; $("play").textContent = playing ? "❚❚ Pause" : "▶ Play"; };
  cv.onclick = (e) => {
    const r = cv.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * W, my = ((e.clientY - r.top) / r.height) * cv.height;
    const k = Math.min(T, Math.floor(t));
    let best = null, bd = 40;
    for (let i = 0; i < n; i++) { if (tr.ccrp.state[k * n + i] === 3) continue; const [x, y] = xy("ccrp", k, i), d = Math.hypot(X(x) - mx, Y(y) - my); if (d < bd) { bd = d; best = i; } }
    if (best !== null) setPerson(best);
  };
  let last = performance.now();
  const loop = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) { t = Math.min(T, t + dt * 3); if (t >= T) { playing = false; $("play").textContent = "▶ Play"; } }
    draw();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
