// 3D factory world: loads data/scene.json (exported by `uv run world`) and builds it.
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { Sky } from "three/addons/objects/Sky.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import { animateDetails } from "./details.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { Navigator } from "./controls.js";
import { animateMachines } from "./machines.js";
import { buildWorld } from "./world.js";

const scene = await (await fetch("data/scene.json")).json();
const app = document.getElementById("app");

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25));  // laptop-friendly
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.75;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
app.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
labelRenderer.domElement.className = "labels";
app.appendChild(labelRenderer.domElement);

const world3d = new THREE.Scene();
const span = Math.max(scene.width, scene.height);
const sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(55), THREE.MathUtils.degToRad(-140));
const sky = new Sky();  // physical sky (Preetham) with the same sun as the shadows
sky.scale.setScalar(5000);
sky.material.uniforms.turbidity.value = 5;
sky.material.uniforms.rayleigh.value = 1.2;
sky.material.uniforms.mieCoefficient.value = 0.004;
sky.material.uniforms.mieDirectionalG.value = 0.8;
sky.material.uniforms.sunPosition.value.copy(sunDir);
world3d.add(sky);
world3d.fog = new THREE.Fog("#c9d8e6", span * 3, span * 9);
const pmrem = new THREE.PMREMGenerator(renderer);
world3d.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

world3d.add(new THREE.HemisphereLight("#eef4ff", "#5a5650", 0.55));
const sun = new THREE.DirectionalLight("#fff1dc", 2.4);
sun.position.copy(sunDir).multiplyScalar(span * 1.5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const sc = sun.shadow.camera;
sc.left = sc.bottom = -span * 0.75;
sc.right = sc.top = span * 0.75;
sc.near = 1;
sc.far = span * 4;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
world3d.add(sun);

const { root, parts, F } = buildWorld(scene);
world3d.add(root);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 12000);

// Ambient occlusion and anti-aliasing (toggle "High quality" off on slow machines).
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(world3d, camera));
const gtao = new GTAOPass(world3d, camera, innerWidth, innerHeight);
gtao.updateGtaoMaterial({ radius: 0.6, distanceExponent: 1.5, thickness: 1.5, scale: 1.0 });
gtao.blendIntensity = 0.85;
composer.addPass(gtao);
composer.addPass(new OutputPass());
composer.addPass(new SMAAPass(innerWidth * renderer.getPixelRatio(), innerHeight * renderer.getPixelRatio()));
let highQuality = false;  // ambient occlusion is heavy: opt in
const nav = new Navigator(camera, renderer.domElement, scene, F);
const VIEWS = {
  overview: () => [F.P(-scene.width * 0.15, -scene.height * 0.55, span * 0.55), F.P(scene.width / 2, scene.height / 2, 0)],
  top: () => [F.P(scene.width / 2, scene.height / 2 - 0.01, span * 1.15), F.P(scene.width / 2, scene.height / 2, 0)],
  inside: () => [F.P(scene.width * 0.1, scene.height * 0.15, 4.5), F.P(scene.width * 0.6, scene.height * 0.6, 2)],
};
const [p0, t0] = VIEWS.overview();
camera.position.copy(p0);
nav.orbit.target.copy(t0);

// ------------------------------------------------------------------ panel and keys --
const $ = (id) => document.getElementById(id);
$("info").innerHTML =
  `${scene.width} x ${scene.height} m &middot; ${scene.machines.length} machines ` +
  `(${scene.machines.filter((m) => m.hot).length} hot) &middot; ${scene.sensors.length} gas sensors &middot; ` +
  `${scene.exits.length} exits &middot; ${scene.leak_sites.length} leak sites on the gas pipes &middot; ` +
  `${scene.people.length} workers`;
for (const [id, fn] of Object.entries(VIEWS)) $(`view-${id}`).onclick = () => nav.flyTo(...fn());
const goto = $("goto");
scene.machines.forEach((m, i) => goto.add(new Option(`${m.name}${m.hot ? " (hot)" : ""}`, `m${i}`)));
scene.exits.forEach((e, i) => goto.add(new Option(`exit E${i} (${e.side})`, `e${i}`)));
goto.onchange = () => {
  const v = goto.value;
  if (!v) return;
  const [x, y, h] = v[0] === "m"
    ? (() => { const r = scene.machines[+v.slice(1)].rect; return [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, scene.machines[+v.slice(1)].height]; })()
    : [...scene.exits[+v.slice(1)].xy, 2];
  nav.flyTo(F.P(x - 8, y - 10, h + 8), F.P(x, y, h / 2));
  goto.value = "";
};
const layer = (id, fn) => { $(id).onchange = (e) => fn(e.target.checked); fn($(id).checked); };
layer("show-roof", (on) => { parts.roof.panels.visible = on; });
layer("show-walls", (on) => { parts.outer.cladding.visible = on; });
layer("show-frame", (on) => { parts.roof.frame.visible = on; });
let labelsOn = true;
const labelObjects = [];
root.traverse((o) => { if (o.isCSS2DObject) labelObjects.push(o); });
layer("show-labels", (on) => { labelsOn = on; });
function cullLabels() {  // near the floor, only nearby labels (they would show through walls)
  const low = camera.position.y < 4;
  for (const o of labelObjects) {
    o.visible = labelsOn && (!low || o.getWorldPosition(new THREE.Vector3()).distanceTo(camera.position) < 22);
  }
}
layer("show-grid", (on) => { parts.grid.visible = on; });
layer("show-overhead", (on) => { parts.overhead.visible = on; });
layer("show-people", (on) => { parts.people.visible = on; });
layer("high-quality", (on) => { highQuality = on; });
layer("show-pipes", (on) => { parts.pipes.visible = on; });

$("walk").onclick = () => {
  const e = scene.exits[0];
  const inward = { west: [3, 0], east: [-3, 0], south: [0, 3], north: [0, -3] }[e.side];
  nav.startWalk(e.xy[0] + inward[0], e.xy[1] + inward[1], scene.width / 2, scene.height / 2);
};
$("orbit").onclick = () => nav.startOrbit();
nav.walk.addEventListener("lock", () => { $("walkhelp").hidden = false; $("crosshair").hidden = false; $("show-walls").checked = true; parts.outer.cladding.visible = true; });
nav.walk.addEventListener("unlock", () => { $("walkhelp").hidden = true; $("crosshair").hidden = true; if (nav.mode === "walk") $("resume").hidden = false; });
$("resume").onclick = () => { $("resume").hidden = true; nav.walk.lock(); };
addEventListener("keydown", (e) => {
  if (e.code === "Digit1") nav.flyTo(...VIEWS.overview());
  if (e.code === "Digit2") nav.flyTo(...VIEWS.top());
  if (e.code === "KeyO" && nav.mode === "walk") { $("resume").hidden = true; nav.startOrbit(); }
});

// ----------------------------------------------------------------------- minimap --
const mini = $("minimap"), mg = mini.getContext("2d");
const ms = Math.min(240 / scene.width, 160 / scene.height);
mini.width = Math.round(scene.width * ms);
mini.height = Math.round(scene.height * ms);
const base = document.createElement("canvas");
base.width = mini.width;
base.height = mini.height;
(() => {
  const g = base.getContext("2d"), c = scene.cell * ms;
  g.fillStyle = "#e9ecef";
  g.fillRect(0, 0, base.width, base.height);
  scene.solid.forEach((row, iy) => row.forEach((v, ix) => {
    if (v) { g.fillStyle = "#59606a"; g.fillRect(ix * c, base.height - (iy + 1) * c, c + 0.5, c + 0.5); }
  }));
  for (const m of scene.machines) if (m.hot) {
    g.strokeStyle = "#ff6a00"; g.lineWidth = 2;
    g.strokeRect(m.rect[0] * ms, base.height - m.rect[3] * ms, (m.rect[2] - m.rect[0]) * ms, (m.rect[3] - m.rect[1]) * ms);
  }
  g.strokeStyle = "#e0b000"; g.lineWidth = 2;
  for (const r of scene.pipes) {
    g.beginPath(); g.moveTo(r.from[0] * ms, base.height - r.from[1] * ms); g.lineTo(r.to[0] * ms, base.height - r.to[1] * ms); g.stroke();
  }
  g.fillStyle = "#1e90ff";
  for (const [x, y] of scene.sensors) g.fillRect(x * ms - 2, base.height - y * ms - 2, 4, 4);
  g.fillStyle = "#12a150";
  for (const e of scene.exits) { g.beginPath(); g.arc(e.xy[0] * ms, base.height - e.xy[1] * ms, 4, 0, 7); g.fill(); }
})();
mini.onclick = (ev) => {
  const x = ev.offsetX / ms, y = (mini.height - ev.offsetY) / ms;
  nav.flyTo(F.P(x - 6, y - 9, 9), F.P(x, y, 0));
};
function drawMinimap() {
  mg.drawImage(base, 0, 0);
  const [x, y] = F.toFactory(camera.position);
  const d = new THREE.Vector3();
  camera.getWorldDirection(d);
  const a = Math.atan2(-d.z, d.x);
  mg.save();
  mg.translate(x * ms, mini.height - y * ms);
  mg.rotate(-a);
  mg.fillStyle = "#d6204a";
  mg.beginPath(); mg.moveTo(9, 0); mg.lineTo(-5, -5); mg.lineTo(-5, 5); mg.closePath(); mg.fill();
  mg.restore();
}

// ------------------------------------------------------------------------- loop --
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
});
const clock = new THREE.Clock();
const FRAME = 1 / 30;  // cap at 30 fps: smooth enough, half the GPU work
let since = 0, frame = 0;
renderer.setAnimationLoop(() => {
  since += clock.getDelta();
  if (since < FRAME) return;
  const dt = Math.min(since, 0.1);
  since = 0;
  frame++;
  nav.update(dt);
  animateMachines(parts.machines, clock.elapsedTime);
  animateDetails(parts.overhead, clock.elapsedTime);
  if (frame % 6 === 0) cullLabels();
  if (highQuality) composer.render(); else renderer.render(world3d, camera);
  labelRenderer.render(world3d, camera);
  drawMinimap();
});
window.__world = { scene, nav, camera, parts };  // for debugging in the console
$("loading").remove();
