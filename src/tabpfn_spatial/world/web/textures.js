// Procedural textures drawn on canvases (no image files to ship).
import * as THREE from "three";

function canvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d")];
}

function toTexture(c, repeat = [1, 1]) {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(...repeat);
  t.anisotropy = 8;
  return t;
}

// deterministic noise so every load looks the same
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// Concrete floor with slab joints, walkway lines around machines, hatched ignition
// zones and green exit pads. One canvas pixel = 1 / ppm metre.
export function floorTexture(scene) {
  const { width: W, height: H } = scene;
  const ppm = Math.min(20, 4096 / Math.max(W, H));
  const [c, g] = canvas(Math.round(W * ppm), Math.round(H * ppm));
  const X = (x) => x * ppm;
  const Y = (y) => (H - y) * ppm;
  g.fillStyle = "#8b8d8c";
  g.fillRect(0, 0, c.width, c.height);
  const r = rng(7);
  for (let i = 0; i < W * H * 6; i++) {
    const v = 110 + Math.floor(r() * 50);
    g.fillStyle = `rgba(${v},${v},${v - 3},${0.08 + r() * 0.12})`;
    g.fillRect(r() * c.width, r() * c.height, 1 + r() * 3, 1 + r() * 3);
  }
  for (let i = 0; i < W * H * 0.02; i++) {  // oil and tyre stains
    g.fillStyle = `rgba(40,40,40,${0.04 + r() * 0.05})`;
    g.beginPath();
    g.ellipse(r() * c.width, r() * c.height, 8 + r() * 30, 4 + r() * 14, r() * 3, 0, 7);
    g.fill();
  }
  g.strokeStyle = "rgba(50,50,50,0.35)";  // slab joints every 6 m
  g.lineWidth = Math.max(1, ppm * 0.03);
  for (let x = 6; x < W; x += 6) { g.beginPath(); g.moveTo(X(x), 0); g.lineTo(X(x), c.height); g.stroke(); }
  for (let y = 6; y < H; y += 6) { g.beginPath(); g.moveTo(0, Y(y)); g.lineTo(c.width, Y(y)); g.stroke(); }

  const cell = scene.cell;
  g.save();  // ignition zones: orange hatch
  for (const [x, y] of scene.ignition) {
    g.fillStyle = "rgba(255,120,20,0.16)";
    g.fillRect(X(x - cell / 2), Y(y + cell / 2), cell * ppm, cell * ppm);
  }
  g.restore();
  g.strokeStyle = "rgba(255,110,0,0.45)";
  g.lineWidth = Math.max(1, ppm * 0.08);
  for (const [x, y] of scene.ignition) {
    for (let k = -1; k <= 1; k += 0.5) {
      g.beginPath();
      g.moveTo(X(x - cell / 2), Y(y - cell / 2 + (k + 0.5) * cell));
      g.lineTo(X(x + cell / 2), Y(y + cell / 2 + (k - 0.5) * cell));
      g.stroke();
    }
  }
  g.strokeStyle = "#e8c21a";  // walkway lines 0.8 m around each machine
  g.lineWidth = ppm * 0.12;
  for (const m of scene.machines) {
    const [x0, y0, x1, y1] = m.rect;
    g.strokeRect(X(x0 - 0.8), Y(y1 + 0.8), (x1 - x0 + 1.6) * ppm, (y1 - y0 + 1.6) * ppm);
  }
  for (const e of scene.exits) {  // green exit pads with an arrow pointing out
    const [x, y] = e.xy;
    g.fillStyle = "rgba(20,160,70,0.55)";
    g.fillRect(X(x - 1.5), Y(y + 1.5), 3 * ppm, 3 * ppm);
    const out = { west: [-1, 0], east: [1, 0], south: [0, -1], north: [0, 1] }[e.side];
    g.strokeStyle = "white";
    g.lineWidth = ppm * 0.18;
    g.beginPath();
    g.moveTo(X(x - out[0]), Y(y - out[1]));
    g.lineTo(X(x + out[0]), Y(y + out[1]));
    g.stroke();
  }
  return toTexture(c);
}

export function corrugatedTexture(color = "#a9b1b8") {
  const [c, g] = canvas(256, 256);
  g.fillStyle = color;
  g.fillRect(0, 0, 256, 256);
  for (let x = 0; x < 256; x += 16) {
    const grad = g.createLinearGradient(x, 0, x + 16, 0);
    grad.addColorStop(0, "rgba(255,255,255,0.18)");
    grad.addColorStop(0.5, "rgba(0,0,0,0.12)");
    grad.addColorStop(1, "rgba(255,255,255,0.18)");
    g.fillStyle = grad;
    g.fillRect(x, 0, 16, 256);
  }
  return toTexture(c);
}

export function blockTexture(base = "#9a9a96", brick = false) {
  const [c, g] = canvas(256, 256);
  g.fillStyle = base;
  g.fillRect(0, 0, 256, 256);
  const r = rng(brick ? 3 : 5);
  const bw = brick ? 48 : 96, bh = brick ? 20 : 48;
  for (let y = 0, row = 0; y < 256; y += bh, row++) {
    for (let x = -(row % 2) * bw / 2; x < 256; x += bw) {
      const d = Math.floor((r() - 0.5) * 24);
      g.fillStyle = `rgba(${128 + d},${brick ? 70 + d : 128 + d},${brick ? 55 + d : 124 + d},0.25)`;
      g.fillRect(x + 2, y + 2, bw - 4, bh - 4);
    }
    g.fillStyle = "rgba(40,40,40,0.35)";
    g.fillRect(0, y, 256, 2);
  }
  return toTexture(c);
}

export function signTexture(text, bg = "#0f8a3c", fg = "white") {
  const [c, g] = canvas(512, 160);
  g.fillStyle = bg;
  g.fillRect(0, 0, 512, 160);
  g.strokeStyle = fg;
  g.lineWidth = 8;
  g.strokeRect(8, 8, 496, 144);
  g.fillStyle = fg;
  g.font = "bold 92px Helvetica, Arial, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 256, 84);
  return toTexture(c);
}

export function hazardStripes() {
  const [c, g] = canvas(128, 128);
  g.fillStyle = "#f2c200";
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = "#1b1b1b";
  for (let k = -128; k < 256; k += 32) {
    g.beginPath();
    g.moveTo(k, 0); g.lineTo(k + 16, 0); g.lineTo(k + 16 - 128, 128); g.lineTo(k - 128, 128);
    g.fill();
  }
  return toTexture(c);
}
