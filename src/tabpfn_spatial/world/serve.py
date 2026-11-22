"""``uv run world``: export the plant and serve the 3D viewer on http://127.0.0.1:8770.

The viewer (``world/web``) is static HTML + JavaScript (Three.js from a pinned CDN
version, so the first load needs internet). ``/data/`` serves the exported scene from
the output folder. Stop with Ctrl+C.
"""

import argparse
import contextlib
import functools
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.world.export import scene

WEB = Path(__file__).parent / "web"


def clean(obj):
    """JSON-safe copy: NaN and infinities become null (browsers reject NaN)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_replay(rep: dict, out: Path) -> Path:
    """Write ``replay.json`` (and the matching ``scene.json``) for the viewer."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.json").write_text(json.dumps(clean(rep["scene"]), allow_nan=False))
    path = out / "replay.json"
    path.write_text(json.dumps(clean(rep), allow_nan=False))
    return path


def write_scene(f: Factory, out: Path, people: int = 0, seed: int = 0) -> Path:
    """Write ``scene.json`` for the viewer into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scene.json"
    path.write_text(json.dumps(scene(f, people, seed)))
    return path


class _Handler(SimpleHTTPRequestHandler):
    """Static files from ``WEB``; ``/data/...`` from the output folder."""

    def __init__(self, *args, data: Path, **kwargs) -> None:
        self.data = data
        super().__init__(*args, directory=str(WEB), **kwargs)

    def translate_path(self, path: str) -> str:
        if path.split("?")[0].startswith("/data/"):
            return str(self.data / path.split("?")[0][len("/data/") :])
        return super().translate_path(path)

    def do_POST(self) -> None:
        """``POST /save/<name>.webm``: a video recorded in the viewer, into ``<data>/videos/``."""
        import re
        from urllib.parse import parse_qs, unquote, urlsplit

        url = urlsplit(self.path)
        name = unquote(url.path.removeprefix("/save/"))
        run = parse_qs(url.query).get("run", [""])[0]
        if run and not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", run):
            self.send_error(400, "bad run name")
            return
        size = int(self.headers.get("Content-Length", 0))
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}\.webm", name) or not 0 < size < 2_000_000_000:
            self.send_error(400, "expected /save/<name>.webm with a body")
            return
        folder = (self.data / run if run else self.data) / "videos"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / name
        target.write_bytes(self.rfile.read(size))
        body = json.dumps({"path": str(target)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always the latest export
        super().end_headers()

    def log_message(self, *args) -> None:  # keep the terminal quiet
        return


def make_server(data: Path, port: int = 8770) -> ThreadingHTTPServer:
    """HTTP server on 127.0.0.1; if ``port`` is taken (another world running), a free one."""
    handler = functools.partial(_Handler, data=data)
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        if port == 0:
            raise
        print(f"port {port} is in use (another world still running?): using a free port")
        return ThreadingHTTPServer(("127.0.0.1", 0), handler)


def main() -> int:
    """CLI: ``uv run world [--plant default|random] [--seed 3] [--port 8770] [--no-open]``."""
    p = argparse.ArgumentParser(description="Serve the 3D factory world.")
    p.add_argument("--plant", choices=("default", "random"), default="default")
    p.add_argument(
        "--layout",
        default=None,
        help="shaped plant: a template (L-plant, U-plant, T-plant, campus) or a kind "
        "(L, U, T, campus) drawn with --seed (unseen shapes)",
    )
    p.add_argument("--seed", type=int, default=3, help="random plant seed")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--people", type=int, default=80, help="workers in the hall")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="folder to write and serve (default: one folder per plant or replay in outputs/world)",
    )
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    p.add_argument("--replay", action="store_true", help="run one emergency and play it in 3D")
    p.add_argument("--keep", action="store_true", help="serve what is already exported")
    p.add_argument("--rung", default="B", help="replay scenario A-E, or shaped L, U, T, K")
    p.add_argument("--leak", type=int, default=500, help="replay leak seed")
    p.add_argument("--models", default="tabpfn", choices=("tabpfn", "perfect", "baseline"))
    p.add_argument(
        "--physics",
        default="3d",
        choices=("2d", "3d"),
        help="3d: gas, alarm and the team's 3D models (final pipeline)",
    )
    p.add_argument("--n-train", type=int, default=200, help="3D training leaks per model")
    p.add_argument(
        "--backend",
        default="auto",
        help="team models: auto (cloud TabPFN when a token is set), cloud, or baseline (offline)",
    )
    a = p.parse_args()
    f = Factory.default()
    if a.layout:
        from tabpfn_spatial.sim.layouts import KINDS, TEMPLATES, shaped, template

        if a.layout in TEMPLATES:
            f = template(a.layout)
        elif a.layout in KINDS:
            f = shaped(np.random.default_rng(a.seed), a.layout, name=f"{a.layout} seed {a.seed}")
        else:
            p.error(f"--layout must be one of {list(TEMPLATES) + list(KINDS)}")
    elif a.plant == "random":
        rng = np.random.default_rng(a.seed)
        f = Factory.random(rng, width=120.0, height=80.0, n_machines=10, n_hot=4,
                           exits_per_side=2, sensor_spacing=15.0)  # fmt: skip
    if a.out is None:  # one folder per run: parallel runs never overwrite each other
        base = Path("outputs/world")
        if a.keep:
            runs = [d for d in base.glob("*") if (d / "scene.json").exists()]
            runs += [base] if (base / "scene.json").exists() else []
            a.out = max(runs, key=lambda d: (d / "scene.json").stat().st_mtime) if runs else base
        elif a.replay:
            from tabpfn_spatial.core.models import pick_backend

            a.backend = pick_backend(a.backend) if a.models == "tabpfn" else "baseline"
            tag = a.backend if a.models == "tabpfn" else "sim"
            a.out = base / f"replay-{a.rung}-leak{a.leak}-{a.models}-{tag}-{a.physics}"
        else:
            a.out = base / (f.name.replace(" ", "-") if f.name else "plant")
    elif a.replay and a.models == "tabpfn":
        from tabpfn_spatial.core.models import pick_backend

        a.backend = pick_backend(a.backend)
    if a.keep:
        print("serving", a.out)
    elif a.replay:
        from tabpfn_spatial.evac.crowd_study import find_rung
        from tabpfn_spatial.world.replay import make_replay

        rung = find_rung(a.rung)
        print(
            f"running {rung.label}, leak {a.leak}, models {a.models} ({a.backend}); "
            "a few minutes with the team's models..."
        )
        print(
            "replay:",
            write_replay(
                make_replay(
                    rung, a.leak, a.models, a.backend, physics=a.physics, n_train=a.n_train
                ),
                a.out,
            ),
        )
    else:
        (a.out / "replay.json").unlink(missing_ok=True)
        print("scene:", write_scene(f, a.out, a.people, a.seed))
    server = make_server(a.out, a.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("3D world:", url, "(Ctrl+C to stop)")
    if not a.no_open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


DEMO = Path(__file__).resolve().parents[3] / "demo"  # saved replays shipped with the repo


def demo() -> int:
    """CLI: ``uv run via-libera [name]``: the presentation in one command.

    Serves a replay saved in ``demo/`` (default: the newest), so nothing is computed and no
    tokens are spent; opens the browser. ``uv run via-libera --list`` lists them.
    """
    import sys

    runs = sorted((d for d in DEMO.glob("*") if (d / "replay.json").exists()),
                  key=lambda d: (d / "replay.json").stat().st_mtime)  # fmt: skip
    args = sys.argv[1:]
    if not runs:
        print(f"no saved replay in {DEMO}; make one: uv run world --replay --rung T")
        return 1
    if args and args[0] == "--list":
        print("\n".join(d.name for d in runs))
        return 0
    pick = next((d for d in runs if args and d.name == args[0]), runs[-1])
    extra = [a for a in args if a != pick.name]
    sys.argv = ["world", "--keep", "--out", str(pick), *extra]
    print(f"Via Libera demo: {pick.name}")
    return main()
