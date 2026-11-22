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

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always the latest export
        super().end_headers()

    def log_message(self, *args) -> None:  # keep the terminal quiet
        return


def make_server(data: Path, port: int = 8770) -> ThreadingHTTPServer:
    """HTTP server on 127.0.0.1 (``port=0`` picks a free one)."""
    handler = functools.partial(_Handler, data=data)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


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
    p.add_argument("--out", type=Path, default=Path("outputs/world"))
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    p.add_argument("--replay", action="store_true", help="run one emergency and play it in 3D")
    p.add_argument("--keep", action="store_true", help="serve what is already exported")
    p.add_argument("--rung", default="B", help="replay scenario A-E, or shaped L, U, T, K")
    p.add_argument("--leak", type=int, default=500, help="replay leak seed")
    p.add_argument("--models", default="tabpfn", choices=("tabpfn", "perfect", "baseline"))
    p.add_argument(
        "--backend", default="baseline", help="tabpfn models: baseline (offline) or cloud"
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
    if a.keep:
        print("serving", a.out)
    elif a.replay:
        from tabpfn_spatial.evac.crowd_study import find_rung
        from tabpfn_spatial.world.replay import make_replay

        rung = find_rung(a.rung)
        print(f"running {rung.label}, leak {a.leak}, models {a.models} (about a minute)...")
        print("replay:", write_replay(make_replay(rung, a.leak, a.models, a.backend), a.out))
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
