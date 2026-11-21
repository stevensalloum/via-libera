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


def write_scene(f: Factory, out: Path) -> Path:
    """Write ``scene.json`` for the viewer into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scene.json"
    path.write_text(json.dumps(scene(f)))
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
    p.add_argument("--seed", type=int, default=3, help="random plant seed")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--out", type=Path, default=Path("outputs/world"))
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    a = p.parse_args()
    f = Factory.default()
    if a.plant == "random":
        rng = np.random.default_rng(a.seed)
        f = Factory.random(rng, width=120.0, height=80.0, n_machines=10, n_hot=4,
                           exits_per_side=2, sensor_spacing=15.0)  # fmt: skip
    print("scene:", write_scene(f, a.out))
    server = make_server(a.out, a.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("3D world:", url, "(Ctrl+C to stop)")
    if not a.no_open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0
