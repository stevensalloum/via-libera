import json
import threading
import urllib.request
from pathlib import Path

from tabpfn_spatial.sim import Factory
from tabpfn_spatial.world.serve import WEB, make_server, write_scene


def test_serves_the_viewer_and_the_scene(tmp_path: Path) -> None:
    write_scene(Factory.default(), tmp_path)
    server = make_server(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(f"{base}/").read().decode()
        assert "<canvas" in html or "main.js" in html
        js = urllib.request.urlopen(f"{base}/main.js").read().decode()
        assert "three" in js
        scene = json.loads(urllib.request.urlopen(f"{base}/data/scene.json").read())
        assert scene["width"] == 60
    finally:
        server.shutdown()
    assert (WEB / "index.html").exists()
