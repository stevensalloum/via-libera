"""Videos recorded in the viewer are saved into the run's folder."""

import json
import threading
import urllib.request
from pathlib import Path

from tabpfn_spatial.world.serve import make_server


def test_post_saves_a_video(tmp_path: Path) -> None:
    server = make_server(tmp_path, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/save/clip.webm", data=b"abc", method="POST"
    )
    with urllib.request.urlopen(req) as r:
        path = Path(json.loads(r.read())["path"])
    server.shutdown()
    assert path == tmp_path / "videos" / "clip.webm" and path.read_bytes() == b"abc"


def test_post_rejects_other_names(tmp_path: Path) -> None:
    import urllib.error

    import pytest

    server = make_server(tmp_path, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/save/..%2Fevil.py", data=b"x", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(req)
    server.shutdown()
    assert not (tmp_path.parent / "evil.py").exists()
