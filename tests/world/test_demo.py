import sys

from tabpfn_spatial.world import serve


def test_demo_ships_a_saved_replay_and_lists_it(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["via-libera", "--list"])
    assert serve.demo() == 0
    names = capsys.readouterr().out.split()
    assert names and all((serve.DEMO / n / "replay.json").exists() for n in names)
