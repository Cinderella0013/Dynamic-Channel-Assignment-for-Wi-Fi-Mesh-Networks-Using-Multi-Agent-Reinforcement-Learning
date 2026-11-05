import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from mesh.baselines.ko import KoLocalCost
from mesh.eval.run import run_episode
from mesh.ns3.env import Ns3MeshEnv
from mesh.ns3.fake import FakeTransport
from mesh.ns3.transport import ProcessTransport, TransportError
from mesh.sim.env import Action, MeshEnv
from mesh.sim.scenarios import make_scenario


@pytest.mark.parametrize("scenario", ["S2", "B1", "S6"])
def test_fake_episode_matches_meshenv(scenario):
    cfg = make_scenario(scenario)
    ref = run_episode(MeshEnv(cfg, 0), KoLocalCost())
    got = run_episode(Ns3MeshEnv(cfg, 0, FakeTransport()), KoLocalCost())
    assert got.keys() == ref.keys()
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-6, abs=1e-6), k


def test_fake_step_observation_matches_meshenv():
    cfg = replace(make_scenario("S2"), episode_len=30)
    a, b = MeshEnv(cfg, 3), Ns3MeshEnv(cfg, 3, FakeTransport())
    oa, ob = a.reset(), b.reset()
    pol_a, pol_b = KoLocalCost(), KoLocalCost()
    for _ in range(30):
        oa, ra, ia = a.step(pol_a.act(oa))
        ob, rb, ib = b.step(pol_b.act(ob))
        assert rb == pytest.approx(ra, rel=1e-6)
        np.testing.assert_allclose(ob.node_feat, oa.node_feat, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ob.edge_attr, oa.edge_attr, rtol=1e-5, atol=1e-5)
        assert set(ib) == set(ia)
    assert b.transport.closed  # done step sends end


def test_init_message_fields():
    cfg = make_scenario("S4")
    env = Ns3MeshEnv(cfg, 7, FakeTransport())
    env.reset()
    m = env.transport.init_msg
    assert m["type"] == "init" and m["n"] == 20 and len(m["pos"]) == 20
    assert m["channels"] == [1, 6, 11, 36, 40, 44, 48]
    assert m["queue_bytes"] == pytest.approx(5 * 6.0 * 1e6 * 0.1 / 8)
    assert m["ext_interf"] == {"channel_idx": 1, "start": 50, "end": 150, "noise_dbm": -85.0}
    assert m["switch_delay_s"] == pytest.approx(0.1) and m["seed"] == 7
    json.dumps(m)  # serialisable


FAKE_BIN = r'''
import json, sys
mode = sys.argv[1]
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "init":
        n = m["n"]
        print(json.dumps({"type": "ready"}), flush=True)
    elif m["type"] == "step":
        if mode == "error":
            print(json.dumps({"type": "error", "msg": "boom"}), flush=True); sys.exit(1)
        if mode == "crash":
            sys.stderr.write("segfault-ish\n"); sys.exit(3)
        z = [0.0] * n
        print(json.dumps({"type": "result", "t": m["t"], "delivered_mbps": m["demand_mbps"], "queue_bytes": z,
                          "src_queue_bytes": z, "drop_bytes": {"overflow": z, "mac": z, "switch": z, "noroute": z},
                          "links": [[0, 1, 9, 1, 30.0, 100.0]]}), flush=True)
    elif m["type"] == "end":
        sys.exit(0)
'''


def _proc(tmp_path, mode):
    script = tmp_path / "bin.py"
    script.write_text(FAKE_BIN)
    return ProcessTransport(cmd=[sys.executable, str(script), mode])


def test_process_transport_roundtrip(tmp_path):
    cfg = replace(make_scenario("S2"), episode_len=3)
    env = Ns3MeshEnv(cfg, 0, _proc(tmp_path, "ok"))
    obs = env.reset()
    for _ in range(3):
        obs, r, info = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
    assert info["done"] and info["lost"] == 0.0
    assert env.transport.proc.returncode == 0


@pytest.mark.parametrize("mode,match", [("error", "boom"), ("crash", "segfault-ish")])
def test_process_transport_raises(tmp_path, mode, match):
    env = Ns3MeshEnv(make_scenario("S2"), 0, _proc(tmp_path, mode))
    obs = env.reset()
    with pytest.raises(TransportError, match=match):
        env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
    env.close()
    assert env.transport.proc.poll() is not None  # reaped


def test_suite_cli_fake_end_to_end(tmp_path, monkeypatch, capsys):
    from mesh.eval import composite
    from mesh.ns3 import suite
    outs = []
    for m in ("m0", "ko"):
        out = tmp_path / f"suite_{m}.json"
        monkeypatch.setattr(sys, "argv", ["x", "--fake", "--method", m, "--scenarios", "S2,B3", "--seeds", "2",
                                          "--seed-start", "1", "--set", "episode_len=20", "--workers", "2",
                                          "--out", str(out)])
        suite.main()
        outs.append(str(out))
    d = json.load(open(outs[1]))
    assert d["seeds"] == [1, 2] and set(d["results"]["ko"]) == {"S2", "B3"}
    ref = run_episode(MeshEnv(replace(make_scenario("B3"), episode_len=20), 2), KoLocalCost())
    assert d["results"]["ko"]["B3"]["aggregate"][1] == pytest.approx(ref["aggregate"], rel=1e-6)
    assert "ko" in composite.report(outs, ours="m0")
