import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ns3"))
import report  # noqa: E402

KEYS = ["aggregate", "jain", "p5", "utility", "n_channel_switch", "n_route_change", "n_drop", "n_interrupted",
        "reward", "loss_rate"]
SEEDS = [0, 1, 2]


def _runs(thr, sw, loss, rng):
    n = len(SEEDS)
    r = {k: list(np.full(n, 0.5) + rng.normal(0, 0.01, n)) for k in KEYS}
    r["aggregate"] = list(thr + rng.normal(0, 0.1, n))
    r["n_channel_switch"] = [float(sw)] * n
    r["n_route_change"] = [2.0] * n
    r["loss_rate"] = [loss] * n
    return r


def _write(root, scale, sub):
    rng = np.random.default_rng(0)
    base = {"egat_h3": (20, 40, 0.01), "ko": (18, 500, 0.03), "m0": (15, 0, 0.05)}
    for k in range(2):
        d = os.path.join(root, f"t{k}", sub)
        os.makedirs(d)
        for name, (thr, sw, loss) in base.items():
            res = {name: {s: _runs(thr * scale, sw, loss, rng) for s in ["S2", "B1", "B3"]}}
            json.dump({"commit": "x", "seeds": SEEDS, "results": res}, open(os.path.join(d, f"suite_{name}.json"), "w"))


def test_report(tmp_path):
    _write(str(tmp_path / "ns3"), 1.1, "")  # ns-3 10% above sim: gate passes
    _write(str(tmp_path / "sim"), 1.0, "a")
    ns3, seeds = report.load(str(tmp_path / "ns3"))
    sim, found = report.load(str(tmp_path / "sim"), "a", seeds=set(seeds))
    assert seeds == found == SEEDS
    txt = report.build(ns3, sim)
    assert "| ko |" in txt and "| m0 |" in txt and "+10." in txt and "passed" in txt
    assert "NOT passed" not in txt


def test_gate_fails(tmp_path):
    _write(str(tmp_path / "ns3"), 1.5, "")
    _write(str(tmp_path / "sim"), 1.0, "a")
    ns3, _ = report.load(str(tmp_path / "ns3"))
    sim, _ = report.load(str(tmp_path / "sim"), "a")
    assert "NOT passed" in report.build(ns3, sim)
