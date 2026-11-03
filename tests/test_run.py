import subprocess
import sys

import numpy as np
from mesh.eval.run import run_episode, POLICIES
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def test_all_scenarios_run_for_all_policies():
    for s in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        for p in POLICIES.values():
            m = run_episode(MeshEnv(make_scenario(s), 0), p())
            assert all(np.isfinite(v) for v in m.values())


def test_ko_beats_m0_under_heavy_load():
    ko = np.mean([run_episode(MeshEnv(make_scenario("S2"), s), POLICIES["ko"]())["utility"] for s in range(5)])
    m0 = np.mean([run_episode(MeshEnv(make_scenario("S2"), s), POLICIES["m0"]())["utility"] for s in range(5)])
    assert ko > m0


def test_cli():
    out = subprocess.run([sys.executable, "-m", "mesh.eval.run", "--scenario", "S1",
                          "--policy", "m0", "--seeds", "2"], capture_output=True, text=True, check=True)
    assert "mean" in out.stdout
