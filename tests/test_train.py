import subprocess
import sys

import numpy as np
from mesh.rl.train import train, LEARNERS
from mesh.eval.run import run_episode
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def test_train_smoke_all_methods(tmp_path):
    for name in LEARNERS:
        ag = train(name, steps=450, seed=0, n=10, out=str(tmp_path / f"{name}.pt"), log_every=10**9)
        m = run_episode(MeshEnv(make_scenario("S1", n=10), 3), ag)
        assert all(np.isfinite(v) for v in m.values())


def test_checkpoint_roundtrip_cpu(tmp_path):
    for name, cls in LEARNERS.items():
        p = str(tmp_path / f"{name}.pt")
        a = train(name, steps=250, seed=0, n=10, out=p, log_every=10**9)
        b = cls.load(p, device="cpu") if name != "q_neutral" else cls.load(p)
        obs = MeshEnv(make_scenario("S1", n=10), 5).reset()
        assert np.array_equal(a.act(obs).next_hop, b.act(obs).next_hop)


def test_eval_cli_with_checkpoint(tmp_path):
    p = str(tmp_path / "li.pt")
    train("li", steps=250, seed=0, n=10, out=p, log_every=10**9)
    out = subprocess.run([sys.executable, "-m", "mesh.eval.run", "--scenario", "S1", "--policy", "li",
                          "--checkpoint", p, "--seeds", "1", "--n", "10"],
                         capture_output=True, text=True, check=True)
    assert "mean" in out.stdout
