"""B4 probe: congestion-aware hysteresis on validation (A+B x seeds 1000-1003), settings a and b.
usage: python runs/v8/qb_probe.py <a|b> ko | <a|b> li <k> | <a|b> egat <k> <base|queue> -> runs/v8/qb/<tag>.json"""
import json
import sys
from dataclasses import replace

import numpy as np

from mesh.baselines.ko import KoLocalCost
from mesh.eval.run import run_episode
from mesh.rl.train import LEARNERS, VAL_SCENARIOS, VAL_SEEDS
from mesh.sim.env import NODE_FEATS, MeshEnv
from mesh.sim.scenarios import make_scenario

st, a = sys.argv[1], sys.argv[2:]
over = {"queue_preserve": True} if st == "b" else {}
if a[0] == "ko":
    tag, pol = "ko", KoLocalCost()
elif a[0] == "li":
    tag, pol = f"li_s{a[1]}", LEARNERS["li"].load(f"runs/v8/ckpt/li_s{a[1]}.pt", device="cpu")
else:
    tag = f"egat_s{a[1]}_{a[2]}"
    pol = LEARNERS["egat"].load(f"runs/v8/ckpt/egat_s{a[1]}.pt", device="cpu")
    pol.cfg.switch_margin = 3.0
    if a[2] == "queue":  # a node whose own queue is building may switch freely
        bi = NODE_FEATS.index("backlog")
        pol.margin_for = lambda obs: np.where(obs.node_feat[:, bi] > 0, 0.0, 3.0)
out = {}
for s in VAL_SCENARIOS:
    rows = [run_episode(MeshEnv(replace(make_scenario(s), **over), vs), pol) for vs in VAL_SEEDS]
    out[s] = {k: [r[k] for r in rows] for k in rows[0]}
json.dump(out, open(f"runs/v8/qb/{st}_{tag}.json", "w"))
