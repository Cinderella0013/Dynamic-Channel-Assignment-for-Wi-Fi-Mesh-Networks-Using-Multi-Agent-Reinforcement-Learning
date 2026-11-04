"""Fairness ablation (spec sixth amendment): worst-case margin for Li / Yuan with hysteresis.
usage: python runs/v8/fair_pick.py li|yuan"""
import json
import sys

import numpy as np

from mesh.eval.composite import _mean_score
from mesh.rl.train import VAL_SCENARIOS

sc, who = list(VAL_SCENARIOS), sys.argv[1]
L = lambda t: json.load(open(f"runs/v8/wc/{t}.json"))
refs = {"ko": [L("ko")], "m0": [L("m0")]}
refs.update({b: [L(f"{b}_s{k}") for k in range(5)] for b in ("li", "yuan", "q_neutral") if b != who})
best = None
for m in ("0", "0.25", "0.5", "1", "2", "4", "8", "16"):
    ours = [L(f"{who}h_s{k}_m{m}") for k in range(5)]
    row = {b: np.mean([_mean_score({"o": v, "r": r}, "o", "r", sc) for v in ours for r in rs]) for b, rs in refs.items()}
    sw = np.mean([np.mean(v[s]["n_channel_switch"]) + np.mean(v[s]["n_route_change"]) for v in ours for s in sc])
    worst = min(row.values())
    print(f"{who} margin {m:>4}: " + " ".join(f"{b} {x:+.3f}" for b, x in row.items()) + f" | worst {worst:+.3f} | switches {sw:.0f}")
    if best is None or worst > best[1]:
        best = (m, worst)
print("chosen margin", best[0])
