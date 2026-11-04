"""Apply the worst-case selection rule to runs/v8/wc/*.json; prints the table and the chosen margin."""
import json
import sys

import numpy as np

from mesh.eval.composite import _mean_score
from mesh.rl.train import VAL_SCENARIOS

sc = list(VAL_SCENARIOS)
OURS = sys.argv[1] if len(sys.argv) > 1 else "v1"  # v1 | egat
L = lambda t: json.load(open(f"runs/v8/wc/{t}.json"))
refs = {"ko": [L("ko")], "m0": [L("m0")]}
refs.update({b: [L(f"{b}_s{k}") for k in range(5)] for b in ("li", "yuan", "q_neutral")})
best = None
for m in ("0", "0.5", "1", "2", "3", "4"):
    v1 = [L(f"{OURS}_s{k}_m{m}") for k in range(5)]
    row = {b: np.mean([_mean_score({"o": v, "r": r}, "o", "r", sc) for v in v1 for r in rs]) for b, rs in refs.items()}
    sw = np.mean([np.mean(v[s]["n_channel_switch"]) + np.mean(v[s]["n_route_change"]) for v in v1 for s in sc])
    worst = min(row.values())
    print(f"margin {m:>3}: " + " ".join(f"{b} {x:+.3f}" for b, x in row.items()) + f" | worst {worst:+.3f} | switches {sw:.0f}")
    if best is None or worst > best[1]:
        best = (m, worst)
print("chosen margin", best[0])
