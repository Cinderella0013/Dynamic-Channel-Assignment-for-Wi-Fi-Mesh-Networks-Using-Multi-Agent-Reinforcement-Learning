"""Worst-case margin selection (spec amendment, third): validation results on A+B x seeds 1000-1003.
usage: python runs/v10/wc_val.py ko|m0  |  <learner> <k>  |  v1|egat|nogru <k> <margin>   -> runs/v10/wc/<tag>.json"""
import json
import sys

from mesh.eval.run import POLICIES
from mesh.rl import train
from mesh.rl.train import LEARNERS, _val_results
train.SCEN_SET.update(airtime="domain", mac_eff=0.65, contention=0.0)  # v10 sim

a = sys.argv[1:]
if len(a) == 1:
    tag, pol = a[0], POLICIES[a[0]]()
elif a[0] in ("v1", "egat", "nogru") and len(a) == 3:
    tag = f"{a[0]}_s{a[1]}_m{a[2]}"
    name = {"v1": "egat_v1", "egat": "egat", "nogru": "egat_nogru"}[a[0]]
    pol = LEARNERS["egat_v1" if a[0] == "v1" else "egat"].load(f"runs/v10/ckpt/{name}_s{a[1]}.pt", device="cpu")
    pol.cfg.switch_margin = float(a[2])
elif a[0] in ("li", "yuan") and len(a) == 3:  # fairness ablation: baseline with hysteresis
    tag = f"{a[0]}h_s{a[1]}_m{a[2]}"
    pol = LEARNERS[a[0]].load(f"runs/v10/ckpt/{a[0]}_s{a[1]}.pt", device="cpu")
    pol.switch_margin = float(a[2])
else:
    tag, pol = f"{a[0]}_s{a[1]}", LEARNERS[a[0]].load(f"runs/v10/ckpt/{a[0]}_s{a[1]}.pt")
json.dump(_val_results(pol, 20), open(f"runs/v10/wc/{tag}.json", "w"))
