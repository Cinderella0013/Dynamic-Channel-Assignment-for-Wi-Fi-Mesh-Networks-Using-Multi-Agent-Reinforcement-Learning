"""Worst-case margin selection (spec amendment, third): validation results on A+B x seeds 1000-1003.
usage: python runs/v8/wc_val.py ko|m0  |  <learner> <k>  |  v1 <k> <margin>   -> runs/v8/wc/<tag>.json"""
import json
import sys

from mesh.eval.run import POLICIES
from mesh.rl.train import LEARNERS, _val_results

a = sys.argv[1:]
if len(a) == 1:
    tag, pol = a[0], POLICIES[a[0]]()
elif a[0] == "v1":
    tag = f"v1_s{a[1]}_m{a[2]}"
    pol = LEARNERS["egat_v1"].load(f"runs/v8/ckpt/egat_v1_s{a[1]}.pt", device="cpu")
    pol.cfg.switch_margin = float(a[2])
else:
    tag, pol = f"{a[0]}_s{a[1]}", LEARNERS[a[0]].load(f"runs/v8/ckpt/{a[0]}_s{a[1]}.pt")
json.dump(_val_results(pol, 20), open(f"runs/v8/wc/{tag}.json", "w"))
