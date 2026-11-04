"""Pick switch_margin for egat_v1 on the validation set (A+B x seeds 1000-1003), composite vs Ko.
usage: python runs/v8/hyst_val.py <training seed> <margin>  -> one line appended to runs/v8/hyst_val.txt"""
import sys

from mesh.baselines.ko import KoLocalCost
from mesh.eval.composite import _mean_score
from mesh.rl.train import LEARNERS, VAL_SCENARIOS, _val_results

k, m = int(sys.argv[1]), float(sys.argv[2])
a = LEARNERS["egat_v1"].load(f"runs/v8/ckpt/egat_v1_s{k}.pt", device="cpu")
a.cfg.switch_margin = m
res = {"ko": _val_results(KoLocalCost(), 20), "v1": _val_results(a, 20)}
sc = list(VAL_SCENARIOS)
st = sum(sum(res["v1"][s]["n_channel_switch"]) + sum(res["v1"][s]["n_route_change"]) for s in sc) / (len(sc) * 4)
with open("runs/v8/hyst_val.txt", "a") as f:
    f.write(f"{k} {m} {_mean_score(res, 'v1', 'ko', sc):+.4f} {st:.1f}\n")
