"""Pick switch_margin for egat (v10 sim) on the validation set, composite vs Ko, as runs/v8/hyst_val.py.
usage: python runs/v10/hyst_val.py <training seed> <margin> <mac_eff> <contention> -> appends to runs/v10/hyst_val.txt"""
import sys
from mesh.baselines.ko import KoLocalCost
from mesh.eval.composite import _mean_score
from mesh.rl import train
from mesh.rl.train import LEARNERS, VAL_SCENARIOS, _val_results
k, m = int(sys.argv[1]), float(sys.argv[2])
train.SCEN_SET.update(airtime="domain", mac_eff=float(sys.argv[3]), contention=float(sys.argv[4]))
a = LEARNERS["egat"].load(f"runs/v10/ckpt/egat_s{k}.pt", device="cpu")
a.cfg.switch_margin = m
res = {"ko": _val_results(KoLocalCost(), 20), "e": _val_results(a, 20)}
sc = list(VAL_SCENARIOS)
st = sum(sum(res["e"][s]["n_channel_switch"]) + sum(res["e"][s]["n_route_change"]) for s in sc) / (len(sc) * 4)
with open("runs/v10/hyst_val.txt", "a") as f:
    f.write(f"{k} {m} {_mean_score(res, 'e', 'ko', sc):+.4f} {st:.1f}\n")
