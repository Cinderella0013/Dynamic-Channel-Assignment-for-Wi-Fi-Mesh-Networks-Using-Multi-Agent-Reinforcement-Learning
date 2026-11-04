"""Validation probe (A+B x seeds 1000-1003): switch hysteresis variants for egat_v1, plus Li and Ko references.
usage: python runs/v8/hval_probe.py ko | li <k> | v1 <k> <mode plain|child> <margin>   -> runs/v8/hval/<tag>.json"""
import json
import sys

import torch

from mesh.baselines.ko import KoLocalCost
from mesh.rl.train import LEARNERS, _val_results

kind = sys.argv[1]
if kind == "ko":
    tag, pol = "ko", KoLocalCost()
elif kind == "li":
    k = int(sys.argv[2]); tag = f"li_s{k}"
    pol = LEARNERS["li"].load(f"runs/v8/ckpt/li_s{k}.pt", device="cpu")
else:
    k, mode, m = int(sys.argv[2]), sys.argv[3], float(sys.argv[4]); tag = f"v1_s{k}_{mode}_{m}"
    pol = LEARNERS["egat_v1"].load(f"runs/v8/ckpt/egat_v1_s{k}.pt", device="cpu")
    pol.cfg.switch_margin = 0.0
    orig_t, orig_act = pol._tensors, pol.act
    state = {}

    def act(obs, explore=False):
        state["obs"] = obs
        return orig_act(obs, explore)

    def tensors(items, h_prev, net=None):
        out = dict(orig_t(items, h_prev, net))
        if net is None and len(items) == 1 and m:
            p, obs = items[0], state["obs"]
            nh = torch.as_tensor(obs.next_hop)
            kids = torch.zeros(len(nh)).index_add_(0, nh.clamp(min=0), (nh >= 0).float())
            w = (1 + kids) if mode == "child" else torch.ones(len(nh))
            cur = torch.as_tensor((p["slots"] == obs.next_hop[:, None]) & (obs.next_hop[:, None] >= 0))
            out["route_logits"] = out["route_logits"] + m * cur[None]
            oh = torch.nn.functional.one_hot(torch.as_tensor(p["cur_chan"]), out["chan_logits"].shape[-1])
            out["chan_logits"] = out["chan_logits"] + m * (w[:, None] * oh)[None]
        return out
    pol.act, pol._tensors = act, tensors
json.dump(_val_results(pol, 20), open(f"runs/v8/hval/{tag}.json", "w"))
