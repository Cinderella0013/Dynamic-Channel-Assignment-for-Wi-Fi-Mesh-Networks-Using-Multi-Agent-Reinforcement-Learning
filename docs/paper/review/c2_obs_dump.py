"""Print the 38-entry agent observation of one env instance (review comment 2).
Run from E:\\MESH:  PYTHONPATH=. python docs/paper/review/c2_obs_dump.py"""
from dataclasses import replace

import numpy as np

from mesh.model.agent import EGATAgent
from mesh.rl.agents_io import F_M, F_O, K_SLOTS, agent_features, neighbor_slots
from mesh.sim.env import MeshEnv
from mesh.sim.radio import CHANNELS
from mesh.sim.scenarios import make_scenario

NAMES = (["offered_load", "unmet", "is_gateway", "hop_to_gw/n", "alive", "backlog"]
         + [f"dl_ch=={c}" for c in CHANNELS]
         + [f"parent_slot=={j}" for j in range(K_SLOTS)] + ["parent_slot==none"]
         + [f"slot{j}.{k}" for j in range(K_SLOTS) for k in ("valid", "snr/40", "rate/143.4", "load_frac", "ack")])

cfg = replace(make_scenario("S2"), airtime="domain", mac_eff=0.65)  # runs/v10/calib.env
env = MeshEnv(cfg, 0)
obs = env.reset()
ko = EGATAgent(seed=0, device="cpu").ko
for _ in range(5):  # a few Ko steps so load/backlog/ack are non-trivial
    obs, r, info = env.step(ko.act(obs))
slots = neighbor_slots(obs)
o, m = agent_features(obs, slots)
x = np.concatenate([o, m], 1)
assert (F_O, F_M) == (18, 20) and x.shape == (cfg.n, 38) and len(NAMES) == 38
print("F_O", F_O, "F_M", F_M, "x", x.shape, "q_max", 5 * cfg.mean_mbps)
gws = set(obs.gateways)
u = next(i for i in range(cfg.n) if i not in gws and obs.node_feat[i, 3] >= 2)  # a node >= 2 hops out
print(f"node {u}  gateways {sorted(gws)}  slots {slots[u].tolist()}  next_hop {obs.next_hop[u]}")
for k, (name, val) in enumerate(zip(NAMES, x[u])):
    print(f"{k:2d} {name:20s} {val:9.4f}")
print("progress mask", EGATAgent(seed=0, device="cpu").progress_mask(obs, slots)[u].tolist())
g = min(gws)
print(f"gateway {g}: slots {slots[g].tolist()} next_hop {obs.next_hop[g]} channels {obs.channels[g].tolist()}")
print("reward", round(r, 3), "n_sw", info["n_channel_switch"], "n_rc", info["n_route_change"], "n_drop", info["n_drop"])
