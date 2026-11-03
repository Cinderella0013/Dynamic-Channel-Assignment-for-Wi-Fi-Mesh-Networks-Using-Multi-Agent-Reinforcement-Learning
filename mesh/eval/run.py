"""Run baselines on a scenario: python -m mesh.eval.run --scenario S2 --policy ko --seeds 5"""
import argparse

import numpy as np

from mesh.baselines.ko import KoLocalCost
from mesh.baselines.m0 import M0
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario

POLICIES = {"m0": M0, "ko": KoLocalCost}
MEAN_KEYS = ("aggregate", "jain", "p5", "utility")
SUM_KEYS = ("n_channel_switch", "n_route_change", "n_drop")


def run_episode(env, policy):
    obs = env.reset()
    rows, reward = [], 0.0
    while True:
        obs, r, info = env.step(policy.act(obs))
        rows.append(info)
        reward += r
        if info["done"]:
            break
    out = {k: float(np.mean([i[k] for i in rows])) for k in MEAN_KEYS}
    out.update({k: float(sum(i[k] for i in rows)) for k in SUM_KEYS})
    out["reward"] = reward
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kind", default="rgg")
    a = ap.parse_args()
    cfg = make_scenario(a.scenario, n=a.n, kind=a.kind)
    results = [run_episode(MeshEnv(cfg, s), POLICIES[a.policy]()) for s in range(a.seeds)]
    keys = list(results[0])
    print("seed  " + "  ".join(f"{k:>16}" for k in keys))
    for s, r in enumerate(results):
        print(f"{s:<4}  " + "  ".join(f"{r[k]:16.3f}" for k in keys))
    print("mean  " + "  ".join(f"{np.mean([r[k] for r in results]):16.3f}" for k in keys))


if __name__ == "__main__":
    main()
