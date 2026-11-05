"""Open-loop calibration: replay the fluid sim's actions and demand into ns-3, compare transport.

python ns3/calibrate.py --scenario S2 --policy ko --seeds 0-4 [--switch-delay 0.1]

The policy acts on the fluid sim's observations; ns-3 receives the same action and the same
per-step demand, so only the transport differs. Runs from Windows (calls ns-3 through wsl) or
inside WSL. Prints per-seed and mean aggregate throughput and loss rate, sim vs ns-3.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mesh.baselines.ko import KoLocalCost
from mesh.baselines.m0 import M0
from mesh.sim.env import MeshEnv
from mesh.sim.radio import CHANNELS
from mesh.sim.scenarios import make_scenario

NS3_ROOT = "~/ns3_validation/ns-3.47"
STEP_S = 0.1
LOG_DIR = Path("runs/ns3/logs")


def ns3_cmd():
    run = f"cd {NS3_ROOT} && LD_LIBRARY_PATH=build/lib ./build/scratch/ns3.47-mesh_bridge"
    if platform.system() == "Windows":
        return ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", run]
    return ["bash", "-lc", run]


def episode(scenario, policy_name, seed, switch_delay, route_hold):
    cfg = make_scenario(scenario)
    env = MeshEnv(cfg, seed)
    policy = {"ko": KoLocalCost, "m0": M0}[policy_name]()
    obs = env.reset()
    p = subprocess.Popen(ns3_cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         stderr=open(LOG_DIR / f"ns3_err_{scenario}_{policy_name}_{seed}.log", "w"))

    def ask(msg):
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"ns-3 exited (code {p.wait()}), see {LOG_DIR}/ns3_err_{scenario}_{policy_name}_{seed}.log")
        reply = json.loads(line)
        if reply["type"] == "error":
            raise RuntimeError(reply["msg"])
        return reply

    ask(dict(type="init", n=cfg.n, pos=env.top.pos.tolist(), gateways=list(env.gw), channels=list(CHANNELS),
             step_s=STEP_S, pkt_bytes=1200, queue_bytes=int(5 * cfg.mean_mbps * 1e6 * STEP_S / 8),
             switch_delay_s=switch_delay, route_hold_s=route_hold, ext_interf=None, seed=seed))
    sim_thr, ns_thr, demand, ns_lost = [], [], [], 0.0
    sim_lost = 0.0
    while True:
        a = policy.act(obs)
        obs, _, info = env.step(a)  # draws this step's demand; failure (S6) applies within the step
        r = ask(dict(type="step", t=env.t - 1, channels=np.asarray(a.channels).tolist(),
                     next_hop=np.asarray(env.next_hop).tolist(), demand_mbps=env.demand.tolist(),
                     alive=env.alive.astype(int).tolist()))
        sim_thr.append(info["aggregate"])
        ns_thr.append(sum(r["delivered_mbps"]))
        demand.append(info["demand_total"])
        sim_lost += info["lost"]
        ns_lost += sum(sum(v) for v in r["drop_bytes"].values()) * 8 / STEP_S / 1e6
        if info["done"]:
            break
    p.stdin.write(json.dumps({"type": "end"}) + "\n")
    p.stdin.close()
    p.wait()
    d = max(sum(demand), 1e-9)
    return dict(seed=seed, sim_thr=float(np.mean(sim_thr)), ns_thr=float(np.mean(ns_thr)),
                sim_loss=sim_lost / d, ns_loss=ns_lost / d)


def seeds_arg(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--policy", default="ko", choices=["ko", "m0"])
    ap.add_argument("--seeds", type=seeds_arg, default=seeds_arg("0-4"))
    ap.add_argument("--switch-delay", type=float, default=0.1)
    ap.add_argument("--route-hold", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda s: episode(a.scenario, a.policy, s, a.switch_delay, a.route_hold), a.seeds))
    print(f"{a.scenario} {a.policy}  seed  sim_thr  ns3_thr  diff%  sim_loss  ns3_loss")
    for r in rows:
        print(f"{r['seed']:>4}  {r['sim_thr']:8.2f} {r['ns_thr']:8.2f} {100 * (r['ns_thr'] / r['sim_thr'] - 1):6.1f}"
              f"  {r['sim_loss']:8.3f} {r['ns_loss']:8.3f}")
    st, nt = np.mean([r["sim_thr"] for r in rows]), np.mean([r["ns_thr"] for r in rows])
    print(f"mean  {st:8.2f} {nt:8.2f} {100 * (nt / st - 1):6.1f}  "
          f"{np.mean([r['sim_loss'] for r in rows]):8.3f} {np.mean([r['ns_loss'] for r in rows]):8.3f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
