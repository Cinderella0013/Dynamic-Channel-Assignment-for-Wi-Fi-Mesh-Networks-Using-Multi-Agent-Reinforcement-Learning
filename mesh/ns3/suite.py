"""Run methods x scenarios x seeds on ns-3 into the run_suite JSON schema (read by mesh.eval.composite).

python -m mesh.ns3.suite --scenarios S2,B1,B3 --seeds 5 --workers 8 --method ko --out runs/ns3/a/t0/suite_ko.json
"""
import argparse
import json
from dataclasses import replace
from multiprocessing import Pool

from mesh.eval.run import run_episode
from mesh.eval.suite import _commit, _factory, _kv
from mesh.ns3.env import Ns3MeshEnv
from mesh.ns3.fake import FakeTransport
from mesh.ns3.transport import DEFAULT_BINARY, ProcessTransport
from mesh.sim.scenarios import make_scenario

_policies = {}  # per worker process: method spec -> policy


def _episode(job):
    spec, agent_overrides, scenario, seed, overrides, binary, fake, delays = job
    if spec not in _policies:
        _policies[spec] = _factory(spec, agent_overrides)[1]()
    cfg = replace(make_scenario(scenario), **overrides)
    env = Ns3MeshEnv(cfg, seed, FakeTransport() if fake else ProcessTransport(binary), **delays)
    try:
        return run_episode(env, _policies[spec])
    finally:
        env.close()


def run_ns3_suite(methods, scenarios, seeds, out=None, overrides=None, agent_overrides=None,
                  binary=DEFAULT_BINARY, fake=False, workers=1, delays=None):
    overrides, agent_overrides, delays = overrides or {}, agent_overrides or {}, delays or {}
    jobs = [(m, agent_overrides, s, seed, overrides, binary, fake, delays)
            for m in methods for s in scenarios for seed in seeds]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_episode, jobs, chunksize=1)
    else:
        rows = [_episode(j) for j in jobs]
    results, i = {}, 0
    for m in methods:
        name = m.partition("=")[0]
        for s in scenarios:
            rs = rows[i:i + len(seeds)]
            i += len(seeds)
            results.setdefault(name, {})[s] = {k: [r[k] for r in rs] for k in rs[0]}
            print(f"{name} {s} reward {sum(r['reward'] for r in rs) / len(rs):.1f}", flush=True)
    if out:
        json.dump({"commit": _commit(), "overrides": overrides, "agent_overrides": agent_overrides,
                   "seeds": list(seeds), "results": results,
                   "ns3": {"binary": "fake" if fake else binary, **delays}}, open(out, "w"), indent=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="S2,B1,B3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--method", action="append", required=True, help="m0 | ko | <learner>=<checkpoint>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="ScenarioConfig override key=value (JSON value)")
    ap.add_argument("--agent-set", action="append", default=[], help="learner config override key=value (JSON value)")
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--fake", action="store_true", help="fluid-model FakeTransport instead of ns-3")
    ap.add_argument("--workers", type=int, default=1, help="parallel episodes (one ns-3 process each)")
    ap.add_argument("--switch-delay", type=float, help="ns-3 ChannelSwitchDelay s (default downtime_steps * 0.1)")
    ap.add_argument("--route-hold", type=float, help="uplink hold after a parent change, s (default as above)")
    a = ap.parse_args()
    delays = {k: v for k, v in (("switch_delay_s", a.switch_delay), ("route_hold_s", a.route_hold)) if v is not None}
    run_ns3_suite(a.method, a.scenarios.split(","), range(a.seed_start, a.seed_start + a.seeds), a.out,
                  _kv(a.set), _kv(a.agent_set), a.binary, a.fake, a.workers, delays)


if __name__ == "__main__":
    main()
