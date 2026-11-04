"""Run methods x scenarios x seeds into a JSON file.

python -m mesh.eval.suite --suite A,B,C --seeds 5 --out runs/v6/suite_ko.json --method ko --method yuan=runs/v6/yuan_s0.pt
"""
import argparse
import json
import subprocess
from dataclasses import replace

from mesh.eval.run import POLICIES, run_episode
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import SUITE_A, SUITE_B, SUITE_C, make_scenario

SUITES = {"A": SUITE_A, "B": SUITE_B, "C": SUITE_C}


def _commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def run_suite(methods, scenarios, seeds, out=None, overrides=None, agent_overrides=None):
    """overrides: ScenarioConfig fields applied to every scenario (e.g. queue_preserve, downtime_steps).
    agent_overrides: learner config fields set at evaluation (e.g. switch_margin); recorded in the output."""
    overrides = overrides or {}
    results = {}
    for name, make in methods.items():
        policy = make()
        for s in scenarios:
            cfg = replace(make_scenario(s), **overrides)
            rows = [run_episode(MeshEnv(cfg, seed), policy) for seed in seeds]
            results.setdefault(name, {})[s] = {k: [r[k] for r in rows] for k in rows[0]}
            print(f"{name} {s} reward {sum(r['reward'] for r in rows) / len(rows):.1f}", flush=True)
    if out:
        json.dump({"commit": _commit(), "overrides": overrides, "agent_overrides": agent_overrides or {},
                   "seeds": list(seeds), "results": results}, open(out, "w"), indent=1)
    return results


def _factory(spec, agent_overrides=None):
    name, _, path = spec.partition("=")
    if not path:
        return name, POLICIES[name]
    from mesh.rl.train import LEARNERS  # torch only when a learner is evaluated
    base = max((k for k in LEARNERS if name.startswith(k)), key=len, default=None)  # egat_v1_oracle -> egat_v1
    if base is None:
        raise ValueError(f"unknown learner label {name!r}")
    trained = LEARNERS[base].load(path)
    for k, v in (agent_overrides or {}).items():
        target = trained.cfg if hasattr(getattr(trained, "cfg", None), k) else trained  # EGAT config or agent attribute
        if not hasattr(target, k):
            raise ValueError(f"unknown learner config field {k!r}")
        setattr(target, k, v)
    return name, lambda: trained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="A,B,C")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0, help="first eval seed (20 = fresh confirmatory seeds)")
    ap.add_argument("--method", action="append", required=True, help="m0 | ko | <learner>=<checkpoint>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="ScenarioConfig override key=value (JSON value)")
    ap.add_argument("--agent-set", action="append", default=[], help="learner config override key=value (JSON value)")
    a = ap.parse_args()
    scenarios = [s for key in a.suite.split(",") for s in SUITES[key]]
    overrides, agent_overrides = _kv(a.set), _kv(a.agent_set)
    run_suite(dict(_factory(m, agent_overrides) for m in a.method), scenarios, range(a.seed_start, a.seed_start + a.seeds), a.out, overrides,
              agent_overrides)


def _kv(pairs):
    out = {}
    for kv in pairs:
        k, _, v = kv.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
