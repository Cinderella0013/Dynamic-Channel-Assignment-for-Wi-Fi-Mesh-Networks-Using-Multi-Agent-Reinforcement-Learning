"""Run methods x scenarios x seeds into a JSON file.

python -m mesh.eval.suite --suite A,B,C --seeds 5 --out runs/v6/suite_ko.json --method ko --method yuan=runs/v6/yuan_s0.pt
"""
import argparse
import json
import subprocess

from mesh.eval.run import POLICIES, run_episode
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import SUITE_A, SUITE_B, SUITE_C, make_scenario

SUITES = {"A": SUITE_A, "B": SUITE_B, "C": SUITE_C}


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def run_suite(methods, scenarios, seeds, out=None):
    results = {}
    for name, make in methods.items():
        policy = make()
        for s in scenarios:
            rows = [run_episode(MeshEnv(make_scenario(s), seed), policy) for seed in seeds]
            results.setdefault(name, {})[s] = {k: [r[k] for r in rows] for k in rows[0]}
            print(f"{name} {s} reward {sum(r['reward'] for r in rows) / len(rows):.1f}", flush=True)
    if out:
        json.dump({"commit": _commit(), "results": results}, open(out, "w"), indent=1)
    return results


def _factory(spec):
    name, _, path = spec.partition("=")
    if not path:
        return name, POLICIES[name]
    from mesh.rl.train import LEARNERS  # torch only when a learner is evaluated
    trained = LEARNERS[name].load(path)
    return name, lambda: trained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="A,B,C")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--method", action="append", required=True, help="m0 | ko | <learner>=<checkpoint>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scenarios = [s for key in a.suite.split(",") for s in SUITES[key]]
    run_suite(dict(_factory(m) for m in a.method), scenarios, range(a.seeds), a.out)


if __name__ == "__main__":
    main()
