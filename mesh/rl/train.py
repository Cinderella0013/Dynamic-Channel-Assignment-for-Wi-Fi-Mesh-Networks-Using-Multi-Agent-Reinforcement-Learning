"""Train an RL baseline: python -m mesh.rl.train --method yuan --steps 100000 --out runs/yuan_s0.pt"""
import argparse
import copy
import time

import numpy as np
import torch

from mesh.baselines.ko import KoLocalCost
from mesh.baselines.oracle import LookaheadOracle
from mesh.model.agent import EGATAgent, EGATv1Agent
from mesh.model.v2 import EGATv2Agent
from mesh.rl.agents_io import per_agent_reward
from mesh.rl.li_gat_a2c import LiAgent
from mesh.rl.q_neutral import QNeutral
from mesh.rl.yuan_gqn import YuanAgent
from mesh.sim.env import MeshEnv
from mesh.eval.composite import _mean_score
from mesh.sim.scenarios import SUITE_A, SUITE_B, make_scenario

LEARNERS = {"q_neutral": QNeutral, "yuan": YuanAgent, "li": LiAgent, "egat": EGATAgent, "egat_v1": EGATv1Agent, "egat_v2": EGATv2Agent}
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")
VAL_SCENARIOS = SUITE_A + SUITE_B  # selection sees every scenario the pre-registered composite is reported on
VAL_SEEDS = (1000, 1001, 1002, 1003)  # validation topologies; evaluation seeds are 0-19
EMA_DECAY = 0.9995  # per env step after BC (~2000-step window); candidates carry these averaged weights


_BUFFERS = ("buf", "bc_buf", "mem", "replay")


def snapshot(agent):
    """Deep copy of the agent without its experience buffers (a v2 replay alone is GBs)."""
    held = {k: getattr(agent, k) for k in _BUFFERS if hasattr(agent, k)}
    for k in held:
        setattr(agent, k, [])
    try:
        return copy.deepcopy(agent)
    finally:
        for k, v in held.items():
            setattr(agent, k, v)


_KO_VAL = {}


def _val_results(policy, n):
    from mesh.eval.run import run_episode
    out = {}
    for s in VAL_SCENARIOS:
        rows = [run_episode(MeshEnv(make_scenario(s, n=n), vs), policy) for vs in VAL_SEEDS]
        out[s] = {k: [r[k] for r in rows] for k in rows[0]}
    return out


def validate(agent, n):
    """Pre-registered composite vs Ko, mean over VAL_SCENARIOS on held-out topologies (no learning, no budget):
    checkpoints are selected on the metric the evaluation reports, not on raw env reward."""
    key = (VAL_SCENARIOS, VAL_SEEDS, n)
    if key not in _KO_VAL:
        _KO_VAL[key] = _val_results(KoLocalCost(), n)
    return _mean_score({"ours": _val_results(agent, n), "ko": _KO_VAL[key]}, "ours", "ko", list(VAL_SCENARIOS))


def _ema_update(ema, net):
    sd = net.state_dict()
    if ema is None:
        return {k: v.detach().clone() for k, v in sd.items()}
    with torch.no_grad():
        for k, v in sd.items():
            if v.is_floating_point():
                ema[k].lerp_(v, 1.0 - EMA_DECAY)
            else:
                ema[k].copy_(v)
    return ema


def train(method, steps, seed=0, n=20, out=None, log_every=5000, val_every=None, teacher="ko", **agent_kwargs):
    """teacher="oracle": the BC phase imitates LookaheadOracle (privileged, training only) instead of Ko."""
    rng = np.random.default_rng(seed)
    agent = LEARNERS[method](seed=seed, **agent_kwargs)
    if teacher == "oracle" and not (hasattr(agent, "phase") and hasattr(agent, "ko")):
        raise ValueError(f"{method} has no behaviour-cloning phase to take an oracle teacher")
    lam_of = getattr(agent, "reward_lam", None)
    if hasattr(agent, "set_budget"):
        agent.set_budget(steps)
    best, best_v = None, -np.inf
    due, last_phase = False, getattr(agent, "phase", None)

    ema = None

    def checkpoint(step):
        nonlocal best, best_v
        cand = snapshot(agent)
        if ema is not None:
            cand.net.load_state_dict(ema)
        v = validate(cand, n)
        if best is None or v > best_v:  # also guards a NaN validation score
            best, best_v = cand, v
        if out:
            cand.save(out[:-3] + f"_val{step}.pt" if out.endswith(".pt") else f"{out}_val{step}")
        print(f"{method} val step {step} composite {v:+.4f} (best {best_v:+.4f}) ema {ema is not None}", flush=True)
    env = obs = None
    ep_rewards, ep_r, t0 = [], 0.0, time.time()
    for step in range(1, steps + 1):
        if obs is None:
            env = MeshEnv(make_scenario(str(rng.choice(SCENARIOS)), n=n), int(rng.integers(2**31)))
            obs = env.reset()
            if teacher == "oracle" and agent.phase == "bc":
                # bound to this episode's env (it reads future traffic); scores moves under the learner's queue rule
                agent.ko = LookaheadOracle(env, queue_preserve=bool(getattr(agent.cfg, "queue_preserve", False)))
        action = agent.act(obs, explore=True)
        a_idx = agent.last_action_idx
        next_obs, r, info = env.step(action)
        lam = lam_of(step, steps) if lam_of else 0.5
        agent.learn(obs, a_idx, per_agent_reward(obs, action, info, r, env.w, lam=lam), next_obs, info["done"])
        if teacher == "oracle" and (agent.phase != "bc" or info["done"]) and not isinstance(agent.ko, KoLocalCost):
            # after BC, and between episodes (validation), the prior/fallback must be Ko: the oracle cannot run at test time
            agent.ko = KoLocalCost()
        if val_every and hasattr(agent, "net") and getattr(agent, "phase", None) != "bc":
            ema = _ema_update(ema, agent.net)  # starts at the post-BC weights
        ep_r += r
        obs = next_obs
        if val_every:
            phase = getattr(agent, "phase", None)
            if phase is not None and phase != last_phase and last_phase == "bc":
                due = True  # the end of behaviour cloning is a validation point too
            last_phase = phase
            due = due or step % val_every == 0
        if info["done"]:
            ep_rewards.append(ep_r)
            ep_r, obs = 0.0, None
            if due:  # validate only between episodes, so recurrent/cooldown state is not disturbed
                checkpoint(step)
                due = False
        if step % log_every == 0:
            recent = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
            print(f"{method} step {step}/{steps}  episodes {len(ep_rewards)}  "
                  f"mean reward (last 10) {recent:.1f}  {time.time() - t0:.0f}s", flush=True)
    if val_every:
        checkpoint(steps)  # final policy is a candidate as well
        agent = best
    if out:
        agent.save(out)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=LEARNERS)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--teacher", choices=("ko", "oracle"), default="ko")
    ap.add_argument("--agent-set", action="append", default=[], help="learner config key=value (JSON value), e.g. ablations")
    a = ap.parse_args()
    from mesh.eval.suite import _kv
    train(a.method, a.steps, a.seed, a.n, a.out, a.log_every, a.val_every, teacher=a.teacher, **_kv(a.agent_set))


if __name__ == "__main__":
    main()
