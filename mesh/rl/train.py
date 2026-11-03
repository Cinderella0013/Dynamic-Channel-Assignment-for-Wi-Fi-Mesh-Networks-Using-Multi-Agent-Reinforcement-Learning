"""Train an RL baseline: python -m mesh.rl.train --method yuan --steps 100000 --out runs/yuan_s0.pt"""
import argparse
import copy
import time

import numpy as np

from mesh.model.agent import EGATAgent, EGATv1Agent
from mesh.model.v2 import EGATv2Agent
from mesh.rl.agents_io import per_agent_reward
from mesh.rl.li_gat_a2c import LiAgent
from mesh.rl.q_neutral import QNeutral
from mesh.rl.yuan_gqn import YuanAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario

LEARNERS = {"q_neutral": QNeutral, "yuan": YuanAgent, "li": LiAgent, "egat": EGATAgent, "egat_v1": EGATv1Agent, "egat_v2": EGATv2Agent}
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")
VAL_SEEDS = (1000, 1001, 1002)  # validation topologies; test seeds are 0-4


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


def validate(agent, n):
    """Mean greedy reward on the validation set (evaluation only: no learning, no budget)."""
    from mesh.eval.run import run_episode
    return float(np.mean([run_episode(MeshEnv(make_scenario(s, n=n), vs), agent)["reward"]
                          for s in SCENARIOS for vs in VAL_SEEDS]))


def train(method, steps, seed=0, n=20, out=None, log_every=5000, val_every=None, **agent_kwargs):
    rng = np.random.default_rng(seed)
    agent = LEARNERS[method](seed=seed, **agent_kwargs)
    lam_of = getattr(agent, "reward_lam", None)
    if hasattr(agent, "set_budget"):
        agent.set_budget(steps)
    best, best_v = None, -np.inf
    due, last_phase = False, getattr(agent, "phase", None)

    def checkpoint(step):
        nonlocal best, best_v
        v = validate(agent, n)
        if best is None or v > best_v:  # also guards a NaN validation score
            best, best_v = snapshot(agent), v
        print(f"{method} val step {step} reward {v:.1f} (best {best_v:.1f})", flush=True)
    env = obs = None
    ep_rewards, ep_r, t0 = [], 0.0, time.time()
    for step in range(1, steps + 1):
        if obs is None:
            env = MeshEnv(make_scenario(str(rng.choice(SCENARIOS)), n=n), int(rng.integers(2**31)))
            obs = env.reset()
        action = agent.act(obs, explore=True)
        a_idx = agent.last_action_idx
        next_obs, r, info = env.step(action)
        lam = lam_of(step, steps) if lam_of else 0.5
        agent.learn(obs, a_idx, per_agent_reward(obs, action, info, r, env.w, lam=lam), next_obs, info["done"])
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
    a = ap.parse_args()
    train(a.method, a.steps, a.seed, a.n, a.out, a.log_every, a.val_every)


if __name__ == "__main__":
    main()
