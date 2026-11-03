"""Train an RL baseline: python -m mesh.rl.train --method yuan --steps 100000 --out runs/yuan_s0.pt"""
import argparse
import time

import numpy as np

from mesh.rl.agents_io import per_agent_reward
from mesh.rl.li_gat_a2c import LiAgent
from mesh.rl.q_neutral import QNeutral
from mesh.rl.yuan_gqn import YuanAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario

LEARNERS = {"q_neutral": QNeutral, "yuan": YuanAgent, "li": LiAgent}
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")


def train(method, steps, seed=0, n=20, out=None, log_every=5000, **agent_kwargs):
    rng = np.random.default_rng(seed)
    agent = LEARNERS[method](seed=seed, **agent_kwargs)
    lam_of = getattr(agent, "reward_lam", None)
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
        if info["done"]:
            ep_rewards.append(ep_r)
            ep_r, obs = 0.0, None
        if step % log_every == 0:
            recent = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
            print(f"{method} step {step}/{steps}  episodes {len(ep_rewards)}  "
                  f"mean reward (last 10) {recent:.1f}  {time.time() - t0:.0f}s", flush=True)
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
    a = ap.parse_args()
    train(a.method, a.steps, a.seed, a.n, a.out, a.log_every)


if __name__ == "__main__":
    main()
