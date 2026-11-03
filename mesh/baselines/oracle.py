"""Clairvoyant lookahead oracle. PRIVILEGED: reads the live env, including future traffic.
Training-time teacher and headroom reference only; never an evaluated method.

Every t_c steps it starts from Ko's action and makes one coordinate pass over the nodes: per node it tries
every downlink channel and every valid parent slot, keeping a candidate if the env reward summed over
`horizon` steps on a deepcopy of the env improves. Between rounds it holds its action, except that it
follows Ko's immediate reroute when a parent breaks.
"""
import copy

import numpy as np

from mesh.baselines.ko import KoLocalCost
from mesh.rl.agents_io import N_DL_CH, _valid_slots, neighbor_slots
from mesh.sim.env import Action


def rollout(env, action, horizon):
    """Summed env reward of holding `action` for `horizon` steps on a copy of `env`."""
    e = copy.deepcopy(env)
    tot = 0.0
    for _ in range(horizon):
        _, r, info = e.step(Action(action.channels.copy(), action.next_hop.copy(), action.queue_preserve))
        tot += r
        if info["done"]:
            break
    return tot


def _with_channel(a, u, c):
    ch = a.channels.copy()
    ch[u, 1] = c
    ch[a.next_hop == u, 0] = c  # children's uplink radio follows the parent's downlink
    return Action(ch, a.next_hop.copy())


def _with_parent(a, u, p):
    ch, nh = a.channels.copy(), a.next_hop.copy()
    nh[u], ch[u, 0] = p, ch[p, 1]
    return Action(ch, nh)


class LookaheadOracle:
    def __init__(self, env, horizon=10, t_c=10):
        self.env, self.horizon = env, horizon
        self.ko, self.last = KoLocalCost(t_c=t_c), None

    def act(self, obs):
        ko = self.ko.act(obs)
        if obs.t % self.ko.t_c and self.last is not None:
            if (ko.next_hop != obs.next_hop).any():  # Ko's broken-parent reaction
                self.last = ko
            return self.last
        best, best_v = ko, rollout(self.env, ko, self.horizon)
        slots = neighbor_slots(obs)
        valid = _valid_slots(obs, slots)
        for u in range(len(slots)):
            if best.next_hop[u] < 0:
                continue  # gateway or dead node
            cands = [_with_channel(best, u, c) for c in range(1, N_DL_CH + 1) if c != best.channels[u, 1]]
            cands += [_with_parent(best, u, int(slots[u, j])) for j in np.flatnonzero(valid[u])
                      if slots[u, j] != best.next_hop[u]]
            for a in cands:
                v = rollout(self.env, a, self.horizon)
                if v > best_v:
                    best, best_v = a, v
        self.last = best
        return best
