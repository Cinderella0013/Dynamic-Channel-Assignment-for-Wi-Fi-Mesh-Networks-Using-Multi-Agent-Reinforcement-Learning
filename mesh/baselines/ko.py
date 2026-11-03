"""Ko local-cost adaptation (this project's reading; see spec section 1.2).

Acts every t_c steps and holds its last action in between, unless a parent link
breaks (parent dead or no usable rate), which triggers an immediate re-decision.
Routing: Bellman-Ford toward gateways on the physical link graph, link cost
(1 + load_frac) / snr_rate (rate the SNR allows on any channel). Hysteresis:
a node switches parent only if the new cost is < SWITCH_GAIN x current cost.
Channels (tree, uplink/downlink split): radio 1 = node's downlink channel,
chosen as the channel with least local cost sum over non-child conflict
neighbours j using it of (1 + offered_load_j); radio 0 = parent's downlink
channel. Nodes are visited by hop count so parents decide first.
"""
import numpy as np

from mesh.sim.env import NODE_FEATS, Action, edge_matrix
from mesh.sim.radio import N_CHANNELS, rate_mbps

SWITCH_GAIN = 0.9


def ko_route_via(obs, alive):
    """Ko's Bellman-Ford toward gateways: via[u, v] = w(u, v) + dist(v), inf if unusable."""
    n = len(alive)
    link = (edge_matrix(obs, "is_link") > 0) & alive[:, None] & alive[None, :]
    rate = np.where(link, rate_mbps(edge_matrix(obs, "snr_db")), 0.0)
    usable = rate > 0
    w = np.where(usable, (1 + edge_matrix(obs, "load_frac")) / np.where(usable, rate, 1.0), np.inf)
    gw = [g for g in obs.gateways if alive[g]]
    dist = np.full(n, np.inf)
    dist[gw] = 0.0
    for _ in range(n):
        new = np.minimum(dist, (w + dist[None, :]).min(axis=1))
        new[gw] = 0.0
        if np.array_equal(new, dist):
            break
        dist = new
    return w + dist[None, :]


class KoLocalCost:
    def __init__(self, t_c=10):
        self.t_c = t_c

    def act(self, obs):
        alive = obs.node_feat[:, NODE_FEATS.index("alive")] > 0
        if obs.t % self.t_c and not self._parent_broken(obs, alive):
            return Action(obs.channels.copy(), obs.next_hop.copy())
        next_hop = self._route(obs, alive)
        return Action(self._assign_channels(obs, next_hop, alive), next_hop)

    @staticmethod
    def _parent_broken(obs, alive):
        """True if some live node's parent died or its link lost every usable rate."""
        rate = edge_matrix(obs, "rate_mbps")
        us = np.flatnonzero(alive & (obs.next_hop >= 0))
        ps = obs.next_hop[us]
        return bool((~alive[ps] | (rate[us, ps] <= 0)).any())

    def _route(self, obs, alive):
        n = len(alive)
        via = ko_route_via(obs, alive)
        dist = via.min(axis=1)
        dist[[g for g in obs.gateways if alive[g]]] = 0.0
        next_hop = np.argmin(via, axis=1)
        for u in range(n):
            cur = obs.next_hop[u]
            if 0 <= cur < n and np.isfinite(via[u, cur]) and via[u, next_hop[u]] >= SWITCH_GAIN * via[u, cur]:
                next_hop[u] = cur
        next_hop[~np.isfinite(dist)] = -1
        next_hop[list(obs.gateways)] = -1
        # ponytail: hysteresis can in theory keep a parent whose own route moved closer
        # to u; a loop then shows up as n_drop. Add a loop check if that is ever non-zero.
        return next_hop

    def _assign_channels(self, obs, next_hop, alive):
        n = len(alive)
        load = obs.node_feat[:, NODE_FEATS.index("offered_load")]
        conflict = edge_matrix(obs, "is_conflict") > 0
        channels = obs.channels.copy()
        depth = np.zeros(n, dtype=int)
        for u in range(n):
            v, d = u, 0
            while next_hop[v] >= 0 and d <= n:
                v, d = next_hop[v], d + 1
            depth[u] = d
        for u in sorted(np.flatnonzero(alive), key=lambda u: (depth[u], u)):
            p = next_hop[u]
            if p >= 0:
                channels[u, 0] = channels[p, 1]
            others = np.flatnonzero(conflict[u] & alive & (next_hop != u))
            costs = [((channels[others] == c).any(axis=1) * (1 + load[others])).sum()
                     for c in range(1, N_CHANNELS)]
            channels[u, 1] = 1 + int(np.argmin(costs))
        return channels
