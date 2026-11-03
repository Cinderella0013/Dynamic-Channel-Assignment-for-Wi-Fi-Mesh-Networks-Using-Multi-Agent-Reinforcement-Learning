"""Ko local-cost adaptation (this project's reading; see spec section 1.2).

Channel: radio 0 stays on common channel 0 (connectivity); every t_c steps each
node, in id order, puts radio 1 on the channel with the lowest local cost
sum over conflict neighbours j using that channel of (1 + offered_load_j).
Routing: Bellman-Ford toward gateways with link cost (1 + load_frac) / rate.
"""
import numpy as np

from mesh.sim.env import NODE_FEATS, Action, edge_matrix
from mesh.sim.radio import N_CHANNELS


class KoLocalCost:
    def __init__(self, t_c=10):
        self.t_c = t_c

    def act(self, obs):
        alive = obs.node_feat[:, NODE_FEATS.index("alive")] > 0
        channels = obs.channels.copy()
        if obs.t % self.t_c == 0:
            channels = self._assign_channels(obs, channels, alive)
        return Action(channels, self._route(obs, alive))

    def _assign_channels(self, obs, channels, alive):
        load = obs.node_feat[:, NODE_FEATS.index("offered_load")]
        conflict = edge_matrix(obs, "is_conflict") > 0
        channels[:, 0] = 0
        for u in np.flatnonzero(alive):
            nbrs = np.flatnonzero(conflict[u] & alive)
            costs = [((channels[nbrs] == c).any(axis=1) * (1 + load[nbrs])).sum()
                     for c in range(1, N_CHANNELS)]
            channels[u, 1] = 1 + int(np.argmin(costs))
        return channels

    def _route(self, obs, alive):
        n = len(alive)
        rate = edge_matrix(obs, "rate_mbps")
        load_frac = edge_matrix(obs, "load_frac")
        usable = (rate > 0) & alive[:, None] & alive[None, :]
        w = np.where(usable, (1 + load_frac) / np.where(usable, rate, 1.0), np.inf)
        dist = np.full(n, np.inf)
        gw = [g for g in obs.gateways if alive[g]]
        dist[gw] = 0.0
        for _ in range(n):
            new = np.minimum(dist, (w + dist[None, :]).min(axis=1))
            new[gw] = 0.0
            if np.array_equal(new, dist):
                break
            dist = new
        next_hop = np.argmin(w + dist[None, :], axis=1)
        next_hop[~np.isfinite(dist)] = -1
        next_hop[list(obs.gateways)] = -1
        return next_hop
