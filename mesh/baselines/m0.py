"""M0: no optimisation. Fixed channels [0, 1] everywhere, min-hop routing."""
import numpy as np

from mesh.sim.env import NODE_FEATS, Action, edge_matrix
from mesh.sim.flows import min_hop_next_hop


class M0:
    def act(self, obs):
        n = len(obs.node_feat)
        alive = obs.node_feat[:, NODE_FEATS.index("alive")] > 0
        adj = edge_matrix(obs, "is_link") > 0
        return Action(np.tile([0, 1], (n, 1)), min_hop_next_hop(adj, obs.gateways, alive))
