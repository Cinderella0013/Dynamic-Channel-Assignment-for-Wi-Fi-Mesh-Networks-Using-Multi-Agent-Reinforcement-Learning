"""Mesh node placement: random geometric graph and grid."""
from dataclasses import dataclass

import numpy as np

from mesh.sim.radio import LINK_THRESHOLD_DBM, rx_power_dbm


@dataclass(frozen=True)
class Topology:
    pos: np.ndarray  # (n, 2) metres
    gateways: tuple[int, ...]


def distances(pos):
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)


def link_adjacency(pos):
    adj = rx_power_dbm(distances(pos)) >= LINK_THRESHOLD_DBM
    np.fill_diagonal(adj, False)
    return adj


def _connected(adj):
    seen = np.zeros(len(adj), bool)
    seen[0] = True
    frontier = seen.copy()
    while frontier.any():
        frontier = adj[frontier].any(axis=0) & ~seen
        seen |= frontier
    return seen.all()


def generate_topology(kind, n, n_gateways, rng):
    if kind == "rgg":
        side = 30 * np.sqrt(n)
        for _ in range(1000):
            pos = rng.uniform(0, side, size=(n, 2))
            if _connected(link_adjacency(pos)):
                break
        else:
            raise RuntimeError(f"no connected rgg topology for n={n} after 1000 tries")
    elif kind == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        idx = np.arange(n)
        pos = np.stack([idx % cols, idx // cols], axis=1) * 35.0
    else:
        raise ValueError(f"unknown topology kind: {kind}")
    gateways = tuple(int(g) for g in rng.choice(n, size=n_gateways, replace=False))
    return Topology(pos=pos, gateways=gateways)
