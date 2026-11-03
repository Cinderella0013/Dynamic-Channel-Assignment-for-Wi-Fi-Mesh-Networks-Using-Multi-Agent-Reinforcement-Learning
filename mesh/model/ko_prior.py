"""Ko's costs and actions expressed in the shared RL action space (EGAT-MAPPO residual prior, BC, fallback)."""
import numpy as np

from mesh.baselines.ko import ko_route_via
from mesh.rl.agents_io import K_SLOTS, N_DL_CH, _valid_slots, alive_mask
from mesh.sim.env import NODE_FEATS, edge_matrix


def _zscore_rows(x):
    """Per-row z-score over finite entries; +inf stays +inf, constant rows become 0."""
    out = np.full_like(x, np.inf, dtype=float)
    for i, row in enumerate(x):
        ok = np.isfinite(row)
        if ok.any():
            v = row[ok]
            out[i, ok] = (v - v.mean()) / (v.std() + 1e-6)
    return out


def ko_slot_costs(obs, slots):
    alive = alive_mask(obs)
    via = ko_route_via(obs, alive)
    n = len(slots)
    cost = via[np.arange(n)[:, None], np.maximum(slots, 0)]
    cost = np.where(_valid_slots(obs, slots), cost, np.inf)
    return _zscore_rows(cost)


def ko_channel_costs(obs):
    alive = alive_mask(obs)
    load = obs.node_feat[:, NODE_FEATS.index("offered_load")]
    conflict = edge_matrix(obs, "is_conflict") > 0
    n = len(alive)
    cost = np.zeros((n, N_DL_CH))
    for u in range(n):
        others = np.flatnonzero(conflict[u] & alive & (obs.next_hop != u))  # Ko ignores its children
        for c in range(N_DL_CH):
            cost[u, c] = ((obs.channels[others] == c + 1).any(axis=1) * (1 + load[others])).sum()
    return _zscore_rows(cost)


def ko_action_idx(obs, slots, ko):
    raw = ko.act(obs)
    valid = _valid_slots(obs, slots)
    c = np.clip(raw.channels[:, 1] - 1, 0, N_DL_CH - 1)
    j = np.zeros(len(slots), dtype=int)
    for u in range(len(slots)):
        hit = np.flatnonzero((slots[u] == raw.next_hop[u]) & (slots[u] >= 0))
        if len(hit):
            j[u] = hit[0]
        elif valid[u].any():
            j[u] = int(np.argmax(valid[u]))
    return c * K_SLOTS + j
