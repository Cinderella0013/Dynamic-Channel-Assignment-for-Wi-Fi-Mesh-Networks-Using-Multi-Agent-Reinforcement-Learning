"""EGAT v2 (spec addendum 4, the user's design): GAT-Q scores -> RAES ordered channel reservation
-> migration refinement -> execute with queue preservation -> ACK feeds the next round."""
import math

import numpy as np

from mesh.rl.agents_io import N_DL_CH, alive_mask
from mesh.sim.env import NODE_FEATS, edge_matrix


def _z(x):
    """z-score over the last axis; a constant row becomes 0 (no preference)."""
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=-1, keepdims=True)
    return np.divide(x - x.mean(axis=-1, keepdims=True), sd, out=np.zeros_like(x), where=sd > 0)


def _depth(next_hop):
    n = len(next_hop)
    depth = np.zeros(n, dtype=int)
    for u in range(n):
        v, d = u, 0
        while next_hop[v] >= 0 and d <= n:
            v, d = next_hop[v], d + 1
        depth[u] = d
    return depth


def _current_slot(obs, slots, fallback):
    hit = (slots == obs.next_hop[:, None]) & (slots >= 0)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), fallback)


def raes(obs, slots, q_ch, q_rt, route_mask, a=1.0, b=1.0, g=1.0, order="backlog", downtime=1):
    """Ordered channel reservation: each node, in turn, reserves
    argmax_c z(Q) - a z(interference from reserved neighbours) - b z(co-channel load) - g z(move cost)."""
    n = len(q_ch)
    alive = alive_mask(obs)
    nf = obs.node_feat
    backlog = nf[:, NODE_FEATS.index("backlog")]
    load = nf[:, NODE_FEATS.index("offered_load")]
    hop = nf[:, NODE_FEATS.index("hop_to_gw")]
    conflict = (edge_matrix(obs, "is_conflict") > 0) & alive[:, None] & alive[None, :]
    cur = np.clip(obs.channels[:, 1] - 1, 0, N_DL_CH - 1)
    if order == "backlog":  # congested nodes reserve first
        seq = sorted(range(n), key=lambda u: (-backlog[u], hop[u], u))
    else:  # "parents": tree order of the current routes
        depth = _depth(obs.next_hop)
        seq = sorted(range(n), key=lambda u: (depth[u], u))
    c = np.full(n, -1, dtype=int)
    score = np.zeros((n, N_DL_CH))
    for u in seq:
        nb = np.flatnonzero(conflict[u])
        interf = np.zeros(N_DL_CH)
        for v in nb[c[nb] >= 0]:
            interf[c[v]] += 1.0 + backlog[v]
        cochan = np.zeros(N_DL_CH)
        np.add.at(cochan, cur[nb], load[nb])
        move = (np.arange(N_DL_CH) != cur[u]) * downtime * backlog[u]
        score[u] = _z(q_ch[u]) - a * _z(interf) - b * _z(cochan) - g * _z(move)
        c[u] = int(np.argmax(score[u])) if alive[u] else cur[u]

    cur_slot = _current_slot(obs, slots, fallback=np.full(n, -1))
    rt = np.where(route_mask, 0.0, -np.inf)
    for u in range(n):
        ok = route_mask[u]
        move = (np.arange(slots.shape[1]) != cur_slot[u]) * downtime * backlog[u]
        rt[u, ok] = _z(q_rt[u, ok]) - g * _z(move[ok]) if ok.sum() > 1 else 0.0
    j = rt.argmax(axis=1)
    return c, j, score


def refine(obs, slots, c, j, ch_score, q_rt, cooldown_left, margin=0.1, k_frac=0.1):
    """Migration refinement: keep a proposed move only if its gain beats the margin, the node is not
    cooling down, no conflicting neighbour moves this round, and at most ceil(k_frac * n_alive) move."""
    n = len(c)
    alive = alive_mask(obs)
    conflict = (edge_matrix(obs, "is_conflict") > 0) & alive[:, None] & alive[None, :]
    c_cur = np.clip(obs.channels[:, 1] - 1, 0, N_DL_CH - 1)
    j_cur = _current_slot(obs, slots, fallback=j)
    zr = _z(q_rt)
    idx = np.arange(n)
    gain = (ch_score[idx, c] - ch_score[idx, c_cur]) + (zr[idx, j] - zr[idx, j_cur])
    proposes = (c != c_cur) | (j != j_cur)
    cand = np.flatnonzero(proposes & alive & (gain > margin) & (np.asarray(cooldown_left) == 0))
    k_max = math.ceil(k_frac * alive.sum())
    moved = np.zeros(n, dtype=bool)
    for u in cand[np.argsort(-gain[cand], kind="stable")]:
        if moved.sum() >= k_max:
            break
        if not (conflict[u] & moved).any():
            moved[u] = True
    return np.where(moved, c, c_cur), np.where(moved, j, j_cur), moved
