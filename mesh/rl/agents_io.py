"""Per-agent interface shared by every learner (agent = mesh node).

Action a = c * K_SLOTS + j: downlink channel index c + 1 on radio 1, parent =
link neighbour in slot j. Radio 0 follows the parent's downlink (the tree
convention Ko uses), so every parent link has a shared channel.
"""
import numpy as np

from mesh.sim.env import NODE_FEATS, Action, edge_matrix
from mesh.sim.radio import N_CHANNELS

K_SLOTS = 4
N_DL_CH = 6
N_ACTIONS = N_DL_CH * K_SLOTS
F_O = 6 + N_CHANNELS + K_SLOTS + 1
F_M = 5 * K_SLOTS


def alive_mask(obs):
    return obs.node_feat[:, NODE_FEATS.index("alive")] > 0


def neighbor_slots(obs):
    """Parent candidates: link neighbours by hop-to-gateway ascending, then SNR descending
    (ties: lower id), -1 padded. Hop first so every routable node has a slot that makes
    progress toward a gateway; pure SNR order picks the nearest nodes, which for ~22% of
    nodes are all no closer to a gateway (Ko through the interface fell from +2714 to -1592).
    """
    link = edge_matrix(obs, "is_link") > 0
    snr = edge_matrix(obs, "snr_db")
    hop = obs.node_feat[:, NODE_FEATS.index("hop_to_gw")]
    slots = np.full((len(link), K_SLOTS), -1, dtype=int)
    for u in range(len(link)):
        nbrs = np.flatnonzero(link[u])
        nbrs = nbrs[np.lexsort((nbrs, -snr[u, nbrs], hop[nbrs]))][:K_SLOTS]
        slots[u, : len(nbrs)] = nbrs
    return slots


def _valid_slots(obs, slots):
    alive = alive_mask(obs)
    valid = (slots >= 0) & alive[np.maximum(slots, 0)]
    valid[list(obs.gateways)] = False
    return valid


def action_mask(obs, slots):
    valid = _valid_slots(obs, slots)
    none = ~valid.any(axis=1)
    valid[none, 0] = True  # no usable parent: slot-0 actions only, decoded to next_hop = -1
    mask = np.tile(valid, (1, N_DL_CH))  # index c * K + j
    mask[~alive_mask(obs)] = True
    return mask


def decode(obs, slots, a):
    a = np.asarray(a, dtype=int)
    n = len(a)
    c, j = a // K_SLOTS, a % K_SLOTS
    valid = _valid_slots(obs, slots)
    parent = np.where(valid[np.arange(n), j], slots[np.arange(n), j], -1)
    channels = np.zeros((n, 2), dtype=int)
    channels[:, 1] = c + 1
    # radio 1 comes straight from each node's own action, so no ordering is needed
    channels[:, 0] = np.where(parent >= 0, channels[np.maximum(parent, 0), 1], 0)
    return Action(channels, parent)


def agent_features(obs, slots):
    nf = obs.node_feat
    n = len(nf)
    alive = alive_mask(obs)
    cols = [NODE_FEATS.index(k) for k in ("offered_load", "unmet", "is_gateway", "hop_to_gw", "alive", "backlog")]
    base = nf[:, cols].astype(np.float64)
    base[:, 3] /= n
    ch = np.eye(N_CHANNELS)[obs.channels[:, 1]]
    match = (slots == obs.next_hop[:, None]) & (slots >= 0)  # -1 padding is not a parent
    cur = np.where(match.any(axis=1), match.argmax(axis=1), K_SLOTS)
    par = np.eye(K_SLOTS + 1)[cur]
    o = np.concatenate([base, ch, par], axis=1)

    valid = (slots >= 0) & alive[np.maximum(slots, 0)]
    u = np.repeat(np.arange(n), K_SLOTS).reshape(n, K_SLOTS)
    v = np.maximum(slots, 0)
    per_slot = [valid.astype(float),
                edge_matrix(obs, "snr_db")[u, v] / 40,
                edge_matrix(obs, "rate_mbps")[u, v] / 143.4,
                edge_matrix(obs, "load_frac")[u, v],
                edge_matrix(obs, "ack_ratio")[u, v]]
    m = np.stack(per_slot, axis=2) * valid[:, :, None]
    return o, m.reshape(n, F_M)


def per_agent_reward(prev_obs, action, info, reward_global, w, lam=0.5):
    alive = alive_mask(prev_obs)
    # radio 1 is the node's own choice; a radio-0 change caused by the parent retuning
    # (same parent, parent's downlink changed) is charged to the parent, not the child
    sw = (action.channels[:, 1] != prev_obs.channels[:, 1]).astype(float)
    r0 = action.channels[:, 0] != prev_obs.channels[:, 0]
    same_parent = (action.next_hop == prev_obs.next_hop) & (action.next_hop >= 0)
    follows = r0 & same_parent & (sw[np.maximum(action.next_hop, 0)] > 0)
    sw += r0 & ~follows
    np.add.at(sw, action.next_hop[follows], 1.0)
    rc = (action.next_hop != prev_obs.next_hop).astype(float)
    r = (lam * info["local_utility"] + (1 - lam) * reward_global / max(alive.sum(), 1)
         - w.gamma * sw - w.delta * rc)
    return np.where(alive, r, 0.0)


def sequential_channels(obs, slots, scores, j):
    """Greedy (evaluation) channel choice shared by every learned method.

    Channel labels are interchangeable, so per-node argmax gives every node the same channel
    whenever nodes see the same state. Nodes decide in tree order (parents first, like Ko's round);
    each takes its highest-scoring channel not already taken by a decided conflict neighbour
    (or the least-used one if all are taken). Needs one sequential coordination round, as Ko does.
    """
    n = len(j)
    alive = alive_mask(obs)
    valid = _valid_slots(obs, slots)
    parent = np.where(valid[np.arange(n), j], slots[np.arange(n), j], -1)
    depth = np.zeros(n, dtype=int)
    for u in range(n):
        v, d = u, 0
        while parent[v] >= 0 and d <= n:
            v, d = parent[v], d + 1
        depth[u] = d
    conflict = (edge_matrix(obs, "is_conflict") > 0) & alive[:, None] & alive[None, :]
    c = np.full(n, -1, dtype=int)
    for u in sorted(range(n), key=lambda u: (depth[u], u)):
        used = np.bincount(c[conflict[u] & (c >= 0)], minlength=N_DL_CH)
        c[u] = int(np.argmax(np.asarray(scores[u], dtype=float) - 1e3 * used))
    return c


def hold_bonus(obs, slots, margin):
    """(n, N_ACTIONS) switch-hysteresis bonus for joint (channel, slot) scores at greedy decoding:
    +margin on actions keeping the node's current radio-1 channel, +margin on those keeping its parent."""
    n = len(slots)
    c = np.arange(N_DL_CH * K_SLOTS) // K_SLOTS
    j = np.arange(N_DL_CH * K_SLOTS) % K_SLOTS
    keep_c = c[None, :] == (obs.channels[:, 1] - 1)[:, None]
    keep_j = (slots[:, j] == obs.next_hop[:, None]) & (obs.next_hop[:, None] >= 0)
    return margin * (keep_c.astype(float) + keep_j)


def greedy_with_sequential_channels(obs, slots, joint):
    """joint: (n, N_ACTIONS) scores with invalid actions at -inf. Returns action indices."""
    n = len(joint)
    a = joint.argmax(axis=1)
    j = a % K_SLOTS
    scores = joint.reshape(n, N_DL_CH, K_SLOTS)[np.arange(n), :, j]
    return sequential_channels(obs, slots, scores, j) * K_SLOTS + j
