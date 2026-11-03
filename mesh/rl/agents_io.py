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
F_O = 5 + N_CHANNELS + K_SLOTS + 1
F_M = 4 * K_SLOTS


def alive_mask(obs):
    return obs.node_feat[:, NODE_FEATS.index("alive")] > 0


def neighbor_slots(obs):
    """Link neighbours by SNR descending (ties: lower id), -1 padded. Static per topology."""
    link = edge_matrix(obs, "is_link") > 0
    snr = edge_matrix(obs, "snr_db")
    slots = np.full((len(link), K_SLOTS), -1, dtype=int)
    for u in range(len(link)):
        nbrs = np.flatnonzero(link[u])
        nbrs = nbrs[np.lexsort((nbrs, -snr[u, nbrs]))][:K_SLOTS]
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
    cols = [NODE_FEATS.index(k) for k in ("offered_load", "unmet", "is_gateway", "hop_to_gw", "alive")]
    base = nf[:, cols].astype(np.float64)
    base[:, 3] /= n
    ch = np.eye(N_CHANNELS)[obs.channels[:, 1]]
    match = slots == obs.next_hop[:, None]
    cur = np.where(match.any(axis=1), match.argmax(axis=1), K_SLOTS)
    par = np.eye(K_SLOTS + 1)[cur]
    o = np.concatenate([base, ch, par], axis=1)

    valid = (slots >= 0) & alive[np.maximum(slots, 0)]
    u = np.repeat(np.arange(n), K_SLOTS).reshape(n, K_SLOTS)
    v = np.maximum(slots, 0)
    per_slot = [valid.astype(float),
                edge_matrix(obs, "snr_db")[u, v] / 40,
                edge_matrix(obs, "rate_mbps")[u, v] / 143.4,
                edge_matrix(obs, "load_frac")[u, v]]
    m = np.stack(per_slot, axis=2) * valid[:, :, None]
    return o, m.reshape(n, F_M)


def per_agent_reward(prev_obs, action, info, reward_global, w):
    alive = alive_mask(prev_obs)
    sw = (action.channels != prev_obs.channels).sum(axis=1)
    rc = (action.next_hop != prev_obs.next_hop).astype(float)
    r = (0.5 * info["local_utility"] + 0.5 * reward_global / max(alive.sum(), 1)
         - w.gamma * sw - w.delta * rc)
    return np.where(alive, r, 0.0)
