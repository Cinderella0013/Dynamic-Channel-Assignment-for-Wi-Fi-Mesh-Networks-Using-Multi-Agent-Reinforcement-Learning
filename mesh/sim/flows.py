"""Hop counts, flow paths from next_hop, fluid throughput allocation."""
from collections import deque

import numpy as np


def hops_to_gateway(adj, gateways, alive):
    hops = np.full(len(adj), -1, dtype=int)
    q = deque()
    for g in gateways:
        if alive[g]:
            hops[g] = 0
            q.append(g)
    while q:
        u = q.popleft()
        for v in np.flatnonzero(adj[u] & alive):
            if hops[v] < 0:
                hops[v] = hops[u] + 1
                q.append(v)
    return hops


def min_hop_next_hop(adj, gateways, alive):
    """Parent with one hop less toward a gateway (lowest id on ties); -1 if none."""
    hops = hops_to_gateway(adj, gateways, alive)
    nh = np.full(len(adj), -1, dtype=int)
    for u in np.flatnonzero(hops > 0):
        nh[u] = np.flatnonzero(adj[u] & alive & (hops == hops[u] - 1))[0]
    return nh


def paths_from_next_hop(next_hop, gateways, usable, alive):
    n = len(next_hop)
    gw = set(gateways)
    paths = []
    for s in range(n):
        if s in gw or not alive[s]:
            paths.append(None)
            continue
        path, u = [s], s
        while u not in gw and len(path) <= n:
            v = int(next_hop[u])
            if not (0 <= v < n) or not alive[v] or not usable[u, v]:
                break
            path.append(v)
            u = v
        paths.append(path if u in gw else None)
    return paths


def allocate(paths, demand, cap):
    # ponytail: proportional share, not max-min fair. Good enough for a fluid model;
    # switch to progressive filling if fairness results look like allocation artifacts.
    load = np.zeros_like(cap, dtype=float)
    for s, p in enumerate(paths):
        if p is not None:
            for u, v in zip(p, p[1:]):
                load[u, v] += demand[s]
    thr = np.zeros(len(paths))
    for s, p in enumerate(paths):
        if p is None or demand[s] <= 0:
            continue
        x = demand[s]
        for u, v in zip(p, p[1:]):
            if load[u, v] > cap[u, v]:
                x = min(x, cap[u, v] * demand[s] / load[u, v])
        thr[s] = x
    return thr, load


def domain_allocate(rg, cc, rate, paths, demand, mac_eff, contention=0.0):
    """Work-conserving airtime model: each loaded link's interference domain (itself + co-channel loaded links in
    carrier-sense range or sharing a node) may use at most eta = mac_eff / (1 + contention * (n_tx - 1)) of the
    airtime; an over-subscribed domain scales the flows through its links. Idle neighbours cost nothing, unlike
    the static 1/k share of link_capacity. Returns (thr, load, cap) with cap = load * eta / D, so load / cap is
    the domain's airtime utilisation."""
    load = np.zeros_like(rate, dtype=float)
    for s, p in enumerate(paths):
        if p is not None and demand[s] > 0:
            for u, v in zip(p, p[1:]):
                load[u, v] += demand[s]
    us, vs = np.nonzero(load > 0)
    air = load[us, vs] / np.maximum(rate[us, vs], 1e-9)
    near = rg.conflict | np.eye(len(rate), dtype=bool)
    m = len(us)
    nb = np.zeros((m, m), bool)
    for i in range(m):
        for j in range(m):
            nb[i, j] = cc[us[j], vs[j]] == cc[us[i], vs[i]] and near[np.ix_((us[i], vs[i]), (us[j], vs[j]))].any()
    dom = nb @ air
    eta = np.array([mac_eff / (1.0 + contention * (len(set(us[nb[i]])) - 1)) for i in range(m)])
    f = np.minimum(1.0, eta / np.maximum(dom, 1e-12))
    g = np.array([f[nb[i]].min() for i in range(m)]) if m else f  # a link is limited by every domain it sits in
    gl = np.ones_like(rate)
    gl[us, vs] = g
    cap = rate * mac_eff
    cap[us, vs] = load[us, vs] * eta / np.maximum(dom, 1e-12)
    thr = np.zeros(len(paths))
    for s, p in enumerate(paths):
        if p is not None and demand[s] > 0:
            thr[s] = demand[s] * min(gl[u, v] for u, v in zip(p, p[1:]))
    return thr, load, cap
