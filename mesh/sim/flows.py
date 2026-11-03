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
