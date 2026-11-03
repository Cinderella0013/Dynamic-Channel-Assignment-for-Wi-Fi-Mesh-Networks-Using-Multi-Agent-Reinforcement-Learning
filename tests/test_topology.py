import numpy as np
from mesh.sim.topology import generate_topology, link_adjacency


def _connected(adj):
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(adj)


def test_rgg_connected_and_deterministic():
    a = generate_topology("rgg", 20, 2, np.random.default_rng(0))
    b = generate_topology("rgg", 20, 2, np.random.default_rng(0))
    assert a.pos.shape == (20, 2) and len(set(a.gateways)) == 2
    assert np.array_equal(a.pos, b.pos) and a.gateways == b.gateways
    assert _connected(link_adjacency(a.pos))


def test_grid_spacing_links_neighbors_only():
    t = generate_topology("grid", 9, 1, np.random.default_rng(0))
    adj = link_adjacency(t.pos)
    assert adj[0, 1] and adj[0, 3] and not adj[0, 2]   # 35 m linked, 70 m not
    assert not adj.diagonal().any() and (adj == adj.T).all()
