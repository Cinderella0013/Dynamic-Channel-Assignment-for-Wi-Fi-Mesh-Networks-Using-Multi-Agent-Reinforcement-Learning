"""Multi-agent Wi-Fi mesh env: joint channel assignment + routing, graph observation."""
from dataclasses import dataclass

import numpy as np

from mesh.sim.flows import allocate, hops_to_gateway, min_hop_next_hop, paths_from_next_hop
from mesh.sim.graph import build_radio_graph, common_channel, link_capacity, link_rate
from mesh.sim.metrics import jain, p5, utility
from mesh.sim.radio import N_CHANNELS, snr_db
from mesh.sim.topology import generate_topology
from mesh.sim.traffic import Traffic

NODE_FEATS = ("offered_load", "unmet", "is_gateway", "hop_to_gw", "alive")
EDGE_FEATS = ("is_link", "is_conflict", "snr_db", "rate_mbps", "load_frac", "same_channel")


@dataclass
class Action:
    channels: np.ndarray  # (n, 2) int in [0, N_CHANNELS)
    next_hop: np.ndarray  # (n,) int, -1 for gateways


@dataclass
class Observation:
    node_feat: np.ndarray   # (n, len(NODE_FEATS)) float32
    channels: np.ndarray    # (n, 2) currently applied
    next_hop: np.ndarray    # (n,) currently applied
    edge_index: np.ndarray  # (2, E) int64, link | conflict pairs, both directions
    edge_attr: np.ndarray   # (E, len(EDGE_FEATS)) float32
    gateways: tuple[int, ...]
    t: int


@dataclass(frozen=True)
class RewardWeights:
    eps: float = 0.01
    gamma: float = 0.1
    delta: float = 0.05
    eta: float = 0.1


class MeshEnv:
    def __init__(self, cfg, seed, weights=RewardWeights()):
        self.cfg = cfg
        self.seed = seed
        self.w = weights

    def reset(self):
        cfg = self.cfg
        top_ss, traffic_ss, _ = np.random.SeedSequence(self.seed).spawn(3)
        self.top = generate_topology(cfg.kind, cfg.n, cfg.n_gateways, np.random.default_rng(top_ss))
        self.rg = build_radio_graph(self.top)
        self.gw = self.top.gateways
        self.is_gw = np.zeros(cfg.n, bool)
        self.is_gw[list(self.gw)] = True
        self.alive = np.ones(cfg.n, bool)
        boost = None
        if cfg.hotspot:
            per_gw = np.stack([hops_to_gateway(self.rg.link, (g,), self.alive) for g in self.gw])
            per_gw = np.where(per_gw < 0, np.iinfo(int).max, per_gw)
            boost = np.flatnonzero((per_gw.argmin(axis=0) == 0) & ~self.is_gw)
        self.traffic = Traffic(cfg.n, np.random.default_rng(traffic_ss), cfg.mean_mbps,
                               boost_nodes=boost, shift_step=cfg.shift_step)
        self.t = 0
        self.channels = np.tile([0, 1], (cfg.n, 1))
        self.next_hop = min_hop_next_hop(self.rg.link, self.gw, self.alive)
        self.demand = np.zeros(cfg.n)
        self.thr = np.zeros(cfg.n)
        self.load = np.zeros((cfg.n, cfg.n))
        self.cap = np.zeros((cfg.n, cfg.n))
        return self._observe(self._ext_noise(0))

    def _ext_noise(self, t):
        ei = self.cfg.ext_interf
        if ei is None or not (ei[1] <= t < ei[2]):
            return None
        noise = np.full(N_CHANNELS, -np.inf)
        noise[ei[0]] = ei[3]
        return noise

    def _cc(self, channels):
        cc = common_channel(channels)
        cc[~self.alive, :] = -1
        cc[:, ~self.alive] = -1
        return cc

    def _validate(self, action):
        n = self.cfg.n
        ch = np.asarray(action.channels)
        nh = np.asarray(action.next_hop)
        if ch.shape != (n, 2) or nh.shape != (n,):
            raise ValueError(f"action shapes {ch.shape}, {nh.shape}; expected ({n}, 2), ({n},)")
        if not (np.issubdtype(ch.dtype, np.integer) and np.issubdtype(nh.dtype, np.integer)):
            raise ValueError("action arrays must be integer")
        if ch.min() < 0 or ch.max() >= N_CHANNELS:
            raise ValueError(f"channel index outside [0, {N_CHANNELS})")
        return ch.copy(), nh.copy()

    def step(self, action):
        channels, next_hop = self._validate(action)
        cfg, t = self.cfg, self.t
        if cfg.failure_step == t:
            deg = np.where(self.is_gw, -1, self.rg.link.sum(axis=1))
            self.alive[int(np.argmax(deg))] = False
        noise = self._ext_noise(t)
        demand = self.traffic.demand(t) * self.alive * ~self.is_gw

        cc = self._cc(channels)
        rate = link_rate(self.rg, cc, noise)
        paths = paths_from_next_hop(next_hop, self.gw, rate > 0, self.alive)
        active = np.zeros_like(self.rg.link)
        for p in paths:
            if p is not None:
                for u, v in zip(p, p[1:]):
                    active[u, v] = True
        cap = link_capacity(self.rg, cc, rate, active)
        thr, load = allocate(paths, demand, cap)

        src = self.alive & ~self.is_gw
        flow_u = np.log(thr + self.w.eps)
        local_u = np.zeros(cfg.n)
        for s in np.flatnonzero(src):
            if paths[s] is not None:
                local_u[paths[s]] += flow_u[s]
        n_sw = int(((channels != self.channels) & self.alive[:, None]).sum())
        n_rc = int(((next_hop != self.next_hop) & src).sum())
        n_drop = int(sum(1 for s in np.flatnonzero(src) if demand[s] > 0 and paths[s] is None))
        u = utility(thr[src], self.w.eps)
        reward = u - self.w.gamma * n_sw - self.w.delta * n_rc - self.w.eta * n_drop

        self.channels, self.next_hop = channels, next_hop
        self.demand, self.thr, self.load, self.cap = demand, thr, load, cap
        self.t += 1
        info = dict(throughput=thr, aggregate=float(thr[src].sum()), jain=jain(thr[src]),
                    p5=p5(thr[src]), utility=u, n_channel_switch=n_sw, n_route_change=n_rc,
                    n_drop=n_drop, local_utility=local_u, done=self.t == cfg.episode_len)
        return self._observe(self._ext_noise(self.t)), float(reward), info

    def _observe(self, noise):
        rg = self.rg
        hops = hops_to_gateway(rg.link, self.gw, self.alive)
        node_feat = np.stack([self.demand, self.demand - self.thr, self.is_gw,
                              hops, self.alive], axis=1).astype(np.float32)
        cc = self._cc(self.channels)
        rate = link_rate(rg, cc, noise)
        load_frac = np.divide(self.load, self.cap, out=np.zeros_like(self.load), where=self.cap > 0)
        us, vs = np.nonzero(rg.link | rg.conflict)
        snr = snr_db(rg.rx_dbm)
        edge_attr = np.stack([rg.link[us, vs], rg.conflict[us, vs], snr[us, vs], rate[us, vs],
                              load_frac[us, vs], cc[us, vs] >= 0], axis=1).astype(np.float32)
        return Observation(node_feat=node_feat, channels=self.channels.copy(),
                           next_hop=self.next_hop.copy(),
                           edge_index=np.stack([us, vs]).astype(np.int64),
                           edge_attr=edge_attr, gateways=self.gw, t=self.t)
