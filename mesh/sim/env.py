"""Multi-agent Wi-Fi mesh env: joint channel assignment + routing, graph observation."""
from dataclasses import dataclass

import numpy as np

from mesh.sim.flows import domain_allocate, allocate, hops_to_gateway, min_hop_next_hop, paths_from_next_hop
from mesh.sim.graph import build_radio_graph, common_channel, link_capacity, link_rate
from mesh.sim.metrics import jain, p5, utility
from mesh.sim.radio import N_CHANNELS, snr_db
from mesh.sim.topology import generate_topology
from mesh.sim.traffic import Traffic

NODE_FEATS = ("offered_load", "unmet", "is_gateway", "hop_to_gw", "alive", "backlog")
EDGE_FEATS = ("is_link", "is_conflict", "snr_db", "rate_mbps", "load_frac", "same_channel", "ack_ratio")


@dataclass
class Action:
    channels: np.ndarray  # (n, 2) int in [0, N_CHANNELS)
    next_hop: np.ndarray  # (n,) int, -1 for gateways
    queue_preserve: bool = False  # buffer queues across this step's switches (CSA-style)


@dataclass
class Observation:
    node_feat: np.ndarray   # (n, len(NODE_FEATS)) float32
    channels: np.ndarray    # (n, 2) currently applied
    next_hop: np.ndarray    # (n,) currently applied
    edge_index: np.ndarray  # (2, E) int64, link | conflict pairs, both directions
    edge_attr: np.ndarray   # (E, len(EDGE_FEATS)) float32
    gateways: tuple[int, ...]
    t: int


def edge_matrix(obs, name):
    """Dense (n, n) view of one edge feature; 0 where there is no edge."""
    n = len(obs.node_feat)
    m = np.zeros((n, n))
    m[obs.edge_index[0], obs.edge_index[1]] = obs.edge_attr[:, EDGE_FEATS.index(name)]
    return m


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

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        cfg = self.cfg
        top_ss, traffic_ss, _ = np.random.SeedSequence(self.seed).spawn(3)
        self.top = generate_topology(cfg.kind, cfg.n, cfg.n_gateways, np.random.default_rng(top_ss), cfg.spacing)
        self.rg = build_radio_graph(self.top)
        self.gw = self.top.gateways
        self.is_gw = np.zeros(cfg.n, bool)
        self.is_gw[list(self.gw)] = True
        self.alive = np.ones(cfg.n, bool)
        per_gw = np.stack([hops_to_gateway(self.rg.link, (g,), self.alive) for g in self.gw])
        self.nearest_gw = np.where(per_gw < 0, np.iinfo(int).max, per_gw).argmin(axis=0)
        boost = np.flatnonzero((self.nearest_gw == 0) & ~self.is_gw) if cfg.hotspot else None
        self.traffic = Traffic(cfg.n, np.random.default_rng(traffic_ss), cfg.mean_mbps,
                               boost_nodes=boost, shift_step=cfg.shift_step, periodic=cfg.periodic,
                               shift_period=cfg.shift_period, static=cfg.static)
        self.t = 0
        self.channels = np.tile([0, 1], (cfg.n, 1))
        self.next_hop = min_hop_next_hop(self.rg.link, self.gw, self.alive)
        self.demand = np.zeros(cfg.n)
        self.offered = np.zeros(cfg.n)
        self.backlog = np.zeros(cfg.n)
        self.thr_hist = []
        self.chan_down = np.zeros(cfg.n, dtype=int)
        self.route_down = np.zeros(cfg.n, dtype=int)
        self.thr = np.zeros(cfg.n)
        self.load = np.zeros((cfg.n, cfg.n))
        self.cap = np.zeros((cfg.n, cfg.n))
        self.sent = np.zeros((cfg.n, cfg.n))
        return self._observe(None)

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
        nh = np.where((nh >= 0) & (nh < n), nh, -1)
        return ch.copy(), nh

    def step(self, action):
        if self.t >= self.cfg.episode_len:
            raise RuntimeError("episode finished; call reset()")
        channels, next_hop = self._validate(action)
        cfg, t = self.cfg, self.t
        q_max = 5.0 * cfg.mean_mbps
        lost_switch = 0.0
        if cfg.failure_step == t:
            deg = np.where(self.is_gw, -1, self.rg.link.sum(axis=1))
            dead = int(np.argmax(deg))
            self.alive[dead] = False
            lost_switch += self.backlog[dead]  # counted once, at death
            self.backlog[dead] = 0.0
        noise = self._ext_noise(t)
        if cfg.hotspot_period and t % cfg.hotspot_period == 0:  # B1: hotspot moves to the next gateway
            k = (t // cfg.hotspot_period) % len(self.gw)
            self.traffic.set_boost(np.flatnonzero((self.nearest_gw == k) & ~self.is_gw))
        demand = self.traffic.demand(t) * self.alive * ~self.is_gw
        offered = self.backlog + demand
        src = self.alive & ~self.is_gw

        # switching downtime: a node that retunes radio 1 cannot use links on that channel,
        # a node that changes parent cannot use its uplink, for downtime_steps steps
        sw_ch = (channels[:, 1] != self.channels[:, 1]) & self.alive
        sw_rt = (next_hop != self.next_hop) & src
        if cfg.downtime_steps > 0:
            self.chan_down[sw_ch] = cfg.downtime_steps
            self.route_down[sw_rt] = cfg.downtime_steps
        cc = self._cc(channels)
        rate = link_rate(self.rg, cc, noise)
        intr = (cc == channels[:, 1][:, None]) & (self.chan_down > 0)[:, None] & (cc >= 0)
        for u in np.flatnonzero((self.route_down > 0) & (next_hop >= 0)):
            intr[u, next_hop[u]] = True
        intr |= intr.T
        rate_now = np.where(intr, 0.0, rate)

        planned = paths_from_next_hop(next_hop, self.gw, rate > 0, self.alive)
        paths = paths_from_next_hop(next_hop, self.gw, rate_now > 0, self.alive)
        flows = src & (offered > 0)  # a node with nothing to send is not a flow
        active = np.zeros_like(self.rg.link)
        for s in np.flatnonzero(flows):
            if paths[s] is not None:
                for u, v in zip(paths[s], paths[s][1:]):
                    active[u, v] = True
        if cfg.airtime == "domain":
            thr, load, cap = domain_allocate(self.rg, cc, rate_now, paths, offered * flows, cfg.mac_eff, cfg.contention)
        else:
            cap = link_capacity(self.rg, cc, rate_now, active, cfg.mac_eff, cfg.contention)
            thr, load = allocate(paths, offered, cap)
        sent = np.zeros_like(load)
        for s in np.flatnonzero(flows):
            if paths[s] is not None:
                for u, v in zip(paths[s], paths[s][1:]):
                    sent[u, v] += thr[s]

        # an unprepared switch drops the queue that was waiting on an interrupted hop
        queue_loss = np.zeros(cfg.n)
        if not (cfg.queue_preserve or action.queue_preserve):
            for s in np.flatnonzero(flows):
                p = planned[s]
                if p is not None and any(intr[u, v] for u, v in zip(p, p[1:])):
                    queue_loss[s] = self.backlog[s]
        lost_switch += queue_loss.sum()
        b = offered - thr - queue_loss
        lost_overflow = float(np.maximum(b - q_max, 0.0).sum())
        new_backlog = np.clip(b, 0.0, q_max)
        self.chan_down = np.maximum(self.chan_down - 1, 0)
        self.route_down = np.maximum(self.route_down - 1, 0)

        # fairness and utility on a 1 s window (utility_window steps), so a 100 ms switching outage
        # with a preserved queue is not scored as starvation. Per node, throughput is averaged over
        # the window steps in which it had something to send. The flow set is method-independent:
        # alive non-gateways with demand at some step of the window.
        active = offered > 0
        self.thr_hist = (self.thr_hist + [(thr * active, active, demand > 0)])[-cfg.utility_window:]
        sent_w = np.sum([h[0] for h in self.thr_hist], axis=0)
        cnt_w = np.sum([h[1] for h in self.thr_hist], axis=0)
        thr_w = np.divide(sent_w, cnt_w, out=np.zeros(cfg.n), where=cnt_w > 0)
        mflows = src & np.any([h[2] for h in self.thr_hist], axis=0)
        flow_u = np.log(thr_w + self.w.eps)
        local_u = np.zeros(cfg.n)
        for s in np.flatnonzero(mflows):
            # a starved flow is charged log(eps) along its planned path (at its source if unroutable)
            local_u[planned[s] if planned[s] is not None else s] += flow_u[s]
        n_sw = int(((channels != self.channels) & self.alive[:, None]).sum())
        n_rc = int(((next_hop != self.next_hop) & src).sum())
        n_drop = int(sum(1 for s in np.flatnonzero(flows) if planned[s] is None))  # routing failures
        n_int = int(sum(1 for s in np.flatnonzero(flows) if planned[s] is not None and paths[s] is None))
        u = utility(thr_w[mflows], self.w.eps)
        reward = u - self.w.gamma * n_sw - self.w.delta * n_rc - self.w.eta * n_drop

        self.channels, self.next_hop = channels, next_hop
        self.demand, self.offered, self.thr, self.load, self.cap = demand, offered, thr, load, cap
        self.backlog = new_backlog
        self.sent = sent
        self.t += 1
        info = dict(throughput=thr, aggregate=float(thr[flows].sum()), jain=jain(thr_w[mflows]),
                    p5=p5(thr_w[mflows]), utility=u, n_channel_switch=n_sw, n_route_change=n_rc,
                    n_drop=n_drop, local_utility=local_u, done=self.t == cfg.episode_len,
                    lost=float(lost_switch + lost_overflow), demand_total=float(demand.sum()),
                    backlog_total=float(self.backlog.sum()), n_interrupted=n_int, n_flows=int(mflows.sum()))
        # observation reflects the step just executed (no lookahead on the interferer)
        return self._observe(noise), float(reward), info

    def _observe(self, noise):
        rg = self.rg
        hops = hops_to_gateway(rg.link, self.gw, self.alive)
        hops = np.where(hops < 0, self.cfg.n, hops)  # unreachable = n, never closer than a gateway
        node_feat = np.stack([self.demand, self.offered - self.thr, self.is_gw,
                              hops, self.alive, self.backlog], axis=1).astype(np.float32)
        cc = self._cc(self.channels)
        rate = link_rate(rg, cc, noise)
        load_frac = np.divide(self.load, self.cap, out=np.zeros_like(self.load), where=self.cap > 0)
        ack = np.divide(self.sent, self.load, out=np.ones_like(self.load), where=self.load > 0)  # ACK: delivered / offered
        us, vs = np.nonzero(rg.link | rg.conflict)
        snr = snr_db(rg.rx_dbm)
        edge_attr = np.stack([rg.link[us, vs], rg.conflict[us, vs], snr[us, vs], rate[us, vs],
                              load_frac[us, vs], cc[us, vs] >= 0, ack[us, vs]], axis=1).astype(np.float32)
        return Observation(node_feat=node_feat, channels=self.channels.copy(),
                           next_hop=self.next_hop.copy(),
                           edge_index=np.stack([us, vs]).astype(np.int64),
                           edge_attr=edge_attr, gateways=self.gw, t=self.t)
