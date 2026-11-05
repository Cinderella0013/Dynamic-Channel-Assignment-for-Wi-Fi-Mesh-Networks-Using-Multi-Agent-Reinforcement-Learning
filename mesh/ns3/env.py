"""MeshEnv with packet transport delegated to ns-3 (docs/ns3/protocol.md section 4).

Python keeps topology, traffic, failure and hotspot (MeshEnv.reset, same seeds); every step the action
and demand go to the transport and the measured delivery, queues and drops replace the fluid model.
"""
import numpy as np

from mesh.sim.env import EDGE_FEATS, MeshEnv, RewardWeights
from mesh.sim.flows import allocate, paths_from_next_hop
from mesh.sim.graph import link_capacity, link_rate
from mesh.sim.metrics import jain, p5, utility
from mesh.sim.radio import CHANNELS

STEP_S = 0.1
PKT_BYTES = 1200


class Ns3MeshEnv(MeshEnv):
    def __init__(self, cfg, seed, transport, weights=RewardWeights(), switch_delay_s=None, route_hold_s=None,
                 mac=None):
        super().__init__(cfg, seed, weights)
        self.transport = transport
        self.mac = mac or {}  # extra init keys for the ns-3 MAC, e.g. {"rts": 1} (protocol section 3)
        self.switch_delay_s = cfg.downtime_steps * STEP_S if switch_delay_s is None else switch_delay_s
        self.route_hold_s = cfg.downtime_steps * STEP_S if route_hold_s is None else route_hold_s
        self.to_mbps = 8 / STEP_S / 1e6  # bytes in one step -> Mbps-step (the sim's backlog unit)

    def reset(self, seed=None):
        cfg, ei = self.cfg, self.cfg.ext_interf
        self.ack = np.ones((cfg.n, cfg.n))
        obs = super().reset(seed)
        self.transport.open({
            "type": "init", "n": cfg.n, "pos": self.top.pos.tolist(), "gateways": list(self.gw),
            "channels": list(CHANNELS), "step_s": STEP_S, "pkt_bytes": PKT_BYTES,
            "queue_bytes": 5 * cfg.mean_mbps / self.to_mbps, "switch_delay_s": self.switch_delay_s,
            "route_hold_s": self.route_hold_s,
            "ext_interf": None if ei is None else dict(channel_idx=ei[0], start=ei[1], end=ei[2], noise_dbm=ei[3]),
            "seed": int(self.seed), **({"queue_preserve": 1} if cfg.queue_preserve else {}), **self.mac})
        return obs

    def close(self):
        self.transport.close()

    def step(self, action):
        if self.t >= self.cfg.episode_len:
            raise RuntimeError("episode finished; call reset()")
        channels, next_hop = self._validate(action)
        cfg, t, n = self.cfg, self.t, self.cfg.n
        if cfg.failure_step == t:  # same victim as MeshEnv; ns-3 counts its queue as switch drops
            deg = np.where(self.is_gw, -1, self.rg.link.sum(axis=1))
            dead = int(np.argmax(deg))
            self.alive[dead] = False
            self.backlog[dead] = 0.0
        noise = self._ext_noise(t)
        if cfg.hotspot_period and t % cfg.hotspot_period == 0:
            k = (t // cfg.hotspot_period) % len(self.gw)
            self.traffic.set_boost(np.flatnonzero((self.nearest_gw == k) & ~self.is_gw))
        demand = self.traffic.demand(t) * self.alive * ~self.is_gw
        offered = self.backlog + demand
        src = self.alive & ~self.is_gw

        res = self.transport.step({"type": "step", "t": t, "channels": channels.tolist(),
                                   "next_hop": next_hop.tolist(), "demand_mbps": demand.tolist(),
                                   "alive": self.alive.astype(int).tolist()})
        thr = np.asarray(res["delivered_mbps"], float)
        new_backlog = np.asarray(res["src_queue_bytes"], float) * self.to_mbps
        lost = float(sum(np.sum(v) for v in res["drop_bytes"].values())) * self.to_mbps
        ack = np.ones((n, n))
        for u, v, ok, fail, _snr, _rate in res["links"]:
            if ok + fail > 0:
                ack[int(u), int(v)] = ok / (ok + fail)

        # routing counts and the fluid load for load_frac, exactly as MeshEnv (downtime bookkeeping included)
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
        flows = src & (offered > 0)
        active = np.zeros_like(self.rg.link)
        for s in np.flatnonzero(flows):
            if paths[s] is not None:
                for u, v in zip(paths[s], paths[s][1:]):
                    active[u, v] = True
        cap = link_capacity(self.rg, cc, rate_now, active, cfg.mac_eff, cfg.contention)
        _, load = allocate(paths, offered, cap)
        self.chan_down = np.maximum(self.chan_down - 1, 0)
        self.route_down = np.maximum(self.route_down - 1, 0)

        # 1 s window fairness / utility: copied from MeshEnv.step
        active = offered > 0
        self.thr_hist = (self.thr_hist + [(thr * active, active, demand > 0)])[-cfg.utility_window:]
        sent_w = np.sum([h[0] for h in self.thr_hist], axis=0)
        cnt_w = np.sum([h[1] for h in self.thr_hist], axis=0)
        thr_w = np.divide(sent_w, cnt_w, out=np.zeros(n), where=cnt_w > 0)
        mflows = src & np.any([h[2] for h in self.thr_hist], axis=0)
        flow_u = np.log(thr_w + self.w.eps)
        local_u = np.zeros(n)
        for s in np.flatnonzero(mflows):
            local_u[planned[s] if planned[s] is not None else s] += flow_u[s]
        n_sw = int(((channels != self.channels) & self.alive[:, None]).sum())
        n_rc = int(((next_hop != self.next_hop) & src).sum())
        n_drop = int(sum(1 for s in np.flatnonzero(flows) if planned[s] is None))
        n_int = int(sum(1 for s in np.flatnonzero(flows) if planned[s] is not None and paths[s] is None))
        u = utility(thr_w[mflows], self.w.eps)
        reward = u - self.w.gamma * n_sw - self.w.delta * n_rc - self.w.eta * n_drop

        self.channels, self.next_hop = channels, next_hop
        self.demand, self.offered, self.thr, self.load, self.cap = demand, offered, thr, load, cap
        self.backlog = new_backlog
        self.ack = ack
        self.t += 1
        info = dict(throughput=thr, aggregate=float(thr[flows].sum()), jain=jain(thr_w[mflows]),
                    p5=p5(thr_w[mflows]), utility=u, n_channel_switch=n_sw, n_route_change=n_rc,
                    n_drop=n_drop, local_utility=local_u, done=self.t == cfg.episode_len,
                    lost=lost, demand_total=float(demand.sum()),
                    backlog_total=float(self.backlog.sum()), n_interrupted=n_int, n_flows=int(mflows.sum()))
        if info["done"]:
            self.close()
        return self._observe(noise), float(reward), info

    def _observe(self, noise):
        obs = super()._observe(noise)
        obs.edge_attr[:, EDGE_FEATS.index("ack_ratio")] = self.ack[obs.edge_index[0], obs.edge_index[1]]
        return obs
