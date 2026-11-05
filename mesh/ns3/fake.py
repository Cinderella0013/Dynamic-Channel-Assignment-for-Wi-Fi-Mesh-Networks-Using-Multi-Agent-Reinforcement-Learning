"""Stand-in for the ns-3 binary: answers protocol messages with the fluid model of mesh.sim.env.

Lets Ns3MeshEnv be tested without ns-3; with it an episode matches MeshEnv on the same seed.
"""
from dataclasses import replace

import numpy as np

from mesh.sim.env import Action, MeshEnv
from mesh.sim.flows import min_hop_next_hop
from mesh.sim.graph import build_radio_graph
from mesh.sim.radio import snr_db
from mesh.sim.scenarios import ScenarioConfig
from mesh.sim.topology import Topology


class _Given:
    """Traffic stub: demand(t) returns the demand sent in the step message."""
    d = None

    def demand(self, t):
        return self.d


class FakeTransport:
    def __init__(self):
        self.env = None
        self.closed = True

    def open(self, m):
        self.init_msg = m
        n, step_s = m["n"], m["step_s"]
        self.to_bytes = 1e6 * step_s / 8  # Mbps-step -> bytes
        ei = m["ext_interf"]
        cfg = replace(ScenarioConfig(name="fake", n=n, n_gateways=len(m["gateways"])),
                      mean_mbps=m["queue_bytes"] / (5 * self.to_bytes), episode_len=10**9,
                      queue_preserve=bool(m.get("queue_preserve", 0)), downtime_steps=round(m["switch_delay_s"] / step_s),
                      ext_interf=None if ei is None else (ei["channel_idx"], ei["start"], ei["end"], ei["noise_dbm"]))
        env = MeshEnv(cfg, m["seed"])
        env.reset()
        # replace the generated topology with the one Python sent
        env.top = Topology(pos=np.asarray(m["pos"], float), gateways=tuple(m["gateways"]))
        env.rg = build_radio_graph(env.top)
        env.gw = env.top.gateways
        env.is_gw = np.zeros(n, bool)
        env.is_gw[list(env.gw)] = True
        env.next_hop = min_hop_next_hop(env.rg.link, env.gw, env.alive)
        env.traffic = _Given()
        self.env, self.closed = env, False

    def step(self, m):
        env, n = self.env, self.env.cfg.n
        drop = {k: np.zeros(n) for k in ("overflow", "mac", "switch", "noroute")}
        alive = np.asarray(m["alive"], bool)
        died = env.alive & ~alive
        drop["switch"][died] = env.backlog[died]  # queued bytes of a node that just died, counted once
        env.backlog[died] = 0.0
        env.alive = alive
        env.traffic.d = np.asarray(m["demand_mbps"], float)
        prev = env.backlog.copy()
        env.step(Action(np.asarray(m["channels"]), np.asarray(m["next_hop"])))
        # ponytail: per-source split of switch vs overflow loss not reconstructed; Ns3MeshEnv only sums drops
        drop["overflow"] += prev + env.demand - env.thr - env.backlog
        us, vs = np.nonzero(env.load > 0)
        snr = snr_db(env.rg.rx_dbm)
        links = [[int(u), int(v), float(env.sent[u, v]), float(env.load[u, v] - env.sent[u, v]),
                  float(snr[u, v]), float(env.cap[u, v])] for u, v in zip(us, vs)]
        b = self.to_bytes
        return {"type": "result", "t": m["t"], "delivered_mbps": env.thr.tolist(),
                "queue_bytes": (env.backlog * b).tolist(), "src_queue_bytes": (env.backlog * b).tolist(),
                "drop_bytes": {k: (v * b).tolist() for k, v in drop.items()}, "links": links}

    def close(self):
        self.closed = True
