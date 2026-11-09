"""Comment 4: inference latency, peak memory and communication of the final EGAT policy at n = 20/50/80.

PYTHONPATH=. python docs/paper/review/c4_bench.py > docs/paper/review/c4_bench.out.txt

Each (n, device) runs in its own subprocess so peak RSS / CUDA peak belong to that configuration only.
Latency = agent.act(obs): observation tensors, Ko prior, potential mask, encoder, heads, hysteresis,
sequential channel decoding, entropy fallback; env.step is excluded. All n agents are computed in one
centralised call, as in the simulator.
"""
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"  # CPU single-thread (numpy and torch)

import json
import platform
import subprocess
import time
import tracemalloc

import numpy as np
import torch

CKPT = "runs/v10/ckpt/egat_nogru_s0.pt"
MARGIN = 2.0
SCEN = {20: "S2", 50: "C3", 80: "C4"}  # all rgg, 2 gateways, 10 Mb/s mean load
LAT_SEEDS = (0, 1)                     # 2 episodes x 200 steps, first 10 steps discarded
WARMUP = 10
TOPO_SEEDS = range(20)                 # evaluation seeds 0-19 for topology statistics
PIN_CPU = 2                            # logical CPU on a P-core (logical 0-11 = P-cores with HT on this part)


def bf_iterations(obs):
    """Synchronous Bellman-Ford rounds until ko_route_via's distances stop changing (mesh/baselines/ko.py:31-36)."""
    from mesh.rl.agents_io import alive_mask
    from mesh.sim.env import edge_matrix
    from mesh.sim.radio import rate_mbps
    alive = alive_mask(obs)
    n = len(alive)
    link = (edge_matrix(obs, "is_link") > 0) & alive[:, None] & alive[None, :]
    rate = np.where(link, rate_mbps(edge_matrix(obs, "snr_db")), 0.0)
    w = np.where(rate > 0, (1 + edge_matrix(obs, "load_frac")) / np.where(rate > 0, rate, 1.0), np.inf)
    gw = [g for g in obs.gateways if alive[g]]
    dist = np.full(n, np.inf)
    dist[gw] = 0.0
    for it in range(1, n + 1):
        new = np.minimum(dist, (w + dist[None, :]).min(axis=1))
        new[gw] = 0.0
        if np.array_equal(new, dist):
            return it
        dist = new
    return n


def decode_rounds(obs, slots, j):
    """Tree depth and the critical path of sequential_channels (mesh/rl/agents_io.py:110-133).

    Node u must wait for every conflict neighbour earlier in the (depth, id) order, so the number of
    sequential rounds is the longest chain in that precedence DAG, not the tree depth."""
    from mesh.rl.agents_io import _valid_slots, alive_mask
    from mesh.sim.env import edge_matrix
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
    order = sorted(range(n), key=lambda u: (depth[u], u))
    chain = np.ones(n, dtype=int)
    for k, u in enumerate(order):
        before = [v for v in order[:k] if conflict[u, v]]
        if before:
            chain[u] = 1 + max(chain[v] for v in before)
    return int(depth.max()) + 1, int(chain.max())


def worker(n, device):
    if device == "cpu":
        torch.set_num_threads(1)
    import psutil
    from mesh.model.agent import EGATAgent
    from mesh.model.ko_prior import ko_channel_costs, ko_slot_costs
    from mesh.sim.env import MeshEnv, edge_matrix
    from mesh.sim.scenarios import make_scenario

    proc = psutil.Process()
    # i5-13450HX is hybrid (6 P + 4 E cores); unpinned single-thread timings varied 3-6x between runs
    # as Windows moved the thread across core types. Pin to one P-core logical CPU.
    proc.cpu_affinity([PIN_CPU])
    agent = EGATAgent.load(CKPT, device=device)
    agent.cfg.switch_margin = MARGIN
    assert agent.phase == "ppo" and not agent.cfg.use_gru
    cfg = make_scenario(SCEN[n])
    assert cfg.n == n
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    lat, t_prep, t_ko, t_fwd, depth, crit, fb = [], [], [], [], [], [], []
    for seed in LAT_SEEDS:
        env = MeshEnv(cfg, seed)
        obs = env.reset()
        for t in range(cfg.episode_len):
            sync()
            t0 = time.perf_counter()
            act = agent.act(obs)
            sync()
            dt = time.perf_counter() - t0
            if t >= WARMUP:
                lat.append(dt)
                fb.append(agent.fallback_rate)
                p = agent._pending["p"]
                d, c = decode_rounds(obs, p["slots"], agent._pending["j"])
                depth.append(d)
                crit.append(c)
                if t % 20 == 0:  # component split on a subsample (does not change agent state)
                    t0 = time.perf_counter(); agent.prepare(obs); t_prep.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); ko_slot_costs(obs, p["slots"]); ko_channel_costs(obs)
                    t_ko.append(time.perf_counter() - t0)
                    with torch.no_grad():
                        sync(); t0 = time.perf_counter(); agent._tensors([p], agent.h[None]); sync()
                    t_fwd.append(time.perf_counter() - t0)
            obs, _, _ = env.step(act)
    rss_now = proc.memory_info().rss
    peak_wset = getattr(proc.memory_info(), "peak_wset", None)  # Windows only
    cuda_peak = torch.cuda.max_memory_allocated() if device == "cuda" else None

    # tracemalloc (numpy/Python heap only; torch CPU tensors are not traced) on 20 decisions
    env = MeshEnv(cfg, 0)
    obs = env.reset()
    tracemalloc.start()
    for _ in range(20):
        act = agent.act(obs)
        obs, _, _ = env.step(act)
    tm_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # topology statistics on evaluation seeds 0-19
    deg_gat, deg_link, deg_conf, ecc, bf = [], [], [], [], []
    for seed in TOPO_SEEDS:
        obs = MeshEnv(cfg, seed).reset()
        link = edge_matrix(obs, "is_link") > 0
        conf = edge_matrix(obs, "is_conflict") > 0
        deg_gat.append((link | conf).sum(1) + 1)  # incl. self-link (adj |= eye, agent.py:55)
        deg_link.append(link.sum(1))
        deg_conf.append(conf.sum(1))
        ecc.append(int(obs.node_feat[:, 3].max()))
        bf.append(bf_iterations(obs))
    deg_gat, deg_link, deg_conf = map(np.concatenate, (deg_gat, deg_link, deg_conf))
    E = int(deg_gat.sum() / len(TOPO_SEEDS))

    ms = lambda a: np.asarray(a) * 1e3
    return dict(
        n=n, scenario=cfg.name, device=device, steps=len(lat),
        lat_med=float(np.median(ms(lat))), lat_p95=float(np.percentile(ms(lat), 95)),
        lat_max=float(ms(lat).max()),
        prep_med=float(np.median(ms(t_prep))), ko_med=float(np.median(ms(t_ko))),
        fwd_med=float(np.median(ms(t_fwd))),
        rss_mb=rss_now / 2**20, peak_wset_mb=None if peak_wset is None else peak_wset / 2**20,
        cuda_peak_mb=None if cuda_peak is None else cuda_peak / 2**20, tracemalloc_peak_mb=tm_peak / 2**20,
        dense_edge_tensor_mb=n * n * 7 * 4 / 2**20, dense_gat_z_mb=n * n * agent.cfg.hidden * 4 / 2**20,
        fallback_rate=float(np.mean(fb)),
        tree_depth=dict(zip(*map(lambda x: x.tolist(), np.unique(depth, return_counts=True)))),
        tree_depth_med=float(np.median(depth)), tree_depth_max=int(max(depth)),
        crit_med=float(np.median(crit)), crit_p95=float(np.percentile(crit, 95)), crit_max=int(max(crit)),
        E_mean=E, E_over_n2=E / n**2,
        deg_gat_mean=float(deg_gat.mean()), deg_gat_max=int(deg_gat.max()),
        deg_link_mean=float(deg_link.mean()), deg_conf_mean=float(deg_conf.mean()), deg_conf_max=int(deg_conf.max()),
        hop_ecc_med=float(np.median(ecc)), hop_ecc_max=int(max(ecc)),
        bf_iter_med=float(np.median(bf)), bf_iter_max=int(max(bf)),
    )


def hardware():
    cpu = platform.processor()
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True).stdout.strip() or cpu
    except OSError:
        pass
    import psutil
    return dict(cpu=cpu, cores=psutil.cpu_count(logical=False), threads=psutil.cpu_count(),
                ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__)


def main():
    if len(sys.argv) == 3:
        print(json.dumps(worker(int(sys.argv[1]), sys.argv[2])))
        return
    print("hardware", json.dumps(hardware()))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    import psutil
    for dev in devices:
        for n in SCEN:
            others = [p.info["cmdline"][-1] for p in psutil.process_iter(["name", "cmdline", "pid"])
                      if (p.info["name"] or "").startswith("python") and p.info["pid"] != os.getpid()
                      and p.info["cmdline"]]
            print(f"background n={n} {dev}: cpu_load={psutil.cpu_percent(interval=1.0)}% "
                  f"other_python={others}", flush=True)  # contention check: latency tails are load-sensitive
            out = subprocess.run([sys.executable, __file__, str(n), dev], capture_output=True, text=True,
                                 env={**os.environ, "PYTHONPATH": "."})
            if out.returncode:
                sys.exit(out.stderr)
            print("result", out.stdout.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
