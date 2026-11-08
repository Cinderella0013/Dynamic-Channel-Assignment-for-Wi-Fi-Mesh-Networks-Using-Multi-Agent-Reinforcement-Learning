# -*- coding: utf-8 -*-
"""EGAT-MAPPO step-by-step demo, stages 0-10 (small and fast versions of the real pipeline).

python demo/run_demo.py                 # every stage
python demo/run_demo.py --stage 3       # one stage (repeat --stage for several)
python demo/run_demo.py --help          # options

Checkpoints: demo/ckpt/<name>_s0.pt (seed 0 of each learner, copied from runs/v10/ckpt); --ckpt-dir to change.
Outputs: demo/out/. Every stage prints the full-scale command it stands for."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before torch/numpy load: evaluation is pinned to one thread
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU ("" would unset the variable on Windows): the demo is small, results match the thesis runs

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Thai text on Windows consoles (same as PYTHONIOENCODING=utf-8)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "demo" / "out"
SIM = {"airtime": "domain", "mac_eff": 0.65, "contention": 0.0}  # runs/v10/calib.env
SIM_FLAGS = "--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
MARGIN = 2.0  # switch_margin of the main model (runs/v10/fair_nogru.txt)
FAIR_MARGIN = 1.0  # Li+H / Yuan+H (runs/v10/fair_li.txt, fair_yuan.txt)
METRICS = ("aggregate", "jain", "p5", "loss_rate")
A = None  # parsed args


def say(*lines):
    for line in lines:
        print(line, flush=True)


def head(n, title, what):
    say("", "=" * 78, f" ขั้นที่ {n}: {title}", "=" * 78, *[f"  {w}" for w in what], "")


def cmd(c):
    say(f"  คำสั่งเต็ม (ใช้ในงานจริง): {c}")


def ckpt(name):
    p = Path(A.ckpt_dir) / f"{name}_s0.pt"
    if not p.exists():
        raise SystemExit(f"ไม่พบ checkpoint {p} (ระบุ --ckpt-dir ให้ชี้ไปที่ runs/v10/ckpt หรือ demo/ckpt)")
    return str(p)


SPECS = {"m0": ("m0", {}), "ko": ("ko", {}), "li": ("li", {}), "yuan": ("yuan", {}), "q_neutral": ("q_neutral", {}),
         "egat_nogru_h": ("egat_nogru", {"switch_margin": MARGIN}),  # use_gru=false is stored in the checkpoint
         "li_h": ("li", {"switch_margin": FAIR_MARGIN}), "yuan_h": ("yuan", {"switch_margin": FAIR_MARGIN})}
_cache = {}  # per worker process: (spec, overrides) -> policy


def spec(label):
    """CLI-style method spec (m0 | ko | label=checkpoint) and its --agent-set overrides."""
    base, over = SPECS[label]
    return (label if base in ("m0", "ko") else f"{label}={ckpt(base)}"), over


def _episode(job):
    from mesh.eval.run import run_episode
    from mesh.eval.suite import _factory
    from mesh.sim.env import MeshEnv
    from mesh.sim.scenarios import make_scenario
    s, over, scen, seed, overrides = job
    key = (s, json.dumps(over))
    if key not in _cache:
        _cache[key] = _factory(s, over)[1]()
    return run_episode(MeshEnv(replace(make_scenario(scen), **overrides), seed), _cache[key])


def run_jobs(runs, scenarios, seeds, overrides):
    """runs: {label: (spec, agent overrides)} -> {label: {scenario: {metric: [per seed]}}}, episodes in parallel."""
    jobs = [(s, o, sc, sd, overrides) for s, o in runs.values() for sc in scenarios for sd in seeds]
    with ProcessPoolExecutor(A.workers) as ex:
        rows = list(ex.map(_episode, jobs))
    res, i = {}, 0
    for lab in runs:
        for sc in scenarios:
            rs, i = rows[i:i + len(seeds)], i + len(seeds)
            res.setdefault(lab, {})[sc] = {k: [r[k] for r in rs] for k in rs[0]}
    return res


def suite(names, scenarios, seeds, tag, **overrides):
    """Same JSON schema as python -m mesh.eval.suite (read by mesh.eval.composite)."""
    from mesh.eval.suite import _commit
    overrides = {**SIM, **overrides}
    res = run_jobs({n: spec(n) for n in names}, scenarios, list(seeds), overrides)
    path = OUT / f"suite_{tag}.json"
    json.dump({"commit": _commit(), "overrides": overrides, "agent_overrides": {n: SPECS[n][1] for n in names},
               "seeds": list(seeds), "results": res}, open(path, "w"), indent=1)
    return path


def table(path, names, scenarios=None):
    d = json.load(open(path))["results"]
    scenarios = scenarios or list(d[names[0]])
    import numpy as np
    say(f"  {'วิธี':<14}" + "".join(f"{k:>11}" for k in METRICS) + f"{'สลับ/ทาง':>11}")
    for n in names:
        rows = [d[n][s] for s in scenarios]
        vals = [np.mean([np.mean(r[k]) for r in rows]) for k in METRICS]
        sw = np.mean([np.mean(r["n_channel_switch"]) + np.mean(r["n_route_change"]) for r in rows])
        say(f"  {n:<14}" + "".join(f"{v:>11.3f}" for v in vals) + f"{sw:>11.1f}")
    say("  (aggregate = Mb/s รวม, jain = ความเป็นธรรม, p5 = Mb/s ของ flow แย่สุด 5%, loss_rate = สัดส่วนข้อมูลที่หาย,",
        "   สลับ/ทาง = จำนวนการเปลี่ยนช่องสัญญาณ + เปลี่ยนเส้นทาง ต่อ episode; ค่าเฉลี่ยทุก scenario)")


def leads(paths, ours, refs, scenarios=None):
    from mesh.eval.composite import _load, _mean_score
    res, _ = _load([str(p) for p in paths])
    scenarios = scenarios or list(res[ours])
    for r in refs:
        say(f"  composite {ours} เทียบ {r:<10}: {_mean_score(res, ours, r, scenarios):+.3f}  ({_mean_score(res, ours, r, scenarios) * 100:+.1f}%)")
    return res


# ---------------------------------------------------------------- stages
def stage0():
    head(0, "ตรวจสภาพแวดล้อมและ unit test",
         ["ทดสอบว่า Python, ไลบรารี และโค้ดทั้งหมดทำงานถูกต้องบนเครื่องนี้ก่อนเริ่มทดลอง"])
    import numpy, scipy, torch
    say(f"  Python {platform.python_version()} ({platform.system()}), numpy {numpy.__version__}, "
        f"scipy {scipy.__version__}, torch {torch.__version__}",
        f"  GPU (CUDA): {'มี' if torch.cuda.is_available() else 'ไม่มี/ปิดไว้'} (เดโมใช้ CPU เสมอ)")
    tests = [] if A.full_tests else ["tests/test_env.py", "tests/test_egat_act.py", "tests/test_ns3_env.py"]
    cmd("python -m pytest -q")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    say(f"  ผล pytest ({'ทั้งหมด' if A.full_tests else 'ชุดย่อย ' + str(len(tests)) + ' ไฟล์; --full-tests = ทุกไฟล์'}): {last}")
    if r.returncode:
        say(r.stdout[-3000:])
        raise SystemExit("unit test ไม่ผ่าน")


def stage1():
    head(1, "ตัวจำลอง (simulator) และ scenario หนึ่งแบบ",
         ["สร้างเครือข่าย mesh 20 โหนดของ S2 (โหลดสูง) ตามค่าที่ปรับเทียบกับ ns-3 แล้ว (airtime=domain, mac_eff=0.65)",
          "แล้วรัน Ko หนึ่ง episode (200 step = 20 วินาที) เพื่อดูว่า simulator ให้ค่าวัดอะไรออกมา"])
    from mesh.baselines.ko import KoLocalCost
    from mesh.eval.run import run_episode
    from mesh.sim.env import MeshEnv
    from mesh.sim.scenarios import make_scenario
    cfg = replace(make_scenario("S2"), **SIM)
    env = MeshEnv(cfg, 0)
    env.reset()
    say(f"  scenario S2: n={cfg.n} โหนด, gateway={list(map(int, env.gw))}, ลิงก์ทิศทางเดียว={int(env.rg.link.sum())}, "
        f"โหลดเฉลี่ย {cfg.mean_mbps} Mb/s ต่อโหนด, episode {cfg.episode_len} step")
    r = run_episode(MeshEnv(cfg, 0), KoLocalCost())
    say("  ผล Ko บน S2 topology seed 0:", *[f"    {k:<16} {v:10.3f}" for k, v in r.items()])
    cmd("python -m mesh.eval.run --scenario S2 --policy ko --seeds 5   (ไม่มี --set: ตัวจำลองค่าเริ่มต้น)")


def stage2():
    sc = ["S2", "B1", "B3"]
    head(2, "baseline แบบ heuristic: M0 และ Ko",
         ["M0 = ทุกลิงก์ใช้ช่องเดียว + เส้นทาง hop น้อยสุด (จุดอ้างอิงล่างสุด)",
          "Ko = การเลือกช่อง/parent แบบ local cost (เป็นครูของ EGAT ในช่วง behaviour cloning)",
          f"รันบน {','.join(sc)} topology seed 0-{A.seeds - 1}; คาดว่า Ko ดีกว่า M0 ชัดเจน"])
    p = suite(["m0", "ko"], sc, range(A.seeds), "s2_heur")
    table(p, ["m0", "ko"])
    leads([p], "ko", ["m0"])
    cmd(f"python -m mesh.eval.suite --suite A,B --seeds 20 {SIM_FLAGS} --method m0 --method ko --out runs/v10/t0/a/suite_heur.json")


def stage3():
    head(3, "ฝึก EGAT-MAPPO (ไม่มี GRU) แบบสั้น: BC จาก Ko แล้วต่อด้วย MAPPO",
         ["ช่วงแรก agent เลียนแบบ Ko (behaviour cloning, BC) แล้วสลับไปเรียนด้วย MAPPO (PPO หลาย agent)",
          f"เดโมฝึก {A.train_steps} step (BC {A.train_steps // 2} step) งานจริงฝึก 100,000 step (BC 10,000) ต่อ seed ใช้เวลาราว 3 ชั่วโมง",
          "เดโมไม่ได้คาดหวังให้โมเดลเก่ง: แค่แสดงว่า pipeline การฝึกครบ (BC -> MAPPO -> validation -> บันทึก)",
          "validation ของเดโมใช้ S2,B1 x topology 1000 (งานจริง: A+B ทั้ง 11 scenario x topology 1000-1003)"])
    from mesh.rl import train
    train.SCEN_SET.update(SIM)
    train.VAL_SEEDS, train.VAL_SCENARIOS = (1000,), ("S2", "B1")  # demo validation; real run: A+B x 1000-1003
    out = OUT / "egat_nogru_demo.pt"
    t = time.time()
    agent = train.train("egat", A.train_steps, seed=0, out=str(out), log_every=max(A.train_steps // 4, 1),
                        val_every=A.train_steps, use_gru=False, bc_steps=A.train_steps // 2, rollout=200)
    say(f"  ฝึกเสร็จใน {time.time() - t:.0f} วินาที, phase สุดท้าย = {agent.phase}, use_gru = {agent.cfg.use_gru}",
        f"  บันทึก checkpoint ที่ {out.relative_to(ROOT)} (ไฟล์ _val*.pt คือ candidate ที่ใช้เลือกโมเดลด้วย validation)")
    cmd(f"python -m mesh.rl.train --method egat --steps 100000 --seed 0 --val-every 10000 {SIM_FLAGS} "
        "--agent-set use_gru=false --out runs/v10/ckpt/egat_nogru_s0.pt")


def stage4():
    head(4, "โหลด checkpoint ที่ฝึกจริง (egat_nogru_s0.pt) และดูผลของ hysteresis",
         ["โหลดโมเดลหลักที่ฝึกครบ 100k step แล้ว ตรวจ config และรันบน S2 เทียบ margin 0 (ไม่มี hysteresis) กับ margin 2",
          "hysteresis: node จะเปลี่ยนช่อง/parent ก็ต่อเมื่อทางใหม่ดีกว่าทางเดิมเกิน margin จึงสลับน้อยลงมาก"])
    from mesh.eval.run import run_episode
    from mesh.model.agent import EGATAgent
    from mesh.sim.env import MeshEnv
    from mesh.sim.scenarios import make_scenario
    ag = EGATAgent.load(ckpt("egat_nogru"), device="cpu")
    say(f"  โหลด {Path(ckpt('egat_nogru')).name}: use_gru={ag.cfg.use_gru}, พารามิเตอร์ {sum(p.numel() for p in ag.net.parameters()):,} ตัว, "
        f"phase={ag.phase}")
    cfg = replace(make_scenario("S2"), **SIM)
    for m in (0.0, MARGIN):
        ag.cfg.switch_margin = m
        r = run_episode(MeshEnv(cfg, 0), ag)
        say(f"  margin {m:>3}: aggregate {r['aggregate']:6.2f} Mb/s, jain {r['jain']:.3f}, loss {r['loss_rate']:.3f}, "
            f"สลับช่อง {r['n_channel_switch']:.0f} ครั้ง, เปลี่ยน parent {r['n_route_change']:.0f} ครั้ง")
    cmd(f"python -m mesh.eval.suite --suite A,B --seeds 20 {SIM_FLAGS} --method egat_nogru_h=runs/v10/ckpt/egat_nogru_s0.pt "
        "--agent-set switch_margin=2.0 --out ...")


def stage5():
    grid = (0.0, 1.0, 2.0, 4.0)
    head(5, "เลือก hysteresis margin บน validation topology (กฎ worst-case)",
         ["ลองหลาย margin บน topology validation (seed 1000+ ไม่ใช่ topology ที่ใช้ทดสอบ 0-19)",
          "แต่ละ margin คิด composite เทียบ Ko, M0, Li, Yuan, Q-neutral แล้วเลือก margin ที่ 'กรณีแย่สุด' ดีที่สุด",
          f"เดโม: margin {grid}, validation seed {A.val_seeds}, learner seed 0; งานจริงใช้ grid 0-4, seed 1000-1003, 5 training seed"])
    import numpy as np
    from mesh.eval.composite import _mean_score
    from mesh.sim.scenarios import SUITE_A, SUITE_B
    sc = list(SUITE_A + SUITE_B)  # = mesh.rl.train.VAL_SCENARIOS
    s, _ = spec("egat_nogru_h")
    runs = {n: spec(n) for n in ("ko", "m0", "li", "yuan", "q_neutral")}
    runs.update({m: (s, {"switch_margin": m}) for m in grid})
    res = run_jobs(runs, sc, A.val_seeds, SIM)
    best = None
    for m in grid:
        v = res[m]
        row = {b: _mean_score({"o": v, "r": res[b]}, "o", "r", sc) for b in ("ko", "m0", "li", "yuan", "q_neutral")}
        sw = np.mean([np.mean(v[x]["n_channel_switch"]) + np.mean(v[x]["n_route_change"]) for x in sc])
        worst = min(row.values())
        say(f"  margin {m:>3}: " + " ".join(f"{b} {x:+.3f}" for b, x in row.items()) + f" | แย่สุด {worst:+.3f} | สลับ {sw:.0f}")
        if best is None or worst > best[1]:
            best = (m, worst)
    say(f"  margin ที่เลือกในเดโม = {best[0]}  (งานจริงเลือกได้ 2.0: runs/v10/fair_nogru.txt)")
    cmd("python runs/v10/wc_val.py nogru <k> <margin>  แล้ว  python runs/v10/fair_pick.py nogru")


def stage6():
    from mesh.sim.scenarios import SUITE_A, SUITE_B
    names = ["egat_nogru_h", "ko", "li", "yuan", "m0", "q_neutral"]
    head(6, "ประเมินผลบน scenario A+B (11 แบบ) เทียบทุก baseline",
         ["A = S1-S6 (โหลดต่ำ/สูง, hotspot, สัญญาณรบกวนภายนอก, เปลี่ยนรูปแบบโหลด, node เสีย)",
          "B = B1-B5 (จุดอ่อนที่รู้ของ baseline: hotspot เคลื่อนที่, โหลดเป็นคาบ, เครือข่ายห่าง, topology แบบ grid, เปลี่ยนเร็ว)",
          f"เดโม: topology seed 0-{A.seeds - 1}, learner training seed 0; งานจริง: seed 0-19 x training seed 0-4"])
    p = suite(names, list(SUITE_A + SUITE_B), range(A.seeds), "s6_ab")
    table(p, names)
    cmd(f"python -m mesh.eval.suite --suite A,B,C --seeds 20 {SIM_FLAGS} --method egat_nogru_h=runs/v10/ckpt/egat_nogru_s0.pt "
        "--agent-set switch_margin=2.0 --out runs/v10/t0/a/suite_egat_nogru_h.json")


def stage7():
    head(7, "composite score และช่วงความเชื่อมั่น (confidence interval)",
         ["composite = ผลรวมถ่วงน้ำหนักของการปรับปรุงสัมพัทธ์ 5 ตัววัด (aggregate 0.25, jain 0.20, p5 0.20, loss 0.20,",
          "ความนิ่ง 0.15) เทียบ baseline; +0.10 = ดีกว่า 10% (กำหนดไว้ล่วงหน้าก่อนเห็นผล)",
          "CI: bootstrap และ t-interval ที่นับความแปรปรวนของ training seed ทั้งสองฝั่ง"])
    p = OUT / "suite_s6_ab.json"
    if not p.exists():
        stage6()
    from mesh.eval.composite import _load, _mean_score, report, significance
    res, _ = _load([str(p)])
    sc = list(res["egat_nogru_h"])
    say("  ผลเดโม (training seed เดียว, topology น้อย: ใช้ดูวิธีคำนวณ ไม่ใช่ผลทางสถิติ), bootstrap 95% CI จาก 300 รอบ:")
    for ref in ("ko", "li", "yuan", "m0", "q_neutral"):
        sg = significance(res, "egat_nogru_h", ref, sc, n_boot=300)
        say(f"    เทียบ {ref:<10} composite {_mean_score(res, 'egat_nogru_h', ref, sc):+.3f}  CI [{sg['ci'][0]:+.3f}, {sg['ci'][1]:+.3f}]"
            f"  scenario ที่ดีกว่าอย่างมีนัยสำคัญ (bootstrap, Holm) {sg['frac_sig_boot']:.0%}")
    if A.full_report:
        (OUT / "composite_demo.md").write_text(report([str(p)], ours="egat_nogru_h"), encoding="utf-8")
        say(f"  รายงานเต็มของเดโม: {(OUT / 'composite_demo.md').relative_to(ROOT)}")
    real = Path(A.runs_dir) / "v10/final/composite_a_nogru_test.md"
    if real.exists():
        rows = [l for l in real.read_text(encoding="utf-8").split("## Significance")[1].split("Per-scenario tests")[0].splitlines() if l.startswith("| ")][1:]
        say("", "  ผลจริงในวิทยานิพนธ์ (seed 0-19, 5 training seed, setting a) คอลัมน์ seed-pair mean และ 95% t-interval:")
        for l in rows:
            c = [x.strip() for x in l.strip("|").split("|")]
            say(f"    เทียบ {c[0]:<10} {c[4]}  {c[5]}" + ("  (egat_h2 = EGAT มี GRU + H)" if c[0] == "egat_h2" else ""))
    else:
        say(f"  (ไม่พบผลจริง {real}: ตั้ง --runs-dir ให้ชี้ไปยังโฟลเดอร์ runs ของเครื่องที่รันการทดลองเต็ม)")
    cmd("python -m mesh.eval.composite runs/v10/t{0..4}/a/suite_*.json ... --ours egat_nogru_h > composite.md  (ดู runs/v10/nogru.sh)")


def stage8():
    from mesh.sim.scenarios import SUITE_A, SUITE_B
    head(8, "ตรวจความเป็นธรรม: เพิ่ม hysteresis ให้ Li และ Yuan ด้วย (Li+H, Yuan+H)",
         ["คำถาม: EGAT ชนะเพราะ hysteresis อย่างเดียวหรือไม่? จึงให้ baseline ได้ hysteresis แบบเดียวกัน",
          f"(margin {FAIR_MARGIN} เลือกด้วยกฎ worst-case เดียวกันบน validation) แล้วเทียบอีกครั้ง"])
    p6 = OUT / "suite_s6_ab.json"
    if not p6.exists():
        stage6()
    p = suite(["li_h", "yuan_h"], list(SUITE_A + SUITE_B), range(A.seeds), "s8_fair")
    table(p, ["li_h", "yuan_h"])
    leads([p6, p], "egat_nogru_h", ["li", "li_h", "yuan", "yuan_h"])
    say("  ถ้า EGAT ยังนำ Li+H/Yuan+H แปลว่าข้อได้เปรียบไม่ได้มาจาก hysteresis อย่างเดียว")
    cmd(f"python -m mesh.eval.suite --suite A,B --seeds 20 {SIM_FLAGS} --method li_h=runs/v10/ckpt/li_s0.pt "
        "--agent-set switch_margin=1 --out ...")


def stage9():
    from mesh.sim.scenarios import SUITE_A, SUITE_B
    names = ["egat_nogru_h", "ko", "m0", "yuan_h"]
    head(9, "setting (b) และ topology ใหม่ที่ไม่เคยใช้ (fresh)",
         ["setting (b): ทุกวิธีเก็บคิวไว้ตอนสลับช่อง (แบบ CSA) จึงไม่เสียข้อมูลจากการสลับ: ตัดข้อได้เปรียบของการสลับน้อย",
          "fresh: topology seed 40+ ที่ไม่มีการตัดสินใจใดเคยเห็น (งานจริง 20-39 และ 40-59 ชุดละ 20)"])
    sc = list(SUITE_A + SUITE_B)
    pb = suite(names, sc, range(A.seeds), "s9_b", queue_preserve=True)
    say("  setting (b), topology seed 0-%d:" % (A.seeds - 1))
    leads([pb], "egat_nogru_h", names[1:])
    pf = suite(names, sc, range(40, 40 + A.seeds), "s9_fresh")
    say("  setting (a), fresh topology seed 40-%d:" % (39 + A.seeds))
    leads([pf], "egat_nogru_h", names[1:])
    cmd(f"python -m mesh.eval.suite --suite A,B --seeds 20 --seed-start 40 {SIM_FLAGS} [--set queue_preserve=true] --method ...")


def _wsl_path(p):
    p = Path(p).resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}" if p.drive else p.as_posix()


def stage10():
    head(10, "ยืนยันผลด้วย ns-3 (packet-level 802.11ax)",
         ["นำ policy ที่ฝึกแล้ว (ไม่ฝึกใหม่) ไปรันใน ns-3.47 ซึ่งจำลองระดับ packet: MAC/PHY, คิว, การชนกันจริง",
          "เดโม: S2 topology seed 0, Ko เทียบ EGAT+H, MAC แบบ RTS/CTS; งานจริง: S2,B1,B3 x seed 0-4 x 5 training seed"])
    binary = A.ns3_binary
    nogru = ckpt("egat_nogru")
    args = ["-m", "mesh.ns3.suite", "--scenarios", "S2", "--seeds", "1", "--workers", "2", "--mac", '{"rts": 1}',
            *[x for k, v in SIM.items() for x in ("--set", f"{k}={v}")], "--method", "ko",
            "--method", "egat_nogru_h=CK", "--agent-set", f"switch_margin={MARGIN}", "--binary", binary, "--out", "OUT"]
    out = OUT / "ns3_demo.json"
    if platform.system() == "Windows":
        if not shutil.which("wsl.exe"):
            return say("  ข้าม: ไม่พบ WSL (ns-3 ต้องรันบน Linux หรือ WSL; ดูคู่มือหัวข้อ ns-3)")
        chk = subprocess.run(["wsl.exe", "-e", "bash", "-lc", f"test -x {binary} && python3 -c 'import torch, scipy'"],
                             capture_output=True)
        if chk.returncode:
            return say(f"  ข้าม: ใน WSL ไม่พบ {binary} หรือ python3 ยังไม่มี torch/scipy (สร้างด้วย bash ns3/build.sh)")
        run = ["wsl.exe", "--cd", _wsl_path(ROOT), "-e", "env", "OMP_NUM_THREADS=1", "python3",
               *[a.replace("CK", _wsl_path(nogru)).replace("OUT", _wsl_path(out)) for a in args]]
    else:
        if not Path(os.path.expanduser(binary)).exists():
            return say(f"  ข้าม: ไม่พบ ns-3 binary {binary} (สร้างด้วย bash ns3/build.sh)")
        run = [sys.executable, *[a.replace("CK", nogru).replace("OUT", str(out)) for a in args]]
    say("  กำลังรัน ns-3 (หนึ่ง process ต่อ episode)...")
    r = subprocess.run(run, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        say(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("ns-3 ล้มเหลว")
    table(out, ["ko", "egat_nogru_h"])
    leads([out], "egat_nogru_h", ["ko"])
    say("  (topology เดียว ตัวเลขแกว่งมาก ใช้ดูว่า pipeline ns-3 ทำงานและทิศทางตรงกับตัวจำลอง)")
    say("  ผลจริง (runs/v10/final/ns3_leads_nogru.txt): EGAT+H นำ Ko +23.1% CI [+15.5, +30.7] ใน ns-3 RTS/CTS")
    cmd('wsl.exe -e bash /mnt/e/MESH/runs/ns3/run_matrix_nogru.sh 2   (ภายใน: python3 -m mesh.ns3.suite --workers 8 --mac \'{"rts": 1}\' ...)')


STAGES = [stage0, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10]


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, action="append", choices=range(len(STAGES)), help="ขั้นที่จะรัน (ซ้ำได้); ไม่ระบุ = ทุกขั้น")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "demo" / "ckpt"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"), help="โฟลเดอร์ runs ของการทดลองเต็ม (สำหรับผลจริงในขั้น 7)")
    ap.add_argument("--seeds", type=int, default=2, help="จำนวน topology ต่อ scenario ในเดโม")
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[1000, 1001], help="validation topology ในขั้น 5")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)), help="จำนวน process ขนาน")
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--full-report", action="store_true", help="ขั้น 7 เขียนรายงาน composite เต็ม (bootstrap 2000 รอบ, ~4 นาที)")
    ap.add_argument("--full-tests", action="store_true", help="ขั้น 0 รัน pytest ทุกไฟล์")
    ap.add_argument("--ns3-binary", default="~/ns3_validation/ns-3.47/build/scratch/ns3.47-mesh_bridge")
    A = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    times = []
    for i in A.stage or range(len(STAGES)):
        t = time.time()
        STAGES[i]()
        times.append((i, time.time() - t))
        say(f"  [ขั้นที่ {i} เสร็จใน {times[-1][1]:.0f} วินาที]")
    say("", "สรุปเวลา: " + ", ".join(f"ขั้น {i} {s:.0f}s" for i, s in times) + f"  รวม {sum(s for _, s in times) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
