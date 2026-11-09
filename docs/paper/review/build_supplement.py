"""Build the versioned study supplement (advisor review v5, comment 6). Stdlib only, no new runs.

    python docs/paper/review/build_supplement.py        (from the repo root)

Writes share/supplement/NW14_EGAT_supplement_<git-short-sha>.zip with
  code/         git archive of HEAD (tracked source, configs, docs, figures scripts)
  runs/         runs/v10 (checkpoints, suite JSONs, reports, logs, scripts), runs/ns3/{v10,v10base} + ns-3 matrix scripts
  MANIFEST.md   commit, environment, seed map, settings, protocol history, regeneration commands, SHA-256 list
  build_supplement.py, c6_supplement.md   this script and the manuscript text that describes the archive
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_DIRS = ["runs/v10", "runs/ns3/v10", "runs/ns3/v10base"]
NS3_EXTRA = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "runs/ns3").glob("*.sh")) + ["runs/ns3/events.log"]
BIG_LOG = 5 * 2**20  # logs above this are left out; events.log, *.json, *.md are always kept
PROTO = "docs/paper/protocol_nogru.md"


def sh(*cmd, binary=False, timeout=120):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=timeout)
        return r.stdout if binary else r.stdout.decode("utf-8", "replace").strip()
    except Exception as e:  # environment records are best effort (e.g. no WSL)
        return b"" if binary else f"(unavailable: {e})"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def keep(p):
    return not (p.suffix == ".log" and p.name != "events.log" and p.stat().st_size > BIG_LOG)


def run_files():
    files, skipped = [], []
    for d in RUN_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file():
                (files if keep(p) else skipped).append(p)
    files += [ROOT / f for f in NS3_EXTRA]
    return files, skipped


def ts(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def protocol_history():
    """Dated evidence that the no-GRU protocol was fixed before the 40-59 confirmation runs."""
    st = (ROOT / PROTO).stat()
    born = getattr(st, "st_birthtime", st.st_ctime)  # Windows: creation time
    blob_now = sh("git", "hash-object", PROTO)
    first = sh("git", "log", "--follow", "--diff-filter=A", "--format=%h", "--", PROTO).split()[-1]
    blob_first = sh("git", "rev-parse", f"{first}:{PROTO}")
    ev = (ROOT / "runs/v10/events.log").read_text().splitlines()
    ev_sel = [l for l in ev if re.search(r"launch|nogru|ablation all done|abl m2 done", l)]
    # every suite file evaluated on topologies 40-59, with its commit field and write time
    f40 = []
    for p in (ROOT / "runs").rglob("*.json"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and any(40 <= s < 60 for s in d.get("seeds", []) if isinstance(s, int)):
            f40.append((p.stat().st_mtime, str(p.relative_to(ROOT)).replace("\\", "/"), d.get("commit", "?")))
    f40.sort()
    commits = sorted({c for _, _, c in f40})
    nogru = (ROOT / "runs/v10/nogru.sh").read_text().splitlines()[1]
    out = [
        "### git history of the protocol file",
        "```", sh("git", "log", "--follow", "--format=%H  %ad  %s", "--date=iso", "--", PROTO), "```",
        f"- File system: `{PROTO}` created {ts(born)}, last modified {ts(st.st_mtime)} (never edited after creation).",
        f"- Content identity: working-copy blob `{blob_now[:12]}`, blob in first commit `{first}` `{blob_first[:12]}`"
        f" -> {'identical' if blob_now == blob_first else 'DIFFERENT'}; the committed text is the text written at creation.",
        f"- `runs/v10/nogru.sh` (created {ts((ROOT / 'runs/v10/nogru.sh').stat().st_mtime)}) line 2: `{nogru}`",
        f"- Suite files evaluated on topologies 40-59: {len(f40)}; earliest written {ts(f40[0][0])}, latest {ts(f40[-1][0])}.",
        f"- Their `commit` fields: {', '.join(f'`{c}`' for c in commits)}. `8aaaf1c` (2026-11-06 23:17) does not contain"
        f" the protocol file (`git cat-file -e 8aaaf1c:{PROTO}` fails); `-dirty` = uncommitted working-tree changes,"
        f" consistent with the protocol file existing, uncommitted, when the runs were made.",
        "", "### runs/v10/events.log (launch, ablation and no-GRU confirmation lines)", "```", *ev_sel, "```",
    ]
    verdict = born < min(t for t, _, _ in f40)
    return "\n".join(out), verdict, born, f40


STATIC = r"""
## Seed map
| role | seeds / topologies | where |
|---|---|---|
| training seeds (all learners) | 0-4 (`--seed k`); ablation 0-2 | `runs/v10/ckpt/<method>_s<k>.pt` |
| validation topologies | 1000-1003 (`VAL_SEEDS`, `mesh/rl/train.py`); checkpoint and hysteresis-margin selection | `*_val<step>.pt`, `hyst_val.txt`, `fair_*.txt` |
| test (exploratory) topologies | 0-19 | `runs/v10/t{0..4}/{a,b}` (t<k> = training seed k) |
| fresh topologies | 20-39 | `runs/v10/fresh/t{0..4}/{a,b}` |
| confirmation topologies (pre-registered) | 40-59 | `runs/v10/fresh2/t{0..4}/{a,b}` |
| sim airtime calibration vs ns-3 | fit on S2 and B3, B1 held out; topologies 0-4 | `runs/v10/calib.env`, `code/ns3/calibrate.py`, `code/docs/ns3/protocol.md` §7 |
| ns-3 evaluation | S2, B1, B3; topologies 0-4 | `runs/ns3/v10`, `runs/ns3/v10base` |

Heuristics M0 and Ko are deterministic: one file per setting (`suite_heur.json` / `heur_<s>.json`), reused for every training seed.

## Settings (a) and (b)
- Simulator for all v10 runs: `--set airtime=domain --set mac_eff=0.65 --set contention=0.0` (`runs/v10/calib.env`).
- **(a)** default: when a device leaves a channel its queued packets are dropped (counted as switch drops).
- **(b)** adds `--set queue_preserve=true`: queues are kept across a channel switch and re-forwarded on the new
  route (CSA-style buffering; `mesh/sim/env.py` `ScenarioConfig.queue_preserve`, ns-3 `init` key `"queue_preserve": 1`).
- Directories: `.../a/` = setting (a), `.../b/` = setting (b); reports `composite_a_*` / `composite_b_*`.

## Three-seed ablation (training seeds 0-2, margin 2, `runs/v10/abl_m2.sh`)
- checkpoints: `runs/v10/ckpt/egat_{noedge,nogru,noko,nobc}_s{0,1,2}.pt` (flags `use_edge_feat`, `use_gru`, `use_ko_prior`, `use_bc` = false; `runs/v10/job_abl.sh`)
- suites: `runs/v10/t{0,1,2}/{a,b}/suite_egat_{noedge,nogru,noko,nobc}.json`; full-GRU reference `suite_egat_h2.json` seeds 0-2.

## ns-3 MAC settings (ns-3.47, 802.11ax, 20 MHz, A-MPDU 65535 B, A-MSDU off)
- **RTS/CTS**: `--mac '{"rts": 1}'` -> `runs/ns3/v10` (`runs/ns3/run_matrix_v10.sh`, `_h2`, `_fair`, `run_matrix_nogru.sh`)
- **base MAC** (no RTS/CTS): `--mac '{}'` -> `runs/ns3/v10base` (`runs/ns3/run_matrix_v10base.sh`, same scripts with base MAC)
- Each ns-3 suite JSON records `ns3.binary` and `ns3.mac`. Policies are evaluated without further training.

## Regeneration commands
Set up (the scripts expect `runs/` inside the source tree; `code/runs/` already holds a few tracked v8 files, so merge):
```
unzip NW14_EGAT_supplement_<sha>.zip -d sup && cd sup
cp -r runs/. code/runs/ && cd code     # Windows: robocopy runs code\runs /E  then  cd code
pip install -e .                       # or the versions in "pip freeze" below
```
| output | command (from `code/`) | inputs |
|---|---|---|
| Tables 5-9 (+ 0.093/0.094 check, zero-reference check) | `PYTHONPATH=. python docs/paper/review/reconstruct_tables.py > tables.md` then compare with `docs/paper/review/reconstruct_tables.out.md` | suite JSONs in `runs/v10`, `runs/ns3/{v10,v10base}` |
| composite reports `runs/v10/**/final/composite_*.md` | the `rep()` lines of `runs/v10/report.sh` and `runs/v10/nogru.sh` (`python -m mesh.eval.composite <suite files> --ours <label>`) | same |
| ns-3 lead summaries `runs/v10/final/ns3_leads_*.txt` | written by an interactive command that was not saved; the Table 9 section of `reconstruct_tables.py` recomputes the S2/B1/B3 values from the same suite files | `runs/ns3/{v10,v10base}` |
| Figures (main, scenarios, switching) | `python docs/paper/figures/make_figures.py` | `runs/v10/final`, `runs/v10/fresh*/final` |
| Figure 1 (architecture) | `python docs/paper/figures/make_architecture.py` | none |
| Tables 1-4, 10 | descriptive, no data | - |
| re-running evaluations (optional) | the `python -m mesh.eval.suite ...` lines in `runs/v10/*.sh`, `python3 -m mesh.ns3.suite ...` in `runs/ns3/*.sh` (WSL, after `bash ns3/build.sh`) | checkpoints |
"""


def manifest(sha_full, files, skipped):
    hist, verdict, _, _ = protocol_history()
    wsl = sh("wsl.exe", "-e", "bash", "-c",
             "cd ~/ns3_validation/ns-3.47 && echo ns-3 VERSION: $(cat VERSION) && g++ --version | head -1 && "
             "cmake --version | head -1 && python3 --version && sha256sum build/scratch/ns3.47-mesh_bridge", timeout=60)
    hashed = [p for p in files if p.suffix == ".pt" or (p.suffix == ".json" and re.match(r"(suite|heur)_", p.name))]
    lines = [
        "# NW14 EGAT-MAPPO study supplement", "",
        f"- Commit: `{sha_full}` (branch `{sh('git', 'rev-parse', '--abbrev-ref', 'HEAD')}`), built "
        f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "- `code/` = `git archive` of that commit. `runs/` = the recorded outputs (not in git). No model was retrained for this archive.",
        f"- Run files: {len(files)}; left out (logs > 5 MB): {len(skipped)}"
        + (": " + ", ".join(str(p.relative_to(ROOT)) for p in skipped) if skipped else "."),
        "", "## Environment",
        f"- `python --version`: {sh(sys.executable, '--version')}",
        f"- torch: {sh(sys.executable, '-c', 'import torch; print(torch.__version__)')}",
        f"- OS: {sys.platform}; ns-3 runs under WSL Ubuntu", "", "WSL / ns-3 build record:", "```", wsl, "```",
        "ns-3 build script `code/ns3/build.sh` (bridge source `code/ns3/mesh_bridge.cc`; model and MAC notes: "
        "`code/docs/ns3/protocol.md` §3, §6, §7):", "```bash", (ROOT / "ns3/build.sh").read_text().strip(), "```",
        "`pip freeze`:", "```", sh(sys.executable, "-m", "pip", "freeze"), "```",
        STATIC,
        "## Dated protocol history (`docs/paper/protocol_nogru.md`)",
        hist, "",
        f"**Result:** protocol file created before the first 40-59 evaluation: **{'yes' if verdict else 'NO'}**. "
        "The git commit of the protocol (2026-11-07 09:42) came after the runs; the earlier date rests on the file-system "
        "creation time, the unchanged blob, the `-dirty` commit fields and the events.log order.",
        "", "## Files", "SHA-256 for checkpoints (`*.pt`) and suite JSONs; size for everything else.", "",
        "| file | bytes | sha256 |", "|---|---|---|",
    ]
    for p in files:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        lines.append(f"| {rel} | {p.stat().st_size} | {sha256(p) if p in hashed else ''} |")
    return "\n".join(lines) + "\n", verdict


def main():
    sha_full = sh("git", "rev-parse", "HEAD")
    out = ROOT / "share/supplement" / f"NW14_EGAT_supplement_{sha_full[:7]}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files, skipped = run_files()
    text, verdict = manifest(sha_full, files, skipped)
    tar = tarfile.open(fileobj=io.BytesIO(sh("git", "archive", "--format=tar", "HEAD", binary=True)))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("MANIFEST.md", text)
        z.write(__file__, "build_supplement.py")
        c6 = Path(__file__).with_name("c6_supplement.md")
        if c6.exists():
            z.write(c6, "c6_supplement.md")
        for m in tar.getmembers():
            if m.isfile():
                z.writestr("code/" + m.name, tar.extractfile(m).read())
        for p in files:
            z.write(p, str(p.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(out) as z:  # reopen: the archive must read back cleanly
        assert z.testzip() is None, "corrupt member"
        n = len(z.namelist())
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MB  {n} entries  protocol-before-40-59={verdict}")


if __name__ == "__main__":
    main()
