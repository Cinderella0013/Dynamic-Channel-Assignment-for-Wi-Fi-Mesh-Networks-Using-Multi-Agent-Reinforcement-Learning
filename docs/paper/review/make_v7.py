"""Build egat-mappo-v7.txt from v6 plus the per-comment answers: python docs/paper/review/make_v7.py

- §2 Table 2 and the "Recent mesh studies" paragraph are replaced from c1_related_work.md.
- Every advisor comment (`// Please ...`) gets one `// Authors' status` line (existing RESOLVED lines are kept,
  comment 2's PARTLY line is replaced) pointing to the matching response.
- The full responses R1-R7 are appended, so the file reads without the repository.
"""
import re
from pathlib import Path

D = Path(__file__).resolve().parent
v6 = (D / "egat-mappo-v6.txt").read_text(encoding="utf-8").split("\n")
c1 = (D / "c1_related_work.md").read_text(encoding="utf-8").split("\n")


def section(lines, start, stop):
    i = next(k for k, x in enumerate(lines) if x.startswith(start))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith(stop))
    return lines[i:j]


# §2: Table 2 block and paragraph
tbl = [x for x in section(c1, "Table 2.", "Note for authors") if x.strip()]
par = next(x for x in section(c1, "## 3.", "(Word count") if x.startswith("Recent mesh studies"))
i = next(k for k, x in enumerate(v6) if x.startswith("Table 2. Position"))
j = next(k for k in range(i, len(v6)) if v6[k].startswith("This work |"))
v6[i:j + 1] = tbl
k = [n for n, x in enumerate(v6) if x.startswith("Recent mesh studies sharpen that comparison")]
assert len(k) == 1
v6[k[0]] = par

# status lines under each comment
STATUS = {
    1: "PARTLY RESOLVED. Table 2 and Section 2 rewritten at mechanism level. Full text of [16] read and cited by "
       "section/page; [15] verified from abstract, keywords and figure captions only; [17] from bibliographic "
       "metadata only (full texts inaccessible). See Response R1.",
    2: "RESOLVED. All definitions now come from the code with file:line references (38-entry observation, q_max, "
       "step order, potential and mask, gateway/no-parent rules, u_i, n_alive, counters, airtime rule); nine "
       "sentences corrected. See Response R2.",
    3: "RESOLVED. One training transition and one deterministic evaluation epoch traced from the code; the 2-nat "
       "fallback uses the sum of the two head entropies and fires on 9 of 43,901 decisions. See Response R3.",
    4: "RESOLVED. Complexity checked against the tensors; latency, memory and coordination measured at 20/50/80 "
       "nodes on stated hardware. See Response R4.",
    6: "RESOLVED. Versioned supplement NW14_EGAT_supplement_fb58cad.zip (code, configuration, checkpoints, "
       "outputs, environment and dated protocol history; Tables 5-9 regenerate from it). See Response R6.",
}
out, n = [], 0
skip = False
for x in v6:
    if x.startswith("// Authors' status") and n == 2:
        continue  # comment 2's PARTLY line is replaced below
    out.append(x)
    if x.startswith("// Please"):
        n += 1
        if n in STATUS:
            out.append(f"// Authors' status, 10 Nov 2026 — {STATUS[n]}")

# appendix with full responses
RESP = [("R1", "Comment 1 (Section 2): mechanism-level comparison with [15]-[17]", "c1_related_work.md"),
        ("R2", "Comment 2 (Section 3): model definitions from the code", "c2_model_definitions.md"),
        ("R3", "Comment 3 (Section 4): training transition, evaluation epoch, fallback and baselines", "c3_trace.md"),
        ("R4", "Comment 4 (Section 5): complexity, latency, memory and communication", "c4_complexity.md"),
        ("R5", "Comments 5 and 7 (Sections 6.4 and 7): reconstruction of the comparisons and Tables 5-9",
         "review_v5_c5_c7.md"),
        ("R6", "Comment 6 (Section 6.5): versioned supplement", "c6_supplement.md")]
out += ["", "=" * 78, "Authors' responses to the review comments (10 Nov 2026)", "=" * 78,
        "Scripts and outputs named below are in the supplement archive (Response R6) under code/docs/paper/review/.", ""]
for tag, title, f in RESP:
    p = D / f
    body = p.read_text(encoding="utf-8") if p.exists() else "(not available)"
    body = re.sub(r"^# .*\n", "", body, count=1)  # drop the file's own title
    out += [f"Response {tag}. {title}", "-" * 78, body.strip(), ""]
(D / "egat-mappo-v7.txt").write_text("\n".join(out), encoding="utf-8")
print("comments", n, "lines", len(out))
