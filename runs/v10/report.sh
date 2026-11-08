#!/bin/bash
# composite reports for v10: fair (with li_h/yuan_h) a/b on test seeds, fresh a/b
cd /e/MESH/runs/v10
rep() { d=$1; s=$2; out=$3; H=$d/t0/$s/suite_heur.json; F=""
  for m in egat_h2 li yuan q_neutral li_h yuan_h; do F="$F $(echo $d/t{0,1,2,3,4}/$s/suite_$m.json)"; done
  python -m mesh.eval.composite $F $H $H $H $H $H --ours egat_h2 > $out 2>&1; }
cp t0/b/suite_heur.json t0/b/suite_heur.json 2>/dev/null
rep . a final/composite_a_egat_h2_fair.md & rep . b final/composite_b_egat_h2_fair.md &
rep fresh a fresh/final/composite_a_egat_h2.md & rep fresh b fresh/final/composite_b_egat_h2.md & wait
