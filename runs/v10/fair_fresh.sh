#!/bin/bash
# v10: Li+H / Yuan+H margin pick on validation (worst-case rule, as v8), their test evals, then fresh-topology (seeds 20-39) evals
cd /e/MESH; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
S="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"; E=runs/v10/events.log
echo "$(date +%F_%T) fair/fresh start (HEAD $(git rev-parse --short HEAD))" >> $E
{ echo ko; echo m0; for k in 0 1 2 3 4; do for b in li yuan q_neutral; do echo "$b $k"; done
  for b in li yuan; do for m in 0 0.25 0.5 1 2 4 8 16; do echo "$b $k $m"; done; done; done; } \
  | xargs -P 8 -L 1 python runs/v10/wc_val.py
python runs/v10/fair_pick.py li > runs/v10/fair_li.txt; python runs/v10/fair_pick.py yuan > runs/v10/fair_yuan.txt
ML=$(awk '/chosen margin/{print $3}' runs/v10/fair_li.txt); MY=$(awk '/chosen margin/{print $3}' runs/v10/fair_yuan.txt)
echo "$(date +%F_%T) fair margins li $ML yuan $MY" >> $E
mkdir -p runs/v10/fresh/t{0,1,2,3,4}/{a,b}
{ for k in 0 1 2 3 4; do for s in a b; do
    echo "test $s li_h li $ML $k"; echo "test $s yuan_h yuan $MY $k"
    for p in "li li -" "yuan yuan -" "q_neutral q_neutral -" "egat_h1 egat 1.0" "li_h li $ML" "yuan_h yuan $MY"; do echo "fresh $s $p $k"; done
  done; done; echo "fresh a heur - - 0"; echo "fresh b heur - - 0"; } | xargs -P 12 -L 1 bash -c '
  w=$0; s=$1; lab=$2; ck=$3; mg=$4; k=$5; S="--set airtime=domain --set mac_eff=0.65 --set contention=0.0"
  if [ $s = a ]; then extra=""; else extra="--set queue_preserve=true"; fi
  if [ $w = test ]; then suite=$([ $s = a ] && echo A,B,C || echo A,B); seeds="--seeds 20"; out=runs/v10/t$k/$s/suite_$lab
  else suite=A,B; seeds="--seeds 20 --seed-start 20"; out=runs/v10/fresh/t$k/$s/suite_$lab; fi
  if [ $lab = heur ]; then meth="--method m0 --method ko"; out=runs/v10/fresh/heur_$s
  else meth="--method $lab=runs/v10/ckpt/${ck}_s$k.pt"; [ $mg != - ] && meth="$meth --agent-set switch_margin=$mg"; fi
  python -m mesh.eval.suite --suite $suite $seeds $extra $S $meth --out $out.json > $out.log 2>&1 \
    || echo "$(date +%F_%T) eval FAIL $out" >> runs/v10/events.log'
for k in 0 1 2 3 4; do for s in a b; do cp runs/v10/fresh/heur_$s.json runs/v10/fresh/t$k/$s/suite_heur.json; done; done
echo "$(date +%F_%T) fair/fresh done" >> $E
